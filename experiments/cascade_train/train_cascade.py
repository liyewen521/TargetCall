#!/usr/bin/env python3
"""Trainer for the cascade TINYX3 (partial warm-start + distill + CTC)."""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from model_cascade import Model  # noqa: E402


def fold_input(x):
    """[N,1,1,L] raw -> folded [N,1,3,T] (M[r,c]=[0,x,0][3c+r])."""
    n, _, _, length = x.shape
    k = (1 - length % 3) % 3
    if k:
        x = F.pad(x, (0, k))
        length += k
    assert (length + 2) % 3 == 0, (length,)
    tf = (length + 2) // 3
    y = F.pad(x, (1, 1))
    return y.reshape(n, 1, tf, 3).transpose(2, 3).contiguous(), tf


def init_cascades(model, state, mode):
    stats = []
    for name, mod in model.named_modules():
        if not (isinstance(mod, nn.Sequential) and name.startswith("depthwise")):
            continue
        convs = list(mod)
        total_rf = 1 + sum(c.kernel_size[-1] - 1 for c in convs)
        w = state.get(name + ".weight")
        sizes = [c.kernel_size[-1] for c in convs]
        for i, c in enumerate(convs):
            c.weight.data.zero_()
            k = c.kernel_size[-1]
            if mode == "truncate" and i == 0 and w is not None:
                W = w.shape[-1]
                start = (W - k) // 2
                c.weight.data.copy_(w[..., start:start + k])
            else:
                c.weight.data[..., 0, k // 2] = 1.0
        stats.append((name, total_rf, sizes, None if w is None else w.shape[-1]))
    return stats


def load_partial(model, state):
    r = model.load_state_dict(state, strict=False)
    return list(r.missing_keys), list(r.unexpected_keys)


class Chunks:
    def __init__(self, directory):
        self.chunks = np.load(os.path.join(directory, "chunks.npy"), mmap_mode="r")
        self.refs = np.load(os.path.join(directory, "references.npy"), mmap_mode="r")
        self.lens = np.load(os.path.join(directory, "reference_lengths.npy"), mmap_mode="r")
        self.n = self.chunks.shape[0]

    def batch(self, idx):
        x = np.asarray(self.chunks[idx], dtype=np.float32)
        r = np.asarray(self.refs[idx], dtype=np.int64)
        l = np.asarray(self.lens[idx], dtype=np.int64)
        return torch.from_numpy(x)[:, None, None, :], torch.from_numpy(r), torch.from_numpy(l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/example_data_dna_r9.4.1_v0")
    ap.add_argument("--weights", default=os.path.join(REPO, "checkpoints", "TINYX3.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "long2.pt"))
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eta-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    ap.add_argument("--distill", type=float, default=1.0)
    ap.add_argument("--init", choices=["truncate", "delta"], default="truncate")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-batch", type=int, default=16)
    ap.add_argument("--resume", default="")
    ap.add_argument("--no-teacher", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    state = torch.load(args.weights, map_location="cpu")
    model = Model().to(device)
    print("[model] params:", sum(p.numel() for p in model.parameters()))

    stats = init_cascades(model, state, args.init)
    print("[cascade-init] %s, %d cascades, rf=%s" % (args.init, len(stats),
          [s[1] for s in stats]))
    missing, unexpected = load_partial(model, state)
    print("[partial-load] missing %d, unexpected %d, conv0 %s"
          % (len(missing), len(unexpected), tuple(model.conv0.weight.shape)))

    teacher = None
    if not args.no_teacher:
        from inference import load_model
        teacher = load_model("TINYX3", device=args.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print("[teacher] original TINYX3 for distillation")

    data = Chunks(args.data)
    val = None
    val_idx = None
    val_dir = os.path.join(args.data, "validation")
    if os.path.isdir(val_dir):
        val = Chunks(val_dir)
        val_idx = np.random.default_rng(args.seed).integers(0, val.n, size=args.val_batch)
    print("[data] train", data.chunks.shape, "val", None if val is None else val.chunks.shape)

    ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "optim" in ck:
            opt.load_state_dict(ck["optim"])
        start_step = int(ck.get("step", 0))
        print("[resume] @ step", start_step)

    sched = None
    if args.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(args.steps - start_step, 1), eta_min=args.eta_min)

    best_path = os.path.splitext(args.out)[0] + "_best.pt"
    best_val = float("inf")

    def save(path, step):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                    "step": step, "args": vars(args)}, tmp)
        os.replace(tmp, path)

    def compute(xb, refs, ref_lens):
        xb, tf = fold_input(xb)
        xb = xb.to(device)
        probs = model(xb)
        logp = torch.log(probs.clamp_min(1e-8))
        in_lens = torch.full((xb.shape[0],), tf, dtype=torch.long, device=device)
        l_ctc = ctc(logp.squeeze(1).transpose(0, 1), refs.to(device), in_lens, ref_lens.to(device))
        if teacher is not None:
            with torch.no_grad():
                tp = teacher(xb)
            l_kl = F.kl_div(logp.reshape(-1, 5), tp.reshape(-1, 5), reduction="batchmean")
        else:
            l_kl = torch.zeros((), device=device)
        return l_ctc, l_kl

    def validate():
        if val is None:
            return None
        model.eval()
        with torch.no_grad():
            xb, refs, ref_lens = val.batch(val_idx)
            l_ctc, l_kl = compute(xb, refs, ref_lens)
        model.train()
        return float(l_ctc), float(l_kl)

    model.train()
    rng = np.random.default_rng(args.seed + start_step)
    run_ctc = run_kl = 0.0
    acc_n = 0
    for step in range(start_step + 1, args.steps + 1):
        idx = rng.integers(0, data.n, size=args.batch)
        xb, refs, ref_lens = data.batch(idx)
        l_ctc, l_kl = compute(xb, refs, ref_lens)
        loss = l_ctc + args.distill * l_kl

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()

        run_ctc += float(l_ctc)
        run_kl += float(l_kl)
        acc_n += 1
        if step % args.log_every == 0 or step == 1:
            print("[step %4d] ctc %.4f  kl %.4f  lr %.2e"
                  % (step, run_ctc / acc_n, run_kl / acc_n, opt.param_groups[0]["lr"]))
            run_ctc = run_kl = 0.0
            acc_n = 0

        if not torch.isfinite(loss):
            print("[warn] non-finite at", step)
            break

        if val is not None and step % args.val_every == 0:
            v = validate()
            score = v[0] + args.distill * v[1]
            print("[val   %4d] ctc %.4f  kl %.4f  score %.4f%s"
                  % (step, v[0], v[1], score, "  *best" if score < best_val else ""))
            if score < best_val:
                best_val = score
                save(best_path, step)

        if args.save_every and step % args.save_every == 0:
            save(args.out, step)

    save(args.out, step)
    if val is not None:
        print("[best] %s  score %.4f" % (best_path, best_val))
    print("[saved]", args.out)


if __name__ == "__main__":
    main()
