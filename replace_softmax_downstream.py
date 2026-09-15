#!/usr/bin/env python3
"""Batch-update downstream model code to softmax semantics (unify with ONNX deploy).

Per file, four changes:
  1. import log_softmax -> softmax
  2. decode(): drop x.exp()  (forward now outputs probs directly)
  3. Decoder.forward(): log_softmax -> softmax
  4. ctc_label_smoothing_loss(): log the probs back for ctc_loss (keeps training correct)
"""
import sys
from pathlib import Path

CTC_DIR = Path('/Projects/ACCESS_IVFS/TargetCall/bonito/ctc')
FILES = [
    'model.py', 'modelb1.py', 'modelb1b2.py', 'modelb1x2.py',
    'modeltinynoskipx01.py', 'modeltinynoskipx011.py', 'modeltinynoskipx0111.py',
    'modeltinynoskipx1.py', 'modeltinynoskipx2.py', 'modeltinynoskipx3.py',
    'modeltinynoskipx4.py',
]

REPLACEMENTS = [
    # 1. import
    ('from torch.nn.functional import log_softmax, ctc_loss',
     'from torch.nn.functional import softmax, ctc_loss'),
    # 2. decode: drop exp
    ('x = x.exp().cpu().numpy().astype(np.float32)',
     'x = x.cpu().numpy().astype(np.float32)'),
    # 3. Decoder.forward
    ('return log_softmax(self.layers(x), dim=-1)',
     'return softmax(self.layers(x), dim=-1)'),
    # 4a. ctc_loss: log back (training only)
    ('ctc_loss(log_probs.to(torch.float32)',
     'ctc_loss(log_probs.log().to(torch.float32)'),
    # 4b. label smoothing: log back
    ('label_smoothing_loss = -((log_probs * weights.to(log_probs.device)).mean())',
     'label_smoothing_loss = -((log_probs.log() * weights.to(log_probs.device)).mean())'),
]

for fname in FILES:
    path = CTC_DIR / fname
    src = path.read_text()
    counts = []
    for old, new in REPLACEMENTS:
        n = src.count(old)
        counts.append(n)
        src = src.replace(old, new)
    path.write_text(src)
    print(f'{fname:28s} ' + ' '.join(f'{c}' for c in counts))
