layer_ref - per-layer int8 reference for input_000, onnx_models/tinyx3_ivfs_q8.onnx

Produced by experiments/cascade_train/dump_layers.py (mode=int): a true integer
pipeline (quantise -> conv -> requantise -> clamp) driven by the q8 annotation,
NOT a quantisation of the float activations. A correct compiler should match
these codes layer by layer, so the first node that differs is the bug.

    python experiments/cascade_train/dump_layers.py --idx 0

model        onnx_models/tinyx3_ivfs_q8.onnx
input        test_vectors/tinyx3_ivfs_folded/input_000.npy
input scale  41.968292236   (p99.9, 0.099% of samples saturate at 127)
prediction   91.08% frame agreement / 76.88% decoded identity vs the float model

Files
  000_input.i8.npy          int8 codes fed to /conv0/Conv
  NNN_<node>.i8.npy         int8 codes of the first 10 nodes
  NNN_<node>.f32.npy        the same tensors as float (onnxruntime)
  conv0_Conv.acc.i32.npy    int32 accumulator of /conv0/Conv, bias included
  all_layers.i8.npz         int8 codes of all 152 nodes, key = node name
  manifest.tsv              node -> bit depths -> files

Caveats
  - all_layers["Softmax"] is the output of the int8 pipeline. It is NOT the same
    as ../output_000.npy, which is the float golden quantised at the boundary
    (that one scores 97.85% identity). Compare compiler dumps against THIS file,
    and boundary vectors against ../output_000.npy.
  - the hidden layers use per-channel scales (12 or 20 values per tensor), so a
    per-tensor comparison will look "constant offset per channel". That is
    expected; check the per-channel pattern before calling it a divergence.
  - --mode ideal reproduces the old behaviour (quantise the float activation of
    every node). It is 99.70% / 97.85% against the float model, i.e. roughly one
    quantisation step better than any real int8 pipeline at every layer; do not
    use it to judge a compiler.

