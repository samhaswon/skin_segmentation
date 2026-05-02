# cpp_demo

Minimal standalone C++ demo for the segmentation sessions in this repo.

It uses:

- hard-coded local model paths
- ONNX Runtime from `../onnxruntime-linux-x64-gpu-1.25.0`
    - https://github.com/microsoft/onnxruntime/releases
- OpenCV from the local `../opencv` submodule
- a simple terminal UI for model selection, GPU selection, chunk refinement, and image path input

## Build

```bash
cmake -S cpp_demo -B build/cpp_demo
cmake --build build/cpp_demo -j
```

## Run

```bash
./build/cpp_demo/cpp_demo
```

Output masks are written to `./output/` from the directory where you launch the executable.
