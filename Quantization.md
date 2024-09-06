# LLM Model Generation

A quick model generation guide so that you can obtain a base `fp16` model and follow the process to quantize to any known scheme:

## Base Model Sources

We can source some based gguf files below and generate alternative

|Model Variant|Size|Source|
| ------------- | ---- | ---- |
| LLaMa3|8b|[link](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/llama8b_f16.gguf)|
| LLaMa3|70b|[link](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/llama70b_f16.gguf)|
| LLaMa3|405B| |
| Grok-1|365B| |


## Quantization

We use the GGML tooling to generate our variable ggml toolings. Setup is below

```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cmake -B build -S .
cd build
cmake --build . -j 32
```

Then following the guide below to generate quantization variations

```
llama.cpp/build/bin/llama-quantize --pure <input_gguf> <output_gguf> <format> <threads>
```

For formats we should target  `Q_4_1` and `Q_4_K`.

## MLIR Generation

Once we have any particular gguf file we can export the representative IR

```
python3 -m sharktank.examples.export_paged_llm_v1 --gguf-file <input_gguf> --output-mlir <output-mlir>
```
