
# Introduction
This file provides status of readiness of halo models like llama3, grok1 etc. on MI3xx 


# Glossary
TPn: Tensor Parallel using n GPUs

# Artifacts

Guideline:
1) gguf files for llama3 8b and 70b upload to "sharkblob" -> "halo-models" container on Azure
2) 

## TP1
Models           |     FP16        |   FP8           |     Q4_K 
:--------------: | :-------------: |:----------------:|:----------------:
llama3-8b | [mlir](https://github.com/nod-ai/llm-dev/blob/main/models/llama.8b/llama.8b.fp16.mlir)| [gguf](https://sharkpublic.blob.core.windows.net/sharkpublic/llama_gguf/llama.8b.fp16.gguf) |
llama3-70b | | |
llama3-405b | | |
grok-1 | | |

## TP2



## TP4


## TP8
