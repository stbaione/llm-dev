
# Introduction
This file provides status of readiness of halo models like llama3, grok1 etc. on MI3xx 


# Glossary
TPn: Tensor Parallel using n GPUs

# Tasks and Issues
task      | owner      | status
:-------: | :--------: |:-------:
Export very large gguf and mlir files for large models | Kyle | 
iree-compile on mlir of parts of model, and file issues | Archana | [Attention IRs](https://github.com/nod-ai/llm-dev/tree/main/models/llama_attention_irs)
Numerical issues for any component, tracy profile | Avi |
Numerical correctness of 70b FP8 (gets from AMD quark team) vs a gold provided by quork | Dan |
Upload gguf and mlir files | Rob/Dan/Ian | In progress
Fix crash for the issue Rob raised for llama3-8b | Mahesh | [18353](https://github.com/iree-org/iree/issues/18353) [18229](https://github.com/iree-org/iree/issues/18229)
Causal (Masked) Attention Support for torch to linalg | Zach |
Causal (Masked) Attention Support for gpu codegen | Rohan/Stan | ETA 8/29
Non-Causal/Causal Attention IR lowering codegen | Ian Wood | 
Fix issue with export of large constants in exported MLIR | Stella | This is causing [18353](https://github.com/iree-org/iree/issues/18353)
RotaryEmbeddingLayer support static_tables=False | Vivek | [](https://github.com/nod-ai/sharktank/issues/156)

# Artifacts

## Guideline:
1) small files and MLIR files check into [llm-dev](https://github.com/nod-ai/llm-dev)
2) large files upload to [sharkblobs](https://portal.azure.com/#@amdcloud.onmicrosoft.com/resource/subscriptions/8c190d1b-eb91-48d5-bec5-3e7cb7412e6c/resourceGroups/pdue-nod-ai-rg/providers/Microsoft.Storage/storageAccounts/sharkblobs/storagebrowser) -> "halo-models" container on Azure and put link to that in the table(s) below
3) Very large files, store on GPU server and note the name/location of/on the machine in table(s) below 

Note: If a link to Azure sharkblob below gives you an error, either use az cli to download (see section Accessing sharkblobs on Azure) or click on [sharkblobs](https://portal.azure.com/#@amdcloud.onmicrosoft.com/resource/subscriptions/8c190d1b-eb91-48d5-bec5-3e7cb7412e6c/resourceGroups/pdue-nod-ai-rg/providers/Microsoft.Storage/storageAccounts/sharkblobs/storagebrowser) , then click on "Blob containers" and then navigate to the file manually and download it. 

## TP1
Models           |     FP16        |   FP8           |     Q4_K         |    Q4_1       |    Attention IRs
:--------------: | :-------------: |:----------------:|:---------------:|:-------------:|:------------------:
llama2-7b | | [irpa](https://sharkblobs.blob.core.windows.net/dan/qdq_full_transpose.irpa) [mlir](https://sharkblobs.blob.core.windows.net/dan/batch_llama_v1.mlir) | | | [Attention IRs](https://github.com/nod-ai/llm-dev/tree/main/models/llama_attention_irs)
llama3-8b | [mlir](https://github.com/nod-ai/llm-dev/raw/main/models/llama.8b/llama.8b.fp16.mlir) [gguf](https://sharkpublic.blob.core.windows.net/sharkpublic/llama_gguf/llama.8b.fp16.gguf) | | [mlir](https://github.com/nod-ai/llm-dev/raw/main/models/llama.8b/llama.8b.q4_1.mlir) [gguf](https://sharkpublic.blob.core.windows.net/sharkpublic/llama_gguf/llama.8b.q4_1.gguf) | [mlir](https://github.com/nod-ai/llm-dev/raw/main/models/llama.8b/llama.8b.q4_k.mlir) [gguf](https://sharkpublic.blob.core.windows.net/sharkpublic/llama_gguf/llama.8b.q4_k.gguf) |
llama3-70b | | | | |
llama3-405b |NA | NA|NA | |
grok-1 |NA |NA | | |

## TP2
Models           |     FP16        |   FP8           |     Q4_K     |  Q4_1
:--------------: | :-------------: |:----------------:|:----------------: | :----------------:
llama3-8b | | |
llama3-70b | | |
llama3-405b |NA |NA |
grok-1 |NA | |


## TP4
Models           |     FP16        |   FP8           |     Q4_K   |  Q4_1
:--------------: | :-------------: |:----------------:|:----------------:| :----------------:
llama3-8b | | |
llama3-70b | | |
llama3-405b |NA | |
grok-1 | | |

## TP8
Models           |     FP16        |   FP8           |     Q4_K |  Q4_1
:--------------: | :-------------: |:----------------:|:----------------:| :----------------:
llama3-8b | | | 
llama3-70b | | |
llama3-405b | | |
grok-1 | | |

## MLIR generation and Compilation
[Quantization](https://github.com/nod-ai/llm-dev/blob/main/Quantization.md)
```
iree-compile --iree-hal-target-backends=rocm <mlir file>
```

## Accessing sharkblobs on Azure:
In browser, click on [sharkblobs](https://portal.azure.com/#@amdcloud.onmicrosoft.com/resource/subscriptions/8c190d1b-eb91-48d5-bec5-3e7cb7412e6c/resourceGroups/pdue-nod-ai-rg/providers/Microsoft.Storage/storageAccounts/sharkblobs/storagebrowser) , then click on "Blob-containers" and the click on "halo-models"

Or, use command line by first installing az cli as:
```
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```
And then, get the account key for the storage account by clicking on "Storage Accounts" in Azure Services or searching "sharkblobs" in the top search bar. Then, click on sharkblobs. Then, on the left side bar, under Security + networking, click on "Access keys". Copy the account key from here and use in the following command
To upload:
```
az storage blob upload --account-name sharkblobs --container-name halo-models --name <azure path, example: llama3_8b/tp1/llama.mlir> --file <local_path_on_computer> --account-key <key_retrieved_from_directions_above>
```

To download:
```
az storage blob download --account-name sharkblobs --container-name halo-models --name <azure path, example: llama3_8b/tp1/llama.mlir> --file <local_path_on_computer> --account-key <key_retrieved_from_directions_above>
```


# AMD GPU Machines
[MI250-300](https://github.com/nod-ai/playbook/blob/main/HOWTO/access-mi250-mi300.md)
