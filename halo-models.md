
# Introduction
This file provides status of readiness of halo models like llama3, grok1 etc. on MI3xx 


# Glossary
TPn: Tensor Parallel using n GPUs

# Artifacts

## Guideline:
1) small files and MLIR files check into [llm-dev](https://github.com/nod-ai/llm-dev)
2) large files upload to [sharkblobs](https://portal.azure.com/#@amdcloud.onmicrosoft.com/resource/subscriptions/8c190d1b-eb91-48d5-bec5-3e7cb7412e6c/resourceGroups/pdue-nod-ai-rg/providers/Microsoft.Storage/storageAccounts/sharkblobs/storagebrowser) -> "halo-models" container on Azure and put link to that in the table(s) below
3) Very large files, store on GPU server and note the name/location of/on the machine in table(s) below 

## TP1
Models           |     FP16        |   FP8           |     Q4_K         |    Q1_K
:--------------: | :-------------: |:----------------:|:---------------:|:-------------:
llama3-8b | [mlir](https://github.com/nod-ai/llm-dev/raw/main/models/llama.8b/llama.8b.fp16.mlir) [gguf](https://sharkpublic.blob.core.windows.net/sharkpublic/llama_gguf/llama.8b.fp16.gguf) | | [mlir](https://github.com/nod-ai/llm-dev/raw/main/models/llama.8b/llama.8b.q4_1.mlir) [gguf](https://sharkpublic.blob.core.windows.net/sharkpublic/llama_gguf/llama.8b.q4_1.gguf) | [mlir](https://github.com/nod-ai/llm-dev/raw/main/models/llama.8b/llama.8b.q4_k.mlir) [gguf](https://sharkpublic.blob.core.windows.net/sharkpublic/llama_gguf/llama.8b.q4_k.gguf)
llama3-70b | | | |
llama3-405b |NA | NA|NA |
grok-1 |NA |NA | |

## TP2
Models           |     FP16        |   FP8           |     Q4_K 
:--------------: | :-------------: |:----------------:|:----------------:
llama3-8b | | |
llama3-70b | | |
llama3-405b |NA |NA |
grok-1 |NA | |


## TP4
Models           |     FP16        |   FP8           |     Q4_K 
:--------------: | :-------------: |:----------------:|:----------------:
llama3-8b | | |
llama3-70b | | |
llama3-405b |NA | |
grok-1 | | |

## TP8
Models           |     FP16        |   FP8           |     Q4_K 
:--------------: | :-------------: |:----------------:|:----------------:
llama3-8b | | | 
llama3-70b | | |
llama3-405b | | |
grok-1 | | |

## MLIR generation and Compilation
[Quantization](https://github.com/nod-ai/llm-dev/blob/main/Quantization.md)
```
iree-compile --iree-hal-target-backends=rocm <mlir file>
```

## Uploading to Sharkblobs on Azure:
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
# Assignments
task      | owner      | status
:-------: | :--------: |:-------:
Export very large gguf and mlir files for large models | Kyle | 
iree-compile on mlir of parts of model, and file issues | Archana | 
Numerical issues for any component, tarci profile | Avi |
Numerical correctness of 70b FP8 (gets from AMD quark team) vs a gold provided by quork | Dan |
Upload gguf and mlir files | Rob | 
Fix crash for the issue Rob raised for llama3-8b | Mahesh | 
Causal (Masked) Attention Support for torch to linalg | Zach |
Causal (Masked) Attention Support for gpu codegen | Rohan |



# AMD GPU Machines
[MI250-300](https://github.com/nod-ai/playbook/blob/main/HOWTO/access-mi250-mi300.md)
