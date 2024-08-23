
# Introduction
This file provides status of readiness of halo models like llama3, grok1 etc. on MI3xx 


# Glossary
TPn: Tensor Parallel using n GPUs

# Artifacts

## Guideline:
1) small files check into [llm-dev](https://github.com/nod-ai/llm-dev)
2) large files upload to "sharkblob" -> "halo-models" container on Azure and put link to that in the table(s) below
3) Very large files, store on GPU server and note the name/location of/on the machine in table(s) below 

## TP1
Models           |     FP16        |   FP8           |     Q4_K 
:--------------: | :-------------: |:----------------:|:----------------:
llama3-8b | [mlir](https://github.com/nod-ai/llm-dev/blob/main/models/llama.8b/llama.8b.fp16.mlir) [gguf](https://sharkpublic.blob.core.windows.net/sharkpublic/llama_gguf/llama.8b.fp16.gguf) | |
llama3-70b | | |
llama3-405b |N/A | N/A|N/A
grok-1 |N/A |N/A |

## TP2
Models           |     FP16        |   FP8           |     Q4_K 
:--------------: | :-------------: |:----------------:|:----------------:
llama3-8b | | |
llama3-70b | | |
llama3-405b |N/A |N/A |
grok-1 |N/A | |


## TP4
Models           |     FP16        |   FP8           |     Q4_K 
:--------------: | :-------------: |:----------------:|:----------------:
llama3-8b | | |
llama3-70b | | |
llama3-405b |N/A | |
grok-1 | | |

## TP8
Models           |     FP16        |   FP8           |     Q4_K 
:--------------: | :-------------: |:----------------:|:----------------:
llama3-8b | | | 
llama3-70b | | |
llama3-405b | | |
grok-1 | | |


## Uploading to Sharkblob on Azure:
In browser, click on https://portal.azure.com/#@amdcloud.onmicrosoft.com/resource/subscriptions/8c190d1b-eb91-48d5-bec5-3e7cb7412e6c/resourceGroups/pdue-nod-ai-rg/providers/Microsoft.Storage/storageAccounts/sharkblobs/storagebrowser , click on "Blob-containers" and the click on "halo-models"

Or, use command line using instructions below:
You can get the account key for the storage account by clicking on "Storage Accounts" in Azure Services or searching "sharkblobs" in the top search bar. Then, click on sharkblobs. Then, on the left side bar, under Security + networking, click on "Access keys". Copy the account key from here and use in the following command

install azure with curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

az storage blob upload/download     --account-name sharkblobs     --container-name halo-models     --name <path_you_want_to_upload_in_azure, example: llama3_8b/tp1/llama.mlir>     --file <local_path_on_computer>     --account-key <key_retrieved_from_directions_above>

# AMD GPU Machines
[MI250-300](https://github.com/nod-ai/playbook/blob/main/HOWTO/access-mi250-mi300.md)
