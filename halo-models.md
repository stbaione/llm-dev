
# Introduction
This page is project tracker to get halo models like llama3, grok1 etc. working on one or more MI3xx using shark/iree. 

# November 20, 2024 Release Goals
- llama3.1 405B sharded across 8 MI300x GPUs producing correct numeical results (P0)
- llama3.1 405B sharded across 8 MI300x GPUs performant at level of vLLM PyTorch (Fused Ops Eager Mode) (P1)

(Note: Use llama3.1 8B or 70B to develop and test)

# Glossary
TPn: Tensor Parallel using n GPUs where a large tensor is sharded across multiple GPUs using sharktank and scatter/gather to/from GPUs is done in single MLIR

# Schedule
(Model is assumed to be llama3.1 in the following table, e.g. "8B FP8" means "llama3.1 8B FP8 model")
|Item                          | 10/18/24      | 10/25/24       | 11/1/24       | 11/8/24      | 11/15/24     |
|------------------------------|---------------|----------------|---------------|--------------|--------------|
| Machine and Storage          | two 8x MI300x SPX mode ensured working with how to use info added to [Nod AI Lab](https://confluence.amd.com/display/ENGIT/Nod.AI+Lab) @saienduri <br>(**Done:10/17**)|-Install 60TB storage on SharkMi300X **(Done:10/21)** <br>-setup one more 8x air-cooled MI300 machine (SharkMi300X-3) with 60TB added **(Done:10/24)**@saienduri |-Setup one more 8X MI300 air-cooled machine (SharkMi300X-4) with 60TB @saienduri <br>-Add 30 TB to each of SharkMi300X and SharkMi300X-3 @saienduri
| Sharktank Modeling | IREE-compilable 8B FP8 MLIR @dan garvey <br>(**Done:10/17**)| -verify numerics using quant-dequant on cpu vs run on MI300 for 8B FP8 @dan <br>-Get 8B 70B (Done:10/23) and 405B FP8 MLIR (ETA:10/24) and verify(CPU vs MI300) numerics for 70B @dan, <br>-Wire up Perlexity flow to run vmfb using iree-run-module (score too high, ETA:10/24) @archana, <br>-Debug 70B running OOM on 1 MI300 @kyle (ETA:10/25) <br>-Quantized sharding support (ETA:10/24) @Ian | Re-enerate and Verify MLIR without decomposition of SDPA for 8B, 70B, 405B for FP16 @kyle
| Sharding | 8 CPU core sharded FP16 numerically verified @boian. compilation [issue](https://github.com/iree-org/iree/issues/18878) ETA:10/25 | 8 GPU sharding for FP16 and FP8 compiling for MI300 @rob/@Ian sharding verified on CPU, compilation fails ETA: 10/25 | 8 GPU sharding for FP16 and FP8 numerics verified for MI300 @boian/@rob | 
| IREE codegeneration | | 8B FP16 attention ahead with dynamic shape generating valid vmfb ETA:10/24 @mahesh, FP8 Attention (use Intrinsic for FP8 effectively) (ETA:10/25) @stanley | Paged Attention @kunwar/@manupa | Perf Tuning (all)
| Inference Profiling| Tracy profile 8B FP16 w/ decoposition @kyle **(Done:10/17)** |Tracy profile for 8B FP8 w/ and w/o decomposition @kyle, <br>- Tracy profile 405B with 3 attention blocks w/ decomposition @Avi |
| Shortfin Serving | |llama3.1 8B FP16 iree compiled working using shortfin @xida (KV cache is messed up, needs help) |
| W/ Serving Inference Performance | | llama3.1 8B Fp16 iree compiled working using shortfin performance numbers @avi | Performance tuning for sharding @boin/@rob |
| Test Automation |-8B FP16 prefill attnhead, decode atttnhead, & full model IREE-compiled perf tests in sharktank CI @avi <br>-8B FP16 IREE-compiled numerics tested using Perlexity @archana |-8B FP8 prefill attnhead, decode atttnhead, full model IREE-compiled perf test in sharktank CI @avi <br>-8B FP8 IREE-compiled numerics tested using Perlexity @archana <br>-8 CPU core sharded 8B FP16 numeric test added @boian | 8 GPU sharded 8B FP8 test added @boin |
| Report dashboard| |  Show currently runnning all perf and numeric llama3.1 component and full model test reports on a page @saienduri |
| Release Packaging/testing | | Have a test release with 8B FP16 @chris | test release with 8B FP8 @chris

# Status-Numerics 

(MI300X GPU, SPX Mode)
|Item                                      | Generate MLIR | Compile to vmfb | IREE invocation | IREE numeric | Serving numeric |
|------------------------------------------|---------------|-----------------|-----------------|--------------|-----------------|
| llama3.1-8B-FP16-decomposed      |PASS [TP1 mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/llama8b_f16.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/llama8b_f16.gguf) [irpa](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/8b_f16.irpa)  |PASS [vmfb](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/8b_f16.vmfb) | PASS
| llama3.1-8B-FP16-decomposed-TP8 | PASS ([MLIR](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/llama3.1_8b_fp16_decomposed_tp8.mlir)) | PASS |
| llama3.1-70B-FP16-decomposed      |PASS [TP1 mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/llama70b_f16.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/llama70b_f16.gguf) [irpa](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/70b_f16.irpa) |PASS [vmfb](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/70b_f16.vmfb) | FAIL [OOM](https://github.com/iree-org/iree/issues/18864) |
| llama3.1-405B-FP16-decomposed  |PASS [TP1 mlir](https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/llm-dev/llama3_405b/llama3.1_405b_fp16_TP1.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_405b/llama405b_fp16.gguf)  |
| llama3.1-8B-FP8-decomposed   |PASS [TP1 mlir](https://sharkpublic.blob.core.windows.net/sharkpublic/dan/native_fp8_e4m3fnuz_llama3_8b.mlir) [irpa](https://sharkpublic.blob.core.windows.net/sharkpublic/dan/native_fp8_e4m3fnuz_llama3_8b.irpa) | Fails in iree, [patch](https://github.com/iree-org/iree/pull/18890)
| llama3.1-70B-FP8-decomposed  |PASS [TP1 mlir](https://sharkpublic.blob.core.windows.net/sharkpublic/dan/native_fp8_e4m3fnuz_llama3_70b.mlir) [irpa](https://sharkpublic.blob.core.windows.net/sharkpublic/dan/native_fp8_e4m3fnuz_llama3_8_70b.irpa) |Fails in iree, [patch](https://github.com/iree-org/iree/pull/18890)
| llama3.1-405B-FP8-decomposed |   |
| llama3.1-8B-FP16      |PASS [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/8b_f16_nondecomposed.mlir)   | Fails in iree, [patch](https://github.com/iree-org/iree/pull/18890)
| llama3.1-70B-FP16      |PASS [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/70b_f16_nondecomposed.mlir)   |Fails in iree, [patch](https://github.com/iree-org/iree/pull/18890)
| llama3.1-405B-FP16  |   |
| llama3.1-8B-FP8   |FAIL qkv must have same data type   |
| llama3.1-70B-FP8  |FAIL qkv must have same data type   |
| llama3.1-405B-FP8 |FAIL qkv must have same data type   |
| llama-toy-size-FP32-TP2-CPU | PASS | PASS |

# Status-Benchmark 

(MI300X GPU, SPX Mode, Time in ms)
|Item                                      | 10/25/24 | 11/1/24 | 11/8/24 | 11/15/24 | Target(vLLM-PyTorch)|
|------------------------------------------|----------|---------|---------|----------|---------------------|
| llama3.1-8B-FP16      |   |
| llama3.1-70B-FP16      |   |
| llama3.1-405B-FP16  |   |
| llama3.1-8B-FP8   |   |
| llama3.1-70B-FP8  |   |
| llama3.1-405B-FP8 |   |

# Issues
| category | issue link | assigned to | status |
|---|---|---|---|
|iree codegen | [18864](https://github.com/iree-org/iree/issues/18864)| unassigned | OOM for 70B |
|quark quantization | [QUARK-71](https://jira.xilinx.com/browse/QUARK-71) | unassigned | FP8 matmul should be used in attention|

# AMD GPU Machines
[MI300](https://confluence.amd.com/display/ENGIT/Nod.AI+Lab#Nod.AILab-MI300NodAIMachines)

# Test Reports
TBD: Sai please put link to nightly tests that test any of component or full model of llama3


## MLIR generation and Compilation
[Quantization](https://github.com/nod-ai/llm-dev/blob/main/Quantization.md)

export fp8:
```
python -m sharktank.examples.export_paged_llm_v1 --irpa-file=native_fp8_e4m3fnuz_llama3_8b.irpa --output-mlir native_fp8_e4m3fnuz_llama3_8b.mlir --no-fake-quant --bs=1 --attention-kernel=torch_sdpa
```

```
iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 <mlir file> -o <vmfb file>
```

## Evaluation tests
### Perplexity
Follow the steps [here](https://github.com/nod-ai/SHARK-Platform/blob/main/sharktank/sharktank/evaluate/README.md)

## Accessing sharkblobs on Azure:
In browser, click on [sharkblobs](https://portal.azure.com/#@amdcloud.onmicrosoft.com/resource/subscriptions/8c190d1b-eb91-48d5-bec5-3e7cb7412e6c/resourceGroups/pdue-nod-ai-rg/providers/Microsoft.Storage/storageAccounts/sharkblobs/storagebrowser) , then click on "Blob-containers" and the click on "halo-models"

Or, use command line by first installing az cli as:
```
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```
And then, get the account key for the storage account by clicking on "Storage Accounts" in Azure Services or searching "sharkblobs" in the top search bar. Then, click on sharkblobs. Then, on the left side bar, under Security + networking, click on "Access keys". Copy the account key from here and use in the following command
To upload:
```
az storage blob upload --account-name sharkblobs --container-name sharkblobs --name <azure path, example: halo-models/llama3_8b/tp1/llama.mlir> --file <local_path_on_computer> --account-key <key_retrieved_from_directions_above>
```

To download:
```
az storage blob download --account-name sharkblobs --container-name sharkblobs --name <azure path, example: halo-models/llama3_8b/tp1/llama.mlir> --file <local_path_on_computer> --account-key <key_retrieved_from_directions_above>
```

if you are downloading from "sharkpublic" then replace instructions above by sharkpublic and get your account access key for sharkpublic.
Example:
```
az storage blob download --account-name sharkpublic --container-name sharkpublic --name ian/llama8b_f16.gguf --file llama8b_f16.gguf --account-key <key string>
```

## Export With `sharktank` and Serve with `shortfin`:

Follow the steps [here](https://gist.github.com/stbaione/be38bfb214d990a4b765804223d6b948).

## Setup SGLang With Shortfin

Follow the steps [here](https://gist.github.com/stbaione/2843eced1b8c1042127bec3ca8774d9e)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Archive
## Status (Old)
(Note: Do not update this one)
|Models | compile | inference (SPX mode) | tracy |
|---|---|---|---|
|llama3.1-8b-FP16| PASS | prefill (1746 ms), decode (71.8 ms), [commands](https://gist.github.com/aviator19941/f10b5b7a7c3975de4363450b4d7ec68f) | [prefill](https://sharkpublic.blob.core.windows.net/sharkpublic/avi/llama8b_f16_prefill.tracy) [decode](https://sharkpublic.blob.core.windows.net/sharkpublic/avi/llama8b_f16_decode.tracy) |
|llama3.1-8b-Q4_1| PASS | prefill (1817 ms), decode (57.3 ms), [commands](https://gist.github.com/aviator19941/f10b5b7a7c3975de4363450b4d7ec68f) | [prefill](https://sharkpublic.blob.core.windows.net/sharkpublic/avi/llama8b_q4_1_prefill_v2.tracy) [decode](https://sharkpublic.blob.core.windows.net/sharkpublic/avi/llama8b_q4_1_decode_v2.tracy) |
|llama3.1-8b-Q4_k| PASS | | |
|llama3.1-70b-Q4_1| PASS | prefill (3543 ms), decode (213 ms), [commands](https://gist.github.com/aviator19941/79ee5afc39c225ec7469030320014fa3) | [prefill](https://sharkpublic.blob.core.windows.net/sharkpublic/avi/llama70b_q4_1_prefill.tracy) [decode](https://sharkpublic.blob.core.windows.net/sharkpublic/avi/llama70b_q4_1_decode.tracy) |
|llama2-7b-FP8| [FAIL](https://github.com/iree-org/iree/issues/18367)| | |
|grok-1-Q4_1| PASS | FAIL, out of memory | [prefill](https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/grok-1/grok-1-q4_1-rocm-prefill.tracy) [decode](https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/grok-1/grok-1-q4_1-rocm-decode.tracy) |


## Goals (Old)
(Note: No longer updated)
- [ ] Attention Compiler Work
  - [ ] Dynamic sequence length
  - [ ] Causal Masking
  - [ ] Flex attention compilation
- [ ] LLaMa 8b prefill and decode
  - [x] validated numerically correct 
  - [ ] export
  - [ ] compiled
  - [ ] benchmarked
  - [ ] replicate for larger variants
- [ ] Mixtral prefill and decode
  - [ ] validated numerically correct 
  - [ ] export
  - [ ] compiled
  - [ ] benchmarked
- [ ] Grok prefill and decode
  - [x] validated numerically correct 
  - [x] export
  - [x] compiled
  - [ ] benchmarked

## Tasks and Issues (Old)
(Scheduled for deprecation, move any relevant to Schedule table at top)
task      | owner      | status | next actions
:-------: | :--------: |:-------: | :------:
Sharded LLaMa | boian | In progress | Landing first sharded tests
Export/Compile LLaMa | kyle | blocked on `torch.aten.complex` | rob is authoring fix
LLaMa 8 prefill comparison | rob | layerwise comparison for prefill is normal | handing off tooling to Avi
LLaMa 8 decode comparison | avi | still investigating cause of numeric issue | reuse rob's tooling to investigate
FP8 quantized model | dan | finishing results from quark | following up with Giuseppe on new `fp8 quantization
Model evaluation tooling | archana | Perplexity CI nightly running in eager mode | working on getting perplexity with vmfb

## Artifacts (Old)
(Note: Update Schedule-Numerics table for llama3.1 artifacts instead of this table (10/20/2024 onwards))
### Guideline:
1) small files and MLIR files check into [llm-dev](https://github.com/nod-ai/llm-dev)
2) large files upload to [sharkblobs](https://portal.azure.com/#@amdcloud.onmicrosoft.com/resource/subscriptions/8c190d1b-eb91-48d5-bec5-3e7cb7412e6c/resourceGroups/pdue-nod-ai-rg/providers/Microsoft.Storage/storageAccounts/sharkblobs/storagebrowser) -> "halo-models" container on Azure and put link to that in the table(s) below
3) Very large files, store on GPU server and note the name/location of/on the machine in table(s) below 

Note: If a link to Azure sharkblob below gives you an error, either use az cli to download (see section Accessing sharkblobs on Azure) or click on [sharkblobs](https://portal.azure.com/#@amdcloud.onmicrosoft.com/resource/subscriptions/8c190d1b-eb91-48d5-bec5-3e7cb7412e6c/resourceGroups/pdue-nod-ai-rg/providers/Microsoft.Storage/storageAccounts/sharkblobs/storagebrowser) , then click on "Blob containers" and then navigate to the file manually and download it. 

### TP1 

Models           |     FP16        |   FP8           |     Q4_1         |    Q4_K       |    Attention IRs
:--------------: | :-------------: |:----------------:|:---------------:|:-------------:|:------------------:
llama2-7b | | [irpa](https://sharkblobs.blob.core.windows.net/dan/qdq_full_transpose.irpa) [mlir](https://sharkblobs.blob.core.windows.net/dan/batch_llama_v1.mlir) | | | [Attention IRs](https://github.com/nod-ai/llm-dev/tree/main/models/llama_attention_irs)
llama3-8b | [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/llama8b_f16.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/llama8b_f16.gguf) | [mlir](https://sharkpublic.blob.core.windows.net/sharkpublic/dan/native_fp8_e4m3fnuz_llama3_8b.mlir) [irpa](https://sharkpublic.blob.core.windows.net/sharkpublic/dan/native_fp8_e4m3fnuz_llama3_8b.irpa) | [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/llama8b_q4_1.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/llama8b_q4_1.gguf) | [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/llama8b_q4_k.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/llama8b_Q4_K.gguf) |
llama3-70b | [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/llama70b_f16.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/llama70b_f16.gguf) |  [mlir](https://sharkpublic.blob.core.windows.net/sharkpublic/dan/native_fp8_e4m3fnuz_llama3_70b.mlir) [irpa](https://sharkpublic.blob.core.windows.net/sharkpublic/dan/native_fp8_e4m3fnuz_llama3_8_70b.irpa)| [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/llama70b_q4_1.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/llama70b_q4_1.gguf) | [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/llama70b_q4_k.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/llama70b_q4_k.gguf) |
llama3-405b | [mlir](https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/llm-dev/llama3_405b/llama3.1_405b_fp16_TP1.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_405b/llama405b_fp16.gguf) | | [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_405b/llama405b_q4_1.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_405b/llama405b_q4_1.gguf) | [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_405b/llama405b_q4_k.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_405b/llama405b_q4_k.gguf) |
grok-1 | [mlir](https://sharkpublic.blob.core.windows.net/sharkpublic/dan/grok.mlir) [gguf](https://sharkpublic.blob.core.windows.net/sharkpublic/llm-dev/grok_1/grok-1-f16.gguf) |NA | [mlir](https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/grok-1/grok-1-q4_1-irpa.mlir) [gguf](https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/grok-1/grok-1-q4_1.gguf) | [gguf](https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/grok-1/grok-1-q4_k.gguf) |


## TP8
Models           |     FP16        |   FP8           |     Q4_1 |  Q4_K
:--------------: | :-------------: |:----------------:|:----------------:| :----------------:
llama3.1-8b | | | 
llama3.1-70b | | |
llama3.1-405b | | |
grok-1 | | |

