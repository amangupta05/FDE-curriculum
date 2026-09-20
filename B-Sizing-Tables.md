# Appendix B: Sizing Tables

What fits where. Every table here is an arithmetic model built from the formulas in Appendix A, not a measurement. Use it to decide what to try; measure before you quote a number to anyone.

**Units.** Weight sizes follow the chapters and model cards and use decimal gigabytes, 1 GB equal to $10^9$ bytes. KV-cache sizes use binary prefixes, 1 MB equal to 1024 KB, matching Chapter 2. The two differ by about 7 percent at the gigabyte scale, which matters only when a model is within a few hundred megabytes of fitting. When that happens, redo the arithmetic in one unit and say which.

**Formulas used.** Weights $M_w = N_{\text{lin}} \cdot (\text{bits per weight})/8 + M_{\text{emb}}$ (Chapter 12). KV cache $m_{kv} = 2 L n_{kv} d_{head} \cdot \text{bytes}$ per token (Chapters 2 and 4). Training state 16 bytes per parameter for mixed-precision AdamW, 10 with 8-bit moments (Chapter 4). Activation memory $c \cdot B T L d$ with $c$ about 36, falling to $2 B T L d + c B T d$ under gradient checkpointing (Chapter 4). LoRA trainable count $r(d_{\text{in}} + d_{\text{out}})$ per module (Chapter 7). Every entry in Appendix A.

## B.1 Assumed model shapes

Every row below is a representative open-weight configuration. Read the real numbers from the model's `config.json` before sizing a real run; vendors change $d_{ff}$, KV-head counts, and vocabularies between releases.

| Size | Representative model | $L$ | $d$ | $h$ | $n_{kv}$ | $d_{head}$ | $d_{ff}$ | $V$ | Embeddings | Parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B | Qwen2.5-0.5B | 24 | 896 | 14 | 2 | 64 | 4,864 | 151,936 | tied | about 0.49B |
| 1.5B | Qwen2.5-1.5B | 28 | 1,536 | 12 | 2 | 128 | 8,960 | 151,936 | tied | 1.54B |
| 3B | Qwen2.5-3B | 36 | 2,048 | 16 | 2 | 128 | 11,008 | 151,936 | tied | 3.09B |
| 7B | Qwen2.5-7B | 28 | 3,584 | 28 | 4 | 128 | 18,944 | 152,064 | untied | 7.62B |
| 8B | Llama-3-8B | 32 | 4,096 | 32 | 8 | 128 | 14,336 | 128,256 | untied | 8.03B |
| 14B | Qwen2.5-14B | 48 | 5,120 | 40 | 8 | 128 | 13,824 | 152,064 | untied | about 14.8B |
| 70B | Llama-3-70B | 80 | 8,192 | 64 | 8 | 128 | 28,672 | 128,256 | untied | about 70.6B |

The 1.5B, 7B, 8B, and 70B rows are the shapes Chapters 2, 7, 12, and 13 count in full. The 0.5B, 3B, and 14B rows are assumed; verify them against the configuration file.

## B.2 Inference memory, weights only

Weights only, in decimal GB. Add the KV cache from B.3, about 0.3 to 0.5 GB of CUDA context, and the engine's working set.

| Size | bf16, 2 bytes | int8, 1 byte | int4, 0.52 bytes | int4 with bf16 embedding and head |
|---|---|---|---|---|
| 0.5B | 0.98 | 0.49 | 0.25 | 0.45 |
| 1.5B | 3.1 | 1.5 | 0.80 | 1.2 |
| 3B | 6.2 | 3.1 | 1.6 | 2.1 |
| 7B | 15.2 | 7.6 | 4.0 | 5.6 |
| 8B | 16.1 | 8.0 | 4.2 | 5.7 |
| 14B | 29.6 | 14.8 | 7.7 | 10.0 |
| 70B | 141 | 70.6 | 36.7 | 39.8 |

The int4 column assumes 4.127 bits per weight, which is NF4 with double quantization or int4 at group size 128 with an fp16 scale. The last column is what actually happens: AWQ, GPTQ, and GGUF keep the embedding and output matrices at higher precision, and for a 150k vocabulary those two matrices are gigabytes. Chapter 12 measures Qwen2.5-7B at about 5.6 GB in AWQ and about 4.7 GB at Q4_K_M, where llama.cpp quantizes the embedding to Q6_K. GGUF k-quant ladder, in bits per weight: Q2_K 2.625, Q3_K 3.4375, Q4_K 4.5, Q5_K 5.5, Q6_K 6.5625.

## B.3 KV cache

Per token and per 8,192-token sequence, from $m_{kv} = 2 L n_{kv} d_{head} \cdot \text{bytes}$. Halve every figure for an 8-bit or FP8 cache.

| Size | $L$ | $n_{kv}$ | $d_{head}$ | bf16 per token | bf16 per 8k sequence | 64 concurrent 8k sequences |
|---|---|---|---|---|---|---|
| 0.5B | 24 | 2 | 64 | 12 KB | 96 MB | 6 GB |
| 1.5B | 28 | 2 | 128 | 28 KB | 224 MB | 14 GB |
| 3B | 36 | 2 | 128 | 36 KB | 288 MB | 18 GB |
| 7B | 28 | 4 | 128 | 56 KB | 448 MB | 28 GB |
| 8B | 32 | 8 | 128 | 128 KB | 1.0 GB | 64 GB |
| 14B | 48 | 8 | 128 | 192 KB | 1.5 GB | 96 GB |
| 70B | 80 | 8 | 128 | 320 KB | 2.5 GB | 160 GB |

Read the last column as the reason PagedAttention exists. It is also the reason grouped-query attention is not optional: Llama-3-8B with 32 KV heads instead of 8 would cost 512 KB per token and 256 GB for the same 64 users.

## B.4 Training memory

Model state only: weights, gradients, and optimizer. Add activations (about $2 B T d L$ bytes with gradient checkpointing, about $36 B T d L$ without, from Chapter 4), the logits tensor (about $10 B T V$ bytes unchunked), and about 0.5 GB of overhead, all from Appendix A's QLoRA budget.

| Size | Full, bf16 AdamW, 16 B/param | Full, 8-bit moments, 10 B/param | LoRA r16, bf16 base | QLoRA r16, int4 base | LoRA trainable params |
|---|---|---|---|---|---|
| 0.5B | 7.8 | 4.9 | 1.1 | 0.6 | about 8.8M |
| 1.5B | 24.7 | 15.4 | 3.4 | 1.4 | about 18M |
| 3B | 49.4 | 30.9 | 6.7 | 2.6 | about 30M |
| 7B | 122 | 76.2 | 15.9 | 6.3 | about 40M |
| 8B | 128 | 80.3 | 16.7 | 6.4 | about 42M |
| 14B | 237 | 148 | 30.7 | 11.1 | about 69M |
| 70B | 1,130 | 706 | 144 | 43.1 | about 207M |

Decimal GB. LoRA and QLoRA rows assume rank 16 on every linear layer, adapters and their AdamW state at 16 bytes per trainable parameter, and the embedding and output matrices left in bf16. With 8-bit paged moments the adapter term drops to 10 bytes per trainable parameter, which is where the 7B QLoRA run on 8 GB finds its last few hundred megabytes. The trainable counts are computed from the B.1 shapes; Chapter 7 derives the 7B figure and Chapter 10 the 1.5B figure.

**What the table leaves out, worked for 7B QLoRA at batch 1 and sequence 1,024.** Base 5.6 GB, adapters 0.40 GB with 8-bit moments, checkpointed activations $2 B T d L = 2 \times 1024 \times 3584 \times 28$ bytes, about 0.21 GB stored plus about 0.15 GB of recompute peak, and logits $10 B T V = 10 \times 1024 \times 152{,}064$ bytes, about 1.56 GB unchunked or about 0.3 GB chunked. With overhead the totals are about 8.6 GB unchunked and about 7.3 GB chunked: the first fails on 8 GB, the second fits. Doubling the sequence to 2,048 doubles both the activation and logits terms and the run stops fitting either way. The logits term is the one that surprises people, and it scales with the vocabulary, not the model.

## B.5 What fits on the RTX 4060, 8 GB

This restates the roadmap's compute plan (section 4) with the arithmetic above. The roadmap wins if the two ever disagree.

| Workload | Fits | Note |
|---|---|---|
| Inference, 7B to 8B at 4-bit | Yes, fully on GPU | 4.7 to 5.6 GB of weights, leaving 1 to 2 GB of cache, about 20k to 40k tokens at 56 KB each |
| Inference, 14B at 4-bit | Partly | About 10 GB, so half the layers offload to CPU; quality checks only, never benchmarks |
| Inference, 1.5B to 3B in bf16 with vLLM | Yes | Keep `gpu_memory_utilization` near 0.85 and the max context modest |
| Pretraining 125M on 300M tokens | Yes, slowly | Roughly 8 to 12 hours, an overnight run with checkpoints |
| LoRA SFT, 0.5B to 1.5B in bf16, seq 2048 | Yes | Batch 2 to 4 with gradient checkpointing |
| QLoRA SFT, 3B, seq 2048 | Yes | Batch 1 to 2 with gradient checkpointing |
| QLoRA SFT, 7B, seq 1024 | Borderline, about 7.3 GB | Unsloth, batch 1, paged 8-bit AdamW, chunked loss (Chapter 7) |
| QLoRA SFT, 7B, seq 2048 | No, about 8.0 GB | Shorten the sequence or rent |
| LoRA SFT, 7B in bf16 | No | Needs about 20 GB |
| Full fine-tune, 0.5B, 8-bit moments | Yes, about 4.9 GB of state | Leaves room for seq 1024 at batch 1 with a chunked loss |
| DPO or ORPO, 0.5B to 1.5B with QLoRA | Yes | The reference is the adapters-disabled base, so memory stays near SFT |
| GRPO, 0.5B | Yes, slowly | Generation dominates |
| GRPO, 1.5B and up | Rent | vLLM colocation for fast rollouts does not fit |
| Embedding or reranker fine-tune, 100M to 350M | Yes | Cached MNRL removes the batch-size cap |
| Quantizing 7B to AWQ or GPTQ | No | Calibration loads fp16 weights, 15.2 GB; use Kaggle's two T4s |

## B.6 GPU reference

| GPU | Memory | bf16 tensor peak, dense | Memory bandwidth | Notes |
|---|---|---|---|---|
| RTX 4060 Laptop | 8 GB GDDR6 | about 45 TFLOPS | about 256 GB/s | Ada, compute capability 8.9, bf16 and FP8; clocks fall under a laptop power cap |
| T4 | 16 GB GDDR6 | none; about 65 TFLOPS fp16 | about 320 GB/s | Turing, no bf16, no FlashAttention-2; two per free Kaggle session |
| L4 | 24 GB GDDR6 | about 121 TFLOPS | about 300 GB/s | Ada, FP8; bandwidth-poor for its memory size |
| A100 40 GB | 40 GB HBM2e | about 312 TFLOPS | about 1.56 TB/s | Ampere, no FP8 |
| A100 80 GB | 80 GB HBM2e | about 312 TFLOPS | about 2.0 TB/s | The default rental for anything that does not fit locally |
| H100 SXM 80 GB | 80 GB HBM3 | about 990 TFLOPS | about 3.35 TB/s | Hopper, FP8; the PCIe card is about 750 TFLOPS and about 2 TB/s |

Every figure in this table is approximately right; verify on the spec sheet before quoting it, and remember that a laptop GPU under sustained load runs below its published clocks. The ridge points $I^{*} = P/\beta$ that follow from these figures are about 176 FLOPs per byte on the 4060, 203 on a T4, and 156 on an A100, which is what Chapters 4 and 13 use.

## B.7 Decode upper bounds

Single-sequence decode is bounded by weight bytes read per token divided by bandwidth. For a 7B model at int4, taking 4.0 GB of weights and ignoring the KV cache and every kernel overhead:

| GPU | Bandwidth | 7B int4 upper bound | Expect in practice |
|---|---|---|---|
| RTX 4060 Laptop | about 256 GB/s | about 64 tokens per second | 28 to 40 in llama.cpp fully on GPU |
| T4 | about 320 GB/s | about 80 | 35 to 55 |
| L4 | about 300 GB/s | about 75 | 35 to 50 |
| A100 40 GB | about 1.56 TB/s | about 390 | 150 to 250 |
| A100 80 GB | about 2.0 TB/s | about 510 | 200 to 350 |
| H100 SXM 80 GB | about 3.35 TB/s | about 840 | 300 to 550 |

Every figure here is approximately right; verify on the spec sheet. Chapter 4 gives the fuller bound, $\beta / (M_{\text{weights}} + \bar{T} m_{kv})$, which also charges the cache read and puts real systems at 60 to 85 percent of it; a throttling laptop GPU with sampling and kernel overhead sits lower, which is what the right-hand column shows. The bound moves with the checkpoint, not just the GPU: at 4.7 GB for a Q4_K_M with a Q6_K embedding it falls to about 54 on the 4060, and at 5.6 GB for the AWQ build with bf16 embedding and head to about 46 (Chapter 12). Chapters 2 and 12 use the same model to bound Qwen2.5-1.5B in bf16 on the 4060 at about 83 tokens per second. Measure yours and use that number.

## B.8 Cost reference, September 2026

These are the prices the roadmap's Appendix D uses. They move every few months; verify before renting anything.

| Resource | Price | Status |
|---|---|---|
| RunPod Community Cloud, A100 80 GB | about 1.39 dollars per hour | verify |
| RunPod Community Cloud, H100 80 GB | about 2.89 dollars per hour | verify |
| RunPod Secure Cloud | roughly double Community Cloud | verify |
| Colab Pro | 9.99 dollars per month for 100 compute units; a T4 uses about 1.2 units per hour, an A100 40 GB about 5.4 | verify |
| Modal Starter | 30 dollars of free credits per month; A10G about 1.10 per hour, A100 40 GB about 3.73 per hour | verify |
| Kaggle Notebooks | free, about 30 GPU hours per week on two T4s or a P100, 12 hours per session | verify |
| AWS g5.xlarge spot, us-east-1 | about 0.44 dollars per hour | verify |
| AWS EKS control plane | 0.10 dollars per hour, plus NAT gateway and EBS; budget about 0.60 to 0.70 per cluster hour | verify |
| RTX 4060 Laptop | electricity only, a few cents per hour under load | verify |
| Hugging Face Hub, Weights and Biases, Langfuse Cloud, Vercel, GitHub Actions on public repositories | free tiers | verify |

Roadmap budget for the sixteen weeks: about 230 to 540 dollars in total, roughly 65 to 155 dollars per month, with a lean path near 100 to 150 dollars. Two thirds of it is frontier API tokens for synthetic data, judges, and agent evaluations, not GPU rental. Three rules keep it there: a forgotten GPU instance is the largest single risk, so set spend limits and auto-stop; EKS bills its control plane, NAT gateway, and EBS while idle, so learn on kind and deploy in short planned sessions; and a 50-step smoke test at full sequence length and batch size before every long run costs minutes and saves nights.

## B.9 Rules of thumb

- bf16 weights cost 2 bytes per parameter, int8 1 byte, int4 about 0.52. Training under mixed-precision AdamW costs 16 bytes per trainable parameter, 10 with 8-bit moments.
- A model you can serve at 4-bit is roughly a model whose parameter count in billions is under 1.6 times your VRAM in gigabytes, before the KV cache.
- The KV cache, not the weights, is what limits concurrency. Compute it per token first, then multiply by context and users.
- Decode speed is bandwidth divided by bytes read per token. If a quoted tokens-per-second number exceeds that bound, it is wrong or it is batched.
- A large untied vocabulary adds gigabytes that no weight quantization removes. Compute the embedding term separately, every time.
- Utilization dominates the self-hosting cost model. A GPU at 30 percent load costs three times its benchmark cost per request.
