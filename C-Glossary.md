# Appendix C: Glossary

Every term the handbook defines, alphabetical, with the chapter that introduces it. Where a term has a different meaning outside this handbook, the definition here is the one the chapters use.

## A

**Activation checkpointing** (4). Storing only each block's inputs during the forward pass and recomputing the interior during the backward pass, trading about a third more compute for several times less activation memory. Also called gradient checkpointing.

**Activation memory** (4). The intermediate tensors the backward pass needs, scaling with batch, sequence length, layers, and width; at long sequences it exceeds the weights.

**AdamW** (3). The default transformer optimizer, keeping a first moment (momentum) and a second moment (per-parameter scaling) per parameter, with weight decay applied separately from the gradient.

**Adapter** (7). The small trainable matrix pair LoRA adds beside a frozen weight; a file of tens of megabytes that can be swapped, stacked, or merged.

**Advantage** (8). A response's reward relative to a baseline. In GRPO it is the reward minus the group mean, divided by the group standard deviation.

**Agent** (21). A system in which the model decides the next action and the runtime executes it, as opposed to a workflow where code decides.

**Agentic loop** (21). The cycle of plan, choose tool, check policy, execute, observe, checkpoint, repeat, with explicit stop conditions and budgets owned by the runtime.

**Alpha, LoRA** (7). The scaling constant applied to the low-rank update, conventionally set to twice the rank; the effective scale is alpha divided by rank.

**Arithmetic intensity** (4). FLOPs performed per byte read from memory; it places a kernel on the compute-bound or memory-bound side of the roofline.

**Attack success rate** (19). The fraction of red-team attempts in a category that achieve their goal, measured before and after each defense.

**Attention** (2). The operation by which each position gathers information from earlier positions through query, key, and value projections; the only operation in a transformer that mixes across positions.

**Audience binding** (22). Issuing an OAuth token valid only for one named resource server, so a token stolen by another server cannot be replayed.

**AWQ** (12). Activation-aware weight quantization, which scales up the small fraction of weight channels that matter most to activations before quantizing, preserving their precision.

## B

**Bake window** (17). The period a canary must stay healthy on online metrics before it is promoted.

**Base model** (7). A model trained only on next-token prediction, without instruction tuning; it completes text rather than following a conversation.

**bf16** (3). A 16-bit float with fp32's exponent range and fewer mantissa bits; needs no loss scaling, supported on Ada and later, not on Turing.

**Bi-encoder** (9). An embedding model that encodes query and passage independently, making vector search over millions of passages possible.

**Blameless postmortem** (20). An incident write-up covering timeline, impact, detection, root cause, contributing factors, and action items, focused on systems rather than individuals.

**Block, KV** (13). The fixed-size unit in which PagedAttention allocates KV cache, addressed through a per-sequence block table.

**BM25** (9). A lexical ranking function with saturation parameter k1 and length-normalization parameter b, used alongside vector search in hybrid retrieval.

**Bootstrap interval** (11). A confidence interval built by resampling evaluation items with replacement thousands of times and taking percentiles of the recomputed metric.

**BPE** (1). Byte-pair encoding: a tokenizer trained by repeatedly merging the most frequent adjacent pair into a new token.

**Bradley-Terry model** (8). The model of pairwise preference in which the probability that one response is preferred is the sigmoid of the reward difference.

**Burn rate** (20). How fast an error budget is being consumed relative to its window; multi-window burn-rate alerts page on fast burn and ticket on slow burn.

**Bytes per token** (1). How many bytes of text one token carries on average; a property of the tokenizer and the data together, and the reason the same text costs more in some languages.

## C

**Canary** (17). Routing a small, stratified share of traffic to a new version to detect regressions before full promotion.

**Cascade** (16). A routing policy that tries the cheap model first and escalates to the expensive one when a confidence signal is weak.

**Catastrophic forgetting** (7). Degradation of general capability caused by fine-tuning on a narrow task; detected with a regression suite.

**Causal mask** (2). Setting scores for future positions to negative infinity before the softmax, so they receive exactly zero weight; the mechanism that makes a decoder autoregressive.

**Chat template** (1, 7). The exact token layout of roles and turns a model was trained on; using the wrong one degrades quality silently.

**Checkpoint** (3, 21). For training, the saved model, optimizer, scheduler, step, data position, and RNG state needed to resume without a loss bump. For an agent, the persisted state after each step that allows a crashed run to resume.

**Chinchilla** (6). The scaling result that, for a fixed training compute budget, loss is minimized at roughly twenty tokens per parameter.

**Chunked prefill** (13). Splitting a long prompt's prefill across several steps so that other users' decode latency stays stable.

**Circuit breaker** (20). A component that stops calling a failing backend after a threshold of errors and falls back, rather than queueing doomed requests.

**Cohen's kappa** (11). Chance-corrected agreement between two raters; above about 0.6 is usable and above 0.8 is strong.

**Completion-only loss** (7). Computing loss on the response tokens while masking the prompt, so the model learns the answer rather than the schema.

**Contamination** (10, 11). Overlap between training data and evaluation data, which invalidates a score; detected by n-gram overlap.

**Context window** (2, 21). The number of tokens a model can attend over at once; quality degrades before it is full, which is why agents compact proactively.

**Continuous batching** (13). Re-forming the serving batch at every decode step as requests arrive and finish, instead of waiting for a whole static batch.

**Contrastive learning** (9). Training embeddings by pulling matched query and passage together and pushing unmatched pairs apart.

**Cross-encoder** (9). A reranker that reads query and passage together and scores their relevance; far more accurate and far slower than a bi-encoder.

**Cross-entropy** (3). The training loss, the negative log probability assigned to the correct next token; at initialization it equals the natural log of the vocabulary size.

**Curation** (17). Selecting the production traces worth human review and eventual training, ranked by signals such as oracle failure and negative feedback.

## D

**Data flywheel** (17). The loop from production traces through curation, review, versioned data, training, an evaluation gate, and canary rollout, back to production.

**DDP** (6). Distributed Data Parallel: every GPU holds a full model copy and its own batch shard, with gradients averaged by all-reduce each step.

**Decode** (2, 13). Generating one token at a time from the KV cache; memory-bound, because each step reads every weight.

**Decontamination** (10). Removing training examples that overlap evaluation items before training.

**Device plugin** (14). The Kubernetes component that advertises GPUs as a schedulable resource so pods can request them.

**Discovery** (25). The engagement phase that establishes outcome, workflow, stakeholders, data, the definition of correct, constraints, and kill criteria.

**Distillation** (10). Training a small student on a large teacher's outputs; answer distillation uses final outputs, rationale distillation includes the reasoning.

**Double quantization** (7, 12). Quantizing the quantization constants themselves, which saves about 0.373 bits per weight in QLoRA.

**DPO** (8). Direct Preference Optimization: training on preference pairs against a frozen reference, with no reward model and no RL loop.

**Drift** (18). A change over time in inputs, outputs, routing mix, or the correct answer itself; measured with the population stability index or a two-sample test.

**Durable execution** (21). Running long tasks so that a crash resumes from the last checkpoint rather than restarting, with idempotency keys making retried writes safe.

## E

**Effective batch** (3). Micro-batch size times gradient accumulation steps times GPU count; the learning rate should be chosen for this, not the micro-batch.

**Embedding** (2, 9). The vector representing a token, or the vector representing a whole text for retrieval.

**Error budget** (20). The allowed shortfall from a service level objective within its window; exhausting it triggers a change freeze.

**Eval-as-CI** (18). Running evaluation suites automatically on pull requests and reporting metric deltas with confidence intervals.

**Execution accuracy** (7, 11). Whether generated SQL returns the same result set as the gold query; the oracle used throughout the text-to-SQL thread.

## F

**False negative, retrieval** (9). A passage mined as a hard negative that is actually relevant; training on it hurts recall.

**FinOps** (20). Managing unit economics of a system: cost per query, per tenant, per feature, with allocation and anomaly detection.

**FlashAttention** (2). An IO-aware attention kernel that tiles the computation and never materializes the full attention matrix, saving memory rather than FLOPs.

**FLOPs, training** (4). Approximately six per parameter per token, covering the forward and backward passes.

**fp16** (3). A 16-bit float with a narrow exponent range; requires dynamic loss scaling to avoid gradient underflow.

**FP8** (12). An 8-bit float in E4M3 or E5M2 layout, supported on Ada and Hopper, near-lossless for weights.

**FSDP** (6). Fully Sharded Data Parallel: parameters, gradients, and optimizer states are sharded across GPUs and gathered as needed, cutting per-GPU memory.

## G

**Game day** (20). A scheduled exercise that injects a failure against a written hypothesis and measures detection and recovery.

**Golden set** (11, 17). A frozen, stratified, expert-verified evaluation set that is never trained on.

**GGUF** (12). The llama.cpp single-file model format carrying weights, tokenizer, and metadata at a chosen quantization level.

**GPTQ** (12). Quantization that sweeps columns one at a time and compensates the induced error in the remaining weights using second-order information.

**Gradient accumulation** (3). Summing gradients over several micro-batches before an optimizer step, to reach a large effective batch on a small GPU.

**Gradient clipping** (3). Rescaling the gradient when its global norm exceeds a threshold, preventing one bad batch from destabilizing training.

**GQA** (2). Grouped-query attention: several query heads share one key-value head, shrinking the KV cache and raising serving concurrency.

**GRPO** (8). Group Relative Policy Optimization: sample a group of responses, score each with a program, and use group-normalized advantages, with no value network.

**Guardrail** (19). A control applied before or after the model call: input classifier, output validator, schema check, or tool allowlist.

## H

**Hard negative** (9). A non-relevant passage that looks relevant, mined to sharpen contrastive training.

**Helm** (14). The Kubernetes package manager; a chart of templates plus a values file per environment.

**HNSW** (9). A graph vector index with layered links, giving approximate nearest-neighbor search in logarithmic time.

**HPA** (14). The Kubernetes HorizontalPodAutoscaler, which scales replicas on a metric; CPU is a poor signal for a GPU model server.

**Human-in-the-loop** (21). Pausing an agent for approval on consequential actions, showing the diff, the reasoning, the risk score, and the undo plan.

## I

**Idempotency key** (21). A unique key that makes a repeated write apply exactly once, which is what makes retries and resumption safe.

**InfoNCE** (9). The contrastive loss over in-batch negatives with a temperature; Multiple Negatives Ranking Loss is its practical form.

**Indirect prompt injection** (19). Instructions hidden in content the model reads, such as a retrieved document or a tool result.

**Instruct model** (7). A model already fine-tuned to follow a conversational format; the right starting point for a chat-style task.

## K

**KEDA** (14). An event-driven autoscaler that scales on external metrics such as request queue depth.

**Kill criteria** (25). The conditions under which a customer would stop the project, established during discovery.

**kind** (14). A tool that runs a Kubernetes cluster inside Docker, used to learn every object locally at zero cost.

**Knee** (13). The point on the throughput-against-latency curve where latency begins rising steeply; the operating point to size for.

**KV cache** (2, 4). The stored keys and values of past tokens that make decoding fast; its size per token is twice the layers times the KV heads times the head dimension times the bytes per element.

## L

**Lineage** (17). The recorded chain from a production trace to a dataset revision to a training run to a registered model version.

**LLM-as-judge** (11). Scoring outputs with a model against a rubric; must be calibrated against human labels and tested for position and length bias.

**lm-eval** (11). The EleutherAI evaluation harness that runs standardized tasks reproducibly, with log-likelihood and generative scoring modes.

**Load shedding** (20). Rejecting low-priority work early under overload so the rest of the traffic is served correctly.

**LoRA** (7). Low-Rank Adaptation: freezing the base weights and training a low-rank update beside each chosen matrix.

**Loss scaling** (3). Multiplying the loss before the backward pass and dividing the gradients after, so fp16 gradients do not underflow.

## M

**Matryoshka** (9). Training embeddings so that leading prefixes of the vector are themselves usable embeddings, allowing cheap truncation.

**MCP** (22). The Model Context Protocol: the standard by which a host connects to servers exposing tools, resources, and prompts.

**Merging** (7). Folding a LoRA adapter into the base weights to produce a standalone checkpoint for quantization and export.

**MFU** (4). Model FLOPs utilization: achieved training FLOPs divided by the hardware peak; small models are memory-bound and land well below peak.

**MinHash** (6, 10). A technique that estimates Jaccard similarity from hashed shingles, used to find near-duplicate documents at scale.

**Mixed precision** (3). Running forward and backward in 16-bit while keeping fp32 master weights and optimizer states.

**Model collapse** (17). Degradation caused by training on a system's own unreviewed outputs over successive flywheel cycles.

**MRR** (9). Mean reciprocal rank: the average of one divided by the rank of the first relevant result.

## N

**nDCG** (9). Normalized discounted cumulative gain: a graded, position-discounted ranking metric normalized by the ideal ordering.

**NF4** (7, 12). 4-bit NormalFloat, whose levels are placed at quantiles of a normal distribution to match the distribution of neural network weights.

## O

**On-policy data** (8). Preference pairs sampled from the model being improved, which train better than pairs from an unrelated model.

**Oracle** (11, 17). A programmatic check that determines correctness without a human or a judge, such as executing SQL and comparing result sets.

**OWASP Top 10 for LLM Applications** (19). The risk checklist enterprise security teams use to review an LLM system.

## P

**Packing** (1, 6, 7). Concatenating examples into fixed-length training windows to avoid padding; requires per-example masking to prevent cross-contamination.

**PagedAttention** (13). Storing the KV cache in fixed blocks with per-sequence block tables, eliminating fragmentation and enabling prefix sharing.

**Paired comparison** (11). Comparing two systems item by item on the same evaluation set, which is far more powerful than comparing two independent intervals.

**Pass consistency** (21). The rate at which all repeated runs of a scenario succeed, as opposed to the average success rate.

**Perplexity** (3). The exponential of cross-entropy loss; comparable only between models sharing a tokenizer.

**PKCE** (22). Proof Key for Code Exchange, which protects the OAuth authorization code flow for public clients.

**Prefill** (2, 13). The single forward pass over the prompt that fills the KV cache; compute-bound, unlike decode.

**Prefix caching** (13). Reusing computed KV blocks across requests that share an identical prompt prefix.

**Pre-norm** (2). Applying normalization before each sublayer rather than after, which is what makes deep stacks trainable.

**Probe** (14). A Kubernetes health check: startup delays judgment during model load, readiness gates traffic, liveness restarts the container.

**Propose-confirm-execute-undo** (21). The write-tool contract in which an action is proposed with a diff, approved by a human, executed once with an idempotency key, and reversible.

**PSI** (18). Population stability index, a binned measure of distribution shift between a reference period and the current one.

## Q

**QLoRA** (7). LoRA over a base quantized to 4-bit NormalFloat, dequantized on the fly, with paged optimizers; the reason a 7B model fine-tunes in single-digit gigabytes.

**Quality SLO** (20). A service level objective on answer quality, computed from online judge sampling and oracle checks.

**Quantization** (12). Storing weights in fewer bits through an affine map of scale and optional zero point, applied per tensor, per channel, or per group.

## R

**RadixAttention** (13). Organizing shared prompt prefixes as a tree so that cache reuse extends across requests and turns.

**Rank, LoRA** (7). The inner dimension of the low-rank update, controlling adapter capacity; 16 is a common default.

**RAG** (9). Retrieval-augmented generation: retrieving passages and grounding the model's answer in them, with citations.

**Reference model** (8). The frozen policy a preference method anchors to; under PEFT it is the same base model with adapters disabled.

**Registry** (17). The store of model versions with named aliases such as champion and challenger, plus lineage back to data and runs.

**Residual stream** (2). The running sum each layer reads from and adds to; the organizing idea of the transformer.

**Reward hacking** (8). A policy exploiting a loophole in the reward function rather than solving the task.

**RLVR** (8). Reinforcement learning with verifiable rewards, where a program rather than a learned model scores the output.

**RMSNorm** (2). Normalization by root mean square without subtracting the mean; cheaper than LayerNorm with equivalent effect.

**Roofline** (4). The model that places a kernel as compute-bound or memory-bound by comparing arithmetic intensity against the hardware's FLOPs-to-bandwidth ratio.

**RoPE** (2). Rotary position embedding: rotating query and key vectors by a position-dependent angle so attention depends on relative distance.

**Rug pull** (19, 22). An MCP server changing a tool's description after the user approved it.

**Runbook** (14, 20). The document an on-call engineer follows to deploy, roll back, scale, rotate a secret, and diagnose known failures.

## S

**Scaling laws** (6). The empirical power-law relationships between loss, parameters, data, and compute.

**Semantic cache** (16). A cache keyed by embedding similarity rather than exact match; its danger is the false hit.

**Sequential testing** (17). The statistical problem created by peeking at a canary repeatedly, which inflates false positives unless thresholds are set conservatively.

**SFT** (7). Supervised fine-tuning on demonstrations; it teaches format, behavior, and narrow skills, but adds knowledge poorly.

**Shadow mode** (17). Running a challenger on live traffic in parallel without showing its answers, scoring it before any exposure.

**SLI, SLO, SLA** (20). A service level indicator is a measurement, an objective is a target for it over a window, an agreement is a contractual commitment.

**Speculative decoding** (13). Drafting several tokens cheaply and verifying them in one target-model pass, with an acceptance rule that preserves the target distribution exactly.

**Spotlighting** (19). Marking untrusted content so the model treats it as data rather than instructions.

**Statement of work** (25). The agreement naming phases, deliverables, acceptance criteria, assumptions, and what is out of scope.

**Streamable HTTP** (22). The MCP remote transport with sessions, required for multi-user customer deployments.

**SwiGLU** (2). A gated MLP variant using three matrices instead of two, standard in Llama-style models.

## T

**Taint and toleration** (14). A node repels pods unless they tolerate the taint; the mechanism reserving GPU nodes for model pods.

**Tensor parallelism** (13). Splitting each layer's matrices across GPUs for models too large for one, requiring fast interconnect.

**Terraform state** (15). The record of the resources Terraform manages, kept in remote storage with locking.

**Threat model** (19). The enumeration of assets, trust boundaries, attacker capabilities, and controls for a system.

**Tokenizer** (1). The learned mapping from text to integer ids and back.

**Tool poisoning** (19, 22). Hidden instructions placed inside an MCP tool's description.

**Trace** (16, 17). The record of one request as nested spans and generations, carrying tenant, route, model, tokens, cost, latency, and quality.

**Trajectory** (21). The sequence of steps an agent took, evaluated separately from whether the end state was correct.

**Trust boundary** (19). A point where data crosses from a less trusted component to a more trusted one.

**TTFT and TPOT** (13). Time to first token, set by prefill and queueing, and time per output token, set by decode and batch size.

## U

**Unsloth** (7). A library of fused kernels and hand-derived backward passes that roughly doubles single-GPU LoRA speed and halves its memory.

**Use-case scoring** (25). Ranking candidate projects on value, feasibility, data readiness, risk, and time to first demo.

## V

**Vocabulary size** (1). The number of distinct tokens; trades embedding parameters and head compute against compression.

**VLM** (23). A vision-language model, pairing a vision encoder and a projector with a language model so image tokens interleave with text.

## W

**Warmup** (3). The opening phase of a learning-rate schedule where the rate rises from near zero, protecting the randomly initialized model and empty optimizer statistics.

**Workflow** (21). A system in which code, not the model, decides the sequence of steps; the right default for most business problems.

**WSL2** (5). Windows Subsystem for Linux version 2, a lightweight Linux virtual machine with GPU passthrough, where all training in this roadmap runs.

## Z

**Zero point** (12). The integer offset in an asymmetric quantization map, allowing the grid to cover a range that is not centered on zero.

**ZeRO** (6). The family of sharding stages behind FSDP: optimizer states, then gradients, then parameters.
