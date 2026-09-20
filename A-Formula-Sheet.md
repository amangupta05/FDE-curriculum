# Appendix A: Formula Sheet

Every formula the handbook uses, grouped by chapter in chapter order. Symbols follow the notation table in the front matter and each chapter's own definitions. Related formulas share a display block, and the line beneath each block defines its symbols and says when to reach for it. Derivations stay in the chapters; this is the sheet to keep open during a run or a customer call. Chapters 5 and 24 state no formulas and have no entries.

## Chapter 1: Tokenization and Data Representation

$$
\text{bpt}(s) = \frac{|\text{utf8}(s)|}{|\text{enc}(s)|}, \qquad N_{\text{emb}} = V d
$$
**Bytes per token and embedding table size.** Symbols: $s$ a string, $\text{utf8}(s)$ its bytes, $\text{enc}(s)$ its token ids, $V$ the vocabulary size, $d$ the model width. Use the first to turn a corpus into a token count, a context budget, or an API bill, and the second when choosing a vocabulary.

## Chapter 2: The Transformer, Derived

$$
x^{(\ell + 1/2)} = x^{(\ell)} + \text{Attn}\big(\text{Norm}(x^{(\ell)})\big), \quad x^{(\ell+1)} = x^{(\ell+1/2)} + \text{MLP}\big(\text{Norm}(x^{(\ell+1/2)})\big), \quad \text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_c x_c^2 + \epsilon}}, \quad \text{SwiGLU}(x) = \big(\text{SiLU}(xW_{gate}) \odot (xW_{up})\big)W_{down}
$$
**The pre-norm block, RMSNorm, SwiGLU.** Symbols: $x^{(\ell)}$ the residual stream at layer $\ell$, $\gamma$ a learned gain, $\epsilon$ a small constant, $W_{gate}, W_{up} \in \mathbb{R}^{d \times d_{ff}}$, $W_{down} \in \mathbb{R}^{d_{ff} \times d}$; LayerNorm is RMSNorm with the mean subtracted and a bias added. Use them when writing a block from scratch and when counting parameters, since SwiGLU costs three matrices, not two.

$$
R(m\theta)\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} x_1 \cos m\theta - x_2 \sin m\theta \\ x_1 \sin m\theta + x_2 \cos m\theta \end{pmatrix}, \qquad \theta_i = b^{-2i/d_{head}}, \qquad \big(R(m\theta)q\big)^{\top}\big(R(n\theta)k\big) = q^{\top}R\big((n-m)\theta\big)k
$$
**RoPE and its relative-position property.** Symbols: $m, n$ absolute positions, $(x_1, x_2)$ one rotated pair of a head's dimensions, $i = 0, \ldots, d_{head}/2 - 1$, $b$ the base (10,000 originally, 500,000 in Llama 3, 1,000,000 in Qwen2.5). Use the rotation when implementing position encoding or raising the base for long context, and the identity to show that only the position difference reaches the score.

$$
S_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_{head}}}, \quad \mathrm{Var}(q_i \cdot k_j) = d_{head}, \quad m_j = \max\big(m_{j-1}, \text{rowmax}(S_j)\big), \quad \ell_j = e^{m_{j-1}-m_j}\ell_{j-1} + \text{rowsum}\big(e^{S_j - m_j}\big), \quad O_j = e^{m_{j-1}-m_j}O_{j-1} + e^{S_j-m_j}V_j
$$
**Scaled scores and the online softmax behind FlashAttention.** Symbols: $S_{ij}$ the score of query $i$ against key $j$ with unit-variance components; in the tiled recurrence $S_j$ is the score tile for key block $j$, $m_j$ the running row maximum, $\ell_j$ the running denominator, $O_j$ the running unnormalized output. Use the variance to derive the $1/\sqrt{d_{head}}$ factor, and the recurrence to explain why attention never materializes a $T \times T$ matrix.

$$
N_{\text{non-embed}} \approx 12 L d^2, \quad N_{\text{layer}} = 2d^2\left(1 + \frac{n_{kv}}{h}\right) + 3 d\, d_{ff}, \quad m_{kv} = 2 L\, n_{kv}\, d_{head} \cdot \text{bytes}, \quad \text{tokens per second} \le \frac{W}{2N}
$$
**Parameter counting, KV-cache size, and the decode ceiling.** Symbols: $L$ layers, $d$ width, $h$ query heads, $n_{kv}$ key-value heads, $d_{ff}$ the MLP width, $m_{kv}$ cache bytes per token, $W$ bandwidth in bytes per second, $N$ parameters in bf16; add $Vd$ once for tied embeddings, twice for untied. Use the counts against a configuration file, the cache to size concurrency, and the ceiling to sanity-check any claimed tokens-per-second number.

## Chapter 3: Training Dynamics

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}), \quad p_\theta(y_t \mid y_{<t}) = \frac{\exp(z_{t,y_t})}{\sum_{j=1}^{V}\exp(z_{t,j})}, \quad \mathcal{L}_0 \approx \ln V, \quad \mathrm{PPL} = e^{\mathcal{L}}, \quad \sigma_{\text{out-proj}} = \frac{0.02}{\sqrt{2L}}
$$
**Cross-entropy, the loss at initialization, perplexity, and residual-branch scaling.** Symbols: $z_{t,j}$ the logit for vocabulary item $j$ at position $t$, $V$ the vocabulary size, targets the inputs shifted by one, $\sigma_{\text{out-proj}}$ the initialization standard deviation of $W_O$ and $W_{down}$, the two matrices that write to the residual stream. Run one forward pass on an untrained batch and compare the loss with $\ln V$; it is the cheapest correctness check in the roadmap. Perplexity is the effective number of equally likely choices the model is deciding among, is exactly $V$ at initialization, and is comparable only between models that share a tokenizer.

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2, \quad \hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \quad \theta_t = \theta_{t-1} - \eta\lambda\theta_{t-1} - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$
**The AdamW update.** Symbols: $g_t$ the gradient at step $t$, $m$ and $v$ the first and second moments with $m_0 = v_0 = 0$, $\beta_1, \beta_2$ their decay rates, $\eta$ the learning rate, $\lambda$ the decoupled weight decay, $\epsilon$ a stabilizer. Without bias correction the first step is $\frac{1-\beta_1}{\sqrt{1-\beta_2}}\,\mathrm{sign}(g_1)$, which is 3.16 at $(0.9, 0.999)$ and 0.45 at $(0.9, 0.95)$; that mismatch is why warmup exists.

$$
\eta(s) = \eta_{\min} + (\eta_{\max}-\eta_{\min})\cdot\frac{1}{2}\left(1 + \cos\left(\pi\frac{s-W}{S-W}\right)\right), \quad g \leftarrow g \cdot \min\left(1, \frac{c}{\lVert g\rVert_2 + 10^{-6}}\right), \quad \text{tokens per step} = B \times T \times A \times (\text{data-parallel ranks})
$$
**Cosine decay, global-norm clipping, and the effective batch.** Symbols: $s$ the step index, $W$ warmup steps, $S$ total steps, $\lVert g \rVert_2$ the global gradient norm, $c$ the clip threshold (1.0 by convention), $A$ the accumulation steps. Set the learning rate for the tokens-per-step figure, not for the micro-batch, and divide each micro-batch loss by $A$ before the backward pass.

## Chapter 4: Memory, Compute, and Throughput

$$
M_{\text{states}} \approx 16 N, \quad 10 N \text{ with 8-bit moments}, \quad 2 N \text{ frozen in bf16}, \quad 0.52 N \text{ frozen in NF4}, \qquad M_{\text{act}} \approx c\, B\, T\, L\, d, \quad c \approx 25 \text{ to } 45, \qquad M_{\text{act, ckpt}} \approx 2 B T L d + c\, B\, T\, d, \qquad M_{\text{logits}} \approx 10 B T V, \qquad \text{recompute overhead} = \frac{8N - 6N}{6N} = \frac{1}{3}
$$
**Bytes per parameter by scenario, activation memory, and the logits term.** Symbols: bytes; $N$ parameters, $B$ the micro-batch in sequences, $T$ the sequence length, $L$ layers, $d$ width, $V$ the vocabulary, $c$ the activation coefficient (use 36 for planning and measure yours). The 16 is 2 bf16 weights, 2 bf16 gradients, 4 fp32 master, 8 for two fp32 moments; 8-bit moments cut the last 8 to 2, freezing the base removes gradients, master copy, and moments together, and inference alone costs 2 bytes in bf16, 1 in int8, and about 0.5 in 4-bit. Checkpointing trades about a third more compute (20 to 30 percent measured) for an order of magnitude less activation memory, and the logits term is the one people forget: chunk the cross-entropy over the sequence axis when the vocabulary is large.

$$
m_{kv} = 2 L\, n_{kv}\, d_{head}\, b, \qquad T_{\text{total}} = \frac{M_{\text{GPU}} - M_{\text{weights}} - M_{\text{overhead}}}{m_{kv}}
$$
**KV cache and total cache capacity.** Symbols: $b$ bytes per element, $M_{\text{GPU}}$ the card's memory, $M_{\text{overhead}}$ the CUDA context, engine workspace, and largest prefill batch (1 to 2 GB for vLLM, 0.4 to 0.6 GB for a plain PyTorch loop), $T_{\text{total}}$ the tokens that fit across all sequences. Use it to convert a GPU into a concurrency number before promising one.

$$
C_{\text{fwd}} \approx 2N \text{ per token}, \qquad C_{\text{train}} \approx 6 N D, \qquad C_{\text{decode}} \approx 2N \text{ per token per sequence}, \qquad M_{\text{LoRA}} \approx b_{\text{base}} N + 16 N_{\text{adapter}} + M_{\text{act}}
$$
**FLOPs and adapter memory.** Symbols: $D$ training tokens, $b_{\text{base}}$ bytes per parameter of the frozen base (2 for bf16, about 0.52 for NF4), $N_{\text{adapter}}$ the trainable count. $6ND$ omits attention score and value products, a fraction about $T/(6d)$ of the total, so it understates long-context runs.

$$
t \ge \max\left(\frac{\text{FLOPs}}{P}, \frac{\text{bytes}}{\beta}\right), \quad I^{*} = \frac{P}{\beta}, \quad I_{\text{prefill}} \approx T, \quad I_{\text{decode}} = \frac{2NB}{2N + B\bar{T}m_{kv}}, \quad \text{tokens per second} \le \frac{\beta}{M_{\text{weights}} + \bar{T}m_{kv}}, \quad \text{MFU} = \frac{6N \cdot (\text{tokens per second})}{P}
$$
**The roofline and model FLOPs utilization.** Symbols: $P$ the peak arithmetic rate, $\beta$ memory bandwidth, $I$ arithmetic intensity in FLOPs per byte, $I^{*}$ the ridge point (about 176 on the 4060, 203 on a T4, 156 on an A100), $\bar{T}$ the mean context length. Prefill is compute-bound for any real prompt, decode is memory-bound at any realistic batch, and MFU is the honest way to report throughput because tokens per second alone conflates model size with hardware.

## Chapter 6: Pretraining at Small Scale

$$
J(A,B) = \frac{|A \cap B|}{|A \cup B|}, \quad P[h_\pi(A) = h_\pi(B)] = J(A,B), \quad \hat{J} = \frac{1}{k}\sum_{i=1}^{k}\mathbf{1}[h_{\pi_i}(A) = h_{\pi_i}(B)], \quad P[\text{candidate}] = 1 - (1 - s^r)^b
$$
**MinHash and LSH banding.** Symbols: $A, B$ shingle sets, $h_\pi$ a min-hash, $k$ hash functions (estimator variance $J(1-J)/k$), $s$ the true similarity, $b$ bands of $r$ rows with $k = br$, threshold $s^{*} \approx (1/b)^{1/r}$. Use them to near-deduplicate a corpus or a synthetic dataset without a pairwise pass, and to pick $b$ and $r$ for the threshold you want.

$$
V_{\text{all-reduce}} = 2\frac{G-1}{G}D_{\text{grad}}, \quad M_{\text{DDP}} = 16\Psi, \quad M_{\text{stage 1}} = 4\Psi + \frac{K\Psi}{G}, \quad M_{\text{stage 2}} = 2\Psi + \frac{(2+K)\Psi}{G}, \quad M_{\text{stage 3}} = \frac{16\Psi}{G}
$$
**Data-parallel communication and memory.** Symbols: $G$ GPUs, $D_{\text{grad}}$ gradient bytes, $\Psi$ parameters, $K = 12$ bytes of fp32 master weight and two moments. Use the first to predict whether the interconnect bottlenecks a step and the rest to choose between DDP and ZeRO stages 1 to 3.

$$
L(N,D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}, \quad E \approx 1.69,\ A \approx 406.4,\ B \approx 410.7,\ \alpha \approx 0.34,\ \beta \approx 0.28, \quad N_{\text{opt}} = G\left(\frac{C}{6}\right)^{a}, \quad D_{\text{opt}} = \frac{1}{G}\left(\frac{C}{6}\right)^{b}, \quad \text{BPB} = \frac{\ell\, n_{\text{tok}}}{n_{\text{bytes}} \ln 2}
$$
**Chinchilla, compute-optimal allocation, and bits per byte.** Symbols: $C$ training FLOPs, $a = \beta/(\alpha+\beta)$, $b = \alpha/(\alpha+\beta)$, $G = (\alpha A/\beta B)^{1/(\alpha+\beta)}$, $\ell$ the mean cross-entropy in nats per token over $n_{\text{tok}}$ tokens and $n_{\text{bytes}}$ bytes; the constants are Hoffmann et al. 2022 as reported and are disputed by Besiroglu et al. 2024. Use the allocation to turn a GPU-hour budget into a size and a token count, then over-train deliberately when inference volume is large, and BPB whenever two tokenizers must be compared.

## Chapter 7: Supervised Fine-Tuning, LoRA, and QLoRA

$$
h = W_0 x + \frac{\alpha}{r}BAx, \quad P_{\text{LoRA}} = r(d_{\text{in}} + d_{\text{out}}), \quad \frac{\partial\mathcal{L}}{\partial B} = \frac{\alpha}{r}\frac{\partial\mathcal{L}}{\partial h}(Ax)^{\top}, \quad \frac{\partial\mathcal{L}}{\partial A} = \frac{\alpha}{r}B^{\top}\frac{\partial\mathcal{L}}{\partial h}x^{\top}
$$
**The LoRA update, count, and gradients.** Symbols: $W_0$ the frozen base weight, $A \in \mathbb{R}^{r \times d_{\text{in}}}$ random at initialization, $B \in \mathbb{R}^{d_{\text{out}} \times r}$ zero at initialization, $r$ the rank, $\alpha$ the scaling numerator. Use the count to report trainable parameters per module, and the gradients to explain why a zero-initialized $B$ still trains and why $\alpha/r$ is a second learning-rate knob.

$$
\frac{8}{64} + \frac{32}{64 \times 256} \approx 0.127 \text{ bits per weight}, \qquad M \approx \underbrace{0.52 N_{\text{lin}} + 2 N_{\text{emb}}}_{\text{base}} + \underbrace{16 P_{\text{LoRA}}}_{\text{10 with 8-bit moments}} + \underbrace{2 B T d L}_{\text{checkpointed activations}} + \underbrace{10 B T V}_{\text{unchunked logits}}
$$
**Double quantization and the QLoRA memory budget.** Symbols: bytes; 8-bit block constants over blocks of 64 weights requantized in groups of 256, giving about 4.127 bits or 0.52 bytes per weight; $N_{\text{lin}}$ the quantized linear parameters, $N_{\text{emb}}$ embedding and output parameters kept in bf16, plus about 0.5 GB of fixed overhead. Run it before every job on 8 GB: the logits term is what usually breaks a large-vocabulary model, and a chunked loss is the fix.

## Chapter 8: Preference Optimization and Reinforcement Learning

$$
P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big), \quad \mathcal{L}_{\mathrm{RM}} = -\mathbb{E}\big[\log\sigma(r_\phi(x,y_w) - r_\phi(x,y_l))\big], \quad \pi^{*}(y \mid x) = \frac{\pi_{\mathrm{ref}}(y \mid x)\exp(r(x,y)/\beta)}{Z(x)}, \quad L^{\mathrm{CLIP}} = \mathbb{E}_t\big[\min(\rho_t\hat{A}_t, \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)\big]
$$
**Bradley-Terry, the reward model, the KL-regularized optimum, and PPO.** Symbols: $y_w$ preferred and $y_l$ rejected responses to prompt $x$, $\sigma$ the logistic sigmoid, $\pi_{\mathrm{ref}}$ the frozen reference, $\beta$ the KL coefficient, $Z(x)$ the partition function over all responses, $\rho_t$ the PPO probability ratio with advantage $\hat{A}_t$ and clip range $\epsilon = 0.2$. The reward-model loss starts at $\ln 2 \approx 0.693$, and the optimum is the starting point of the DPO derivation.

$$
\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta\log\frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}\right)\right], \quad \hat{r}_\theta(x,y) = \beta\log\frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}
$$
**The DPO loss and implicit reward.** Symbols: each $\log\pi(y \mid x)$ is the sum of per-token log-probabilities over response tokens only. The gradient is $-\beta\mathbb{E}\big[\sigma(\hat{r}_\theta(x,y_l) - \hat{r}_\theta(x,y_w))(\nabla_\theta\log\pi_\theta(y_w \mid x) - \nabla_\theta\log\pi_\theta(y_l \mid x))\big]$, whose weight shrinks once a pair is separated; the loss starts at $\ln 2$ when the reference is right, which is the fastest reference-mismatch check there is.

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}, \quad \mathcal{J}_{\mathrm{GRPO}} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big(\min\big(\rho_{i,t}\hat{A}_{i,t}, \mathrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\big) - \beta\mathbb{D}_{i,t}\Big)\right]
$$
**The GRPO advantage and objective.** Symbols: $G$ responses per prompt with rewards $r_i$, $\hat{A}_{i,t} = \hat{A}_i$ for every token under outcome supervision, $o_i$ the $i$-th output, $\rho_{i,t}$ the per-token ratio against the sampling policy, $\mathbb{D}_{i,t} = \pi_{\mathrm{ref}}/\pi_\theta - \log(\pi_{\mathrm{ref}}/\pi_\theta) - 1$ the unbiased KL estimator, $\epsilon = 0.2$ and $\beta = 0.04$ in DeepSeekMath; the loss is the negative. Advantages sum to zero, a unanimous group produces none, and the $1/|o_i|$ factor is the length bias toward long incorrect outputs.

## Chapter 9: Embeddings, Retrieval, and Rerankers

$$
s(q,p) = \frac{f_\theta(q)^\top f_\theta(p)}{\lVert f_\theta(q)\rVert \lVert f_\theta(p)\rVert}, \quad \mathcal{L}_{\mathrm{InfoNCE}} = -\log\frac{\exp(s(q,p^+)/\tau)}{\exp(s(q,p^+)/\tau) + \sum_{j=1}^{K}\exp(s(q,p^-_j)/\tau)}, \quad \frac{\partial\mathcal{L}}{\partial s(q,p_j)} = \frac{1}{\tau}\big(P_j - \mathbb{1}[p_j = p^+]\big)
$$
**Cosine similarity, InfoNCE, and its gradient.** Symbols: $f_\theta$ the encoder, $p^+$ the positive, $p^-_j$ the $K$ negatives, $\tau$ the temperature (0.05 corresponds to a scale of 20), $P_j$ the softmax probability of candidate $j$. Use the gradient to explain why hard negatives dominate the update, and the bound $I(q;p) \ge \log(K+1) - \mathcal{L}$ when arguing for larger batches.

$$
\mathcal{L}_{\mathrm{MNRL}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^{N}\exp(S_{ij}/\tau)}, \quad \mathcal{L}_{\mathrm{MRL}} = \sum_{m \in \mathcal{M}}c_m\mathcal{L}(\tilde{e}_{1:m}), \quad \tilde{e}_{1:m} = \frac{e_{1:m}}{\lVert e_{1:m}\rVert}, \quad e = \frac{\sum_{t=1}^{T}a_t h_t}{\sum_{t=1}^{T}a_t}
$$
**In-batch negatives, Matryoshka, and mean pooling.** Symbols: $S_{ij} = s(q_i, p_j)$ the in-batch similarity matrix with the diagonal as labels, $\mathcal{M}$ the prefix lengths (768, 512, 256) with weights $c_m$, $h_t$ the final hidden state at position $t$, $a_t \in \{0,1\}$ the attention mask. Use MNRL as the default loss when you have only positive pairs, the Matryoshka sum when vectors must be truncatable, and the pooling rule the base model was trained with, never another.

$$
\mathrm{BM25}(q,d) = \sum_{t \in q}\mathrm{IDF}(t)\frac{f(t,d)(k_1+1)}{f(t,d) + k_1\left(1 - b + b\frac{\lvert d\rvert}{\mathrm{avgdl}}\right)}, \quad \mathrm{IDF}(t) = \ln\frac{N - n_t + 0.5}{n_t + 0.5} + 1, \quad \mathrm{RRF}(d) = \sum_{r \in R}\frac{1}{k + \mathrm{rank}_r(d)}
$$
**BM25 and reciprocal rank fusion.** Symbols: $f(t,d)$ the count of term $t$ in $d$, $\lvert d\rvert$ its length, $\mathrm{avgdl}$ the mean length, $N$ documents, $n_t$ documents containing $t$, $k_1 = 1.2$ to 1.5, $b = 0.75$, $R$ the ranked lists and $k = 60$ in the fusion paper. Use BM25 as the lexical half of hybrid retrieval, which rescues part numbers and error codes, and RRF to combine lists without calibrating their scores.

$$
\mathrm{Recall@}k = \frac{\lvert\mathrm{Rel}(q) \cap \{d_1,\dots,d_k\}\rvert}{\lvert\mathrm{Rel}(q)\rvert}, \quad \mathrm{MRR} = \frac{1}{\lvert Q\rvert}\sum_{q \in Q}\frac{1}{\mathrm{rank}_q}, \quad \mathrm{nDCG@}k = \frac{\sum_{i=1}^{k}g(\mathrm{rel}_i)/\log_2(i+1)}{\mathrm{IDCG@}k}, \quad s(q,p) = \sum_{i \in q}\max_{j \in p}\tilde{q}_i^\top\tilde{p}_j
$$
**Retrieval metrics and late interaction.** Symbols: $\mathrm{Rel}(q)$ the relevant set, $\mathrm{rank}_q$ the rank of the first relevant result, $g(\mathrm{rel}_i)$ the gain at rank $i$, IDCG the ideal ordering's DCG, $\tilde{q}_i, \tilde{p}_j$ normalized token vectors in the ColBERT MaxSim score. Use recall@k for the first stage's ceiling, MRR when one right answer exists, nDCG@k for graded relevance, and MaxSim when token-level precision is worth the storage.

## Chapter 10: Synthetic Data and Distillation

$$
C_{\text{api}} = \frac{t_{\text{in}}p_{\text{in}} + t_{\text{cache}}p_{\text{cache}} + t_{\text{out}}p_{\text{out}}}{10^6} \times 1000, \qquad C_{\text{host}} = \frac{P_{\text{gpu}}}{R} \times 1000
$$
**Cost per thousand requests, API and self-hosted.** Symbols: $t$ average tokens per request (uncached input, cached input, output), $p$ prices per million tokens, $P_{\text{gpu}}$ the GPU price per hour, $R$ sustained requests per hour at the utilization actually achieved. Use it for the cost-quality table of a distillation project; the trap is $R$, which is capacity times utilization, not capacity.

$$
p_i^{(T)} = \frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}, \quad \mathcal{L} = (1-\lambda)\mathrm{CE}(y, q^{(1)}) + \lambda T^2\mathrm{KL}\big(p^{(T)} \| q^{(T)}\big), \quad \mathcal{L}_{\text{seq}} = -\frac{1}{N}\sum_{j=1}^{N}\sum_{t=1}^{|y_j|}\log q\big(y_{j,t} \mid x_j, y_{j,<t}\big)
$$
**Logit and sequence-level distillation.** Symbols: $z$ teacher and $v$ student logits with temperature-$T$ softmaxes $p^{(T)}, q^{(T)}$, $\lambda$ the mixing weight, $y$ the ground-truth token, $y_j$ the teacher's sampled response to prompt $x_j$; the KL gradient is $(q_i^{(T)} - p_i^{(T)})/T$, which is why the $T^2$ factor exists. Logit distillation needs full logits and a shared tokenizer, so an API teacher forces the sequence loss, which is ordinary SFT with a completion-only mask.

## Chapter 11: Evaluation and Statistics

$$
\left[\hat{\theta}^{*}_{(\alpha/2)}, \hat{\theta}^{*}_{(1-\alpha/2)}\right], \qquad \mathrm{Var}(\bar{a} - \bar{b}) = \frac{1}{n}\left[\mathrm{Var}(a) + \mathrm{Var}(b) - 2\mathrm{Cov}(a,b)\right]
$$
**The percentile bootstrap and why pairing helps.** Symbols: $\hat{\theta}^{*}_{(q)}$ the $q$-quantile of $B$ bootstrap replicates, $\alpha = 0.05$ for a 95 percent interval, $a_i$ and $b_i$ two systems' per-item scores on the same $n$ items. Bootstrap every number you report, with $B \ge 2000$ and 10,000 when final, and resample clusters rather than items when the items come in clusters.

$$
n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2[p_1(1-p_1) + p_2(1-p_2)]}{(p_1-p_2)^2}, \qquad n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2[\pi_{10} + \pi_{01} - \Delta^2]}{\Delta^2}
$$
**Sample size, unpaired and paired.** Symbols: $p_1, p_2$ the two accuracies, $\pi_{10}, \pi_{01}$ the discordant-pair proportions, $\Delta = \pi_{10} - \pi_{01}$, $z_{1-\alpha/2} = 1.96$ and $z_{1-\beta} = 0.84$ for 95 percent confidence and 80 percent power. Use it before spending API money; it is what shows that 200 items cannot resolve a 3-point difference.

$$
\kappa = \frac{p_o - p_e}{1 - p_e}, \quad p_e = \sum_{k=1}^{K}p_{k\cdot}p_{\cdot k}, \quad P(i \succ j) = \sigma(\beta_i - \beta_j), \quad \pi_i \leftarrow \frac{\sum_{j \ne i}w_{ij}}{\sum_{j \ne i}\dfrac{n_{ij}}{\pi_i + \pi_j}}, \quad E_i = \frac{1}{1 + 10^{(R_j-R_i)/400}}
$$
**Judge calibration and pairwise aggregation.** Symbols: $p_o$ observed and $p_e$ chance agreement over $K$ classes (bands: 0.4 to 0.6 moderate, 0.6 to 0.8 substantial), $\beta_i = \log\pi_i$ the strength of system $i$, $w_{ij}$ its wins over $j$ in $n_{ij}$ comparisons, $R_i$ an Elo rating updated by $R_i + K(S_i - E_i)$. Calibrate a judge with kappa before quoting any judged number, use the MM iteration with a bootstrap for a ranking, and Elo only when ratings must update online.

## Chapter 12: Quantization

$$
t_{\text{step}} \ge \frac{B_w + B_{kv}}{\beta}, \qquad M_w = N_{\text{lin}}\cdot\frac{\text{bits per weight}}{8} + M_{\text{emb}}, \qquad \mathrm{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log p(x_t \mid x_{<t})\right)
$$
**The bandwidth floor, the quantized footprint, and perplexity.** Symbols: $B_w$ weight bytes read per step, $B_{kv}$ KV bytes read, $\beta$ bandwidth, $N_{\text{lin}}$ the quantized linear weights, $M_{\text{emb}}$ embedding and output matrices at their own precision, $T$ scored tokens. Use the first to predict what quantization buys before running anything, the second to predict checkpoint size, and the third only between models that share a tokenizer.

$$
s = \frac{\max_i|w_i|}{q_{\max}}, \quad \hat{w}_i = s q_i, \quad \mathbb{E}[(\hat{w}-w)^2] = \frac{s^2}{12}, \qquad s = \frac{w_{\max}-w_{\min}}{2^b-1}, \quad z = \mathrm{round}\left(-\frac{w_{\min}}{s}\right), \quad \hat{w}_i = s(q_i - z), \qquad \text{bits per weight} = b + \frac{16}{g}\left(+\frac{b}{g}\right)
$$
**Symmetric and asymmetric quantization, error, and storage overhead.** Symbols: $w_i$ a real weight, $s$ the scale (the real value of one grid step), $q_i$ the integer code clamped to the grid, $b$ the bit width, $q_{\max} = 2^{b-1}-1$, $z$ the zero point, $g$ the group size with the bracketed term added when a zero point is stored per group. Weights are roughly symmetric about zero and activations are not; the $s^2/12$ term is why one outlier degrades its whole group.

$$
q_i = \frac{1}{2}\left(\Phi^{-1}\left(\frac{i}{2^k+1}\right) + \Phi^{-1}\left(\frac{i+1}{2^k+1}\right)\right), \qquad i = 1, \ldots, 2^k
$$
**The NormalFloat grid.** Symbols: $\Phi^{-1}$ the inverse standard normal cumulative distribution, the levels rescaled to $[-1,1]$; NF4 uses an asymmetric split of eight positive and seven negative levels plus an exact zero, with a per-block absmax over blocks of 64. Use it to explain why NF4 beats uniform int4 on normally distributed weights.

$$
\min_{\hat{W}}\|WX - \hat{W}X\|_F^2, \quad H = 2XX^\top, \quad \delta = -\frac{w_q - \mathrm{quant}(w_q)}{[H^{-1}]_{qq}}H^{-1}_{:,q}, \quad y = Wx = \big(W\mathrm{diag}(s)\big)\big(\mathrm{diag}(s)^{-1}x\big), \quad s_j = \bar{x}_j^{\alpha}
$$
**Calibration-based quantization.** Symbols: $W$ a layer's weights, $X$ calibration activations, $H$ the Hessian of the row's squared output error, $\delta$ the compensation applied after quantizing column $q$, $s_j$ the AWQ per-channel scale from mean activation magnitude $\bar{x}_j$ with $\alpha \in [0,1]$; SmoothQuant uses $s_j = \max|X_j|^{\alpha}/\max|W_j|^{1-\alpha}$ and FP8 E4M3 uses $s = \max|x|/448$. GPTQ trades weight-space error for output-space error; the equivalent transformation costs nothing at inference.

## Chapter 13: Serving Systems

$$
t_{\text{step}}(B) \approx \frac{2N + B\bar{T}m_{kv}}{\beta}, \qquad \text{throughput} = \frac{B}{t_{\text{step}}(B)}, \qquad 2L \cdot 2d \cdot \frac{2(n-1)}{n}
$$
**Batched step time and tensor-parallel traffic.** Symbols: $B$ the batch, $\bar{T}$ the mean context length, $m_{kv}$ cache bytes per token, $\beta$ bandwidth; the last term is bytes moved per token per GPU at tensor-parallel degree $n$, two all-reduces per layer. Model throughput against batch before benchmarking, and note that tensor parallelism costs collective latency, not bandwidth, so it needs NVLink.

$$
P(y) = q(y)\min\left(1, \frac{p(y)}{q(y)}\right) + (1-\alpha)\frac{\max(0, p(y)-q(y))}{\sum_{y'}\max(0, p(y')-q(y'))}, \quad \mathbb{E}[\text{tokens per step}] = \frac{1-\alpha^{k+1}}{1-\alpha}, \quad \text{speedup} \approx \frac{\mathbb{E}[\text{tokens per step}]}{1+kc}
$$
**Speculative decoding.** Symbols: $p$ the target and $q$ the draft distribution, $\alpha = \sum_y\min(p(y), q(y))$ the acceptance probability, $k$ the draft length, $c$ the draft cost as a fraction of one target step; the first expression simplifies to $p(y)$. Use it to prove speculation changes speed and not the output distribution, and to choose $k$: at $\alpha = 0.8$ and $c = 0.1$ the gain flattens near $k = 5$.

$$
\text{cost per million output tokens} = \frac{c_{\text{gpu}}}{R_{\text{out}} \cdot 3600 \cdot u} \times 10^6, \qquad V^{*} = \frac{24c_{\text{gpu}} + c_{\text{ops}}/30}{n_{\text{in}}p_{\text{in}} + n_{\text{out}}p_{\text{out}}}
$$
**The self-host cost model.** Symbols: $c_{\text{gpu}}$ the GPU price per hour, $R_{\text{out}}$ output tokens per second at full load, $u$ utilization, $n_{\text{in}}, n_{\text{out}}$ tokens per request, $p_{\text{in}}, p_{\text{out}}$ API prices per token, $c_{\text{ops}}$ the monthly operational cost, $V^{*}$ the break-even requests per day. Use it for every API-versus-self-hosted memo and always show the sensitivity to $u$, which moves the answer more than any other input.

## Chapter 14: Kubernetes for Model Serving

$$
T_{\text{start}} = \text{initialDelaySeconds} + \text{failureThreshold} \times \text{periodSeconds}, \quad t_{\text{cold}} = t_{\text{node}} + t_{\text{pull}} + t_{\text{load}} + t_{\text{probe}}, \quad n_{\text{desired}} = \left\lceil n_{\text{current}}\frac{m_{\text{current}}}{m_{\text{target}}}\right\rceil
$$
**Probe, cold-start, and autoscaling arithmetic.** Symbols: $T_{\text{start}}$ the budget a startup probe grants from container start, excluding image pull; $t_{\text{cold}}$ the scale-from-zero latency; $n$ replicas and $m$ the metric averaged over pods, with a 10 percent tolerance and a 300 second scale-down window by default. Use the first so a server that loads for minutes is not killed, the second to judge whether scale-to-zero fits an SLO, the third to predict autoscaler behavior.

## Chapter 15: Infrastructure as Code and CI/CD

$$
\mathrm{Var}(d_i) = \rho - \bar{d}^{\,2} \approx \rho, \qquad \mathrm{SE}(\bar{d}) \approx \sqrt{\rho/N}
$$
**The evaluation gate's standard error.** Symbols: $d_i \in \{-1,0,1\}$ the per-item difference between challenger and champion, $\bar{d}$ the accuracy difference, $\rho$ the discordance rate, $N$ the evaluation set size. Use it to check that a gate threshold is one the set can enforce; a 500-item set cannot police a 2-point regression.

## Chapter 16: Gateways, Routing, and Cost Engineering

$$
(a_2 - \hat{p})V > c_2 - c_1 \iff \hat{p} < a_2 - \frac{c_2-c_1}{V}, \qquad e = \mathrm{TPR}(1 - a_1^{\text{raw}}) + \mathrm{FPR}\,a_1^{\text{raw}}
$$
**The escalation threshold and the escalation rate.** Symbols: $\hat{p}$ the classifier's confidence that the cheap route answers correctly, $a_1^{\text{raw}}$ and $a_2$ the accuracies of the cheap and expensive routes, $c_1, c_2$ their costs, $V$ the value of a correct answer, TPR and FPR the escalation signal's rates. Use the threshold to set a router from costs rather than intuition, and $e$ as the fraction of traffic the expensive route actually sees.

$$
\mathbb{E}[C] = c_1 + c_s + e\,c_2, \qquad \mathbb{E}[A] = a_1^{\text{raw}}(1-\mathrm{FPR}) + e\,a_2, \qquad \mathbb{E}[C] = c_1 + e_1(c_2 + e_2 c_3)
$$
**Cascade cost and accuracy.** Symbols: $c_s$ the cost of the escalation signal itself, $e_i$ the escalation rate out of tier $i$, $c_3$ a third tier's cost. Quote both numbers together: a cascade that is cheaper and less accurate is a different product, and three tiers are rarely worth the complexity unless the middle tier differs in kind.

$$
a_{\text{cached}} = (1-h)a + h(1-\phi), \quad \bar{c} = (1-h)\mathbb{E}[C] + h\,c_{\text{cache}}, \quad k > \frac{m_w - m_r}{1 - m_r}, \quad c = p_{\text{in}}(n_{\text{new}} + m_w n_w + m_r n_r) + p_{\text{out}}n_{\text{out}}
$$
**Semantic caching and prompt caching.** Symbols: $h$ the cache hit rate, $\phi$ the false-hit rate, $a$ the uncached accuracy, $c_{\text{cache}}$ the lookup cost, $m_w$ and $m_r$ the cache-write and cache-read price multipliers, $k$ the uses of a cached prefix within its lifetime, $n_{\text{new}}, n_w, n_r$ the uncached, written, and read prefix tokens. A semantic cache is safe while $\phi \le 1-a$, and a prompt cache pays for itself after about 1.3 uses of a shared prefix.

## Chapter 17: The Data Flywheel

$$
s(t) = \sum_k w_k f_k(t), \qquad \sum_k w_k = 1, \quad w_k \ge 0
$$
**Curation scoring.** Symbols: $t$ a trace, $f_k(t) \in [0,1]$ the value of curation signal $k$ (oracle failure, feedback, judge score, novelty), $w_k$ its weight. Use it to rank the review queue; the weights are a policy decision to write down and revisit, not a constant.

$$
\hat{\Delta} = \frac{1}{n}\sum_{i=1}^{n}(c_i - m_i), \qquad \mathrm{Var}(\hat{\Delta}) \approx \frac{p_{10} + p_{01} - (p_{10}-p_{01})^2}{n}, \qquad \alpha_{\text{look}} = \frac{\alpha}{k}
$$
**Promotion statistics.** Symbols: $c_i, m_i$ challenger and champion correctness on item $i$, $p_{10}, p_{01}$ the discordant proportions, $k$ the number of pre-registered looks at a canary under Bonferroni. Use the paired difference for the gate and the Bonferroni split whenever you will look at a canary more than once.

## Chapter 18: Evaluation Platforms and Monitoring

$$
n = \frac{z^2 p(1-p)}{h^2}, \qquad z = 1.96
$$
**Judge sampling rate.** Symbols: $p$ the expected pass rate, $h$ the half-width you need, $n$ judged samples per window. Use it to set how many production responses to judge: at $p = 0.9$ and $h = 0.03$, about 384 per window.

$$
\mathrm{PSI} = \sum_{i=1}^{B}(\ell_i - r_i)\ln\frac{\ell_i}{r_i}, \quad \mathbb{E}[\mathrm{PSI} \mid \text{no drift}] \approx (B-1)\left(\frac{1}{N} + \frac{1}{M}\right), \quad D = \sup_x|F_{\text{ref}}(x) - F_{\text{live}}(x)| > c(\alpha)\sqrt{\frac{n+m}{nm}}
$$
**Drift detection.** Symbols: $r_i$ the reference and $\ell_i$ the live proportion in bin $i$ over $B$ bins, $N$ and $M$ the reference and live sample sizes, $F$ the empirical cumulative distributions with sample sizes $n$ and $m$, $c(0.05) \approx 1.358$ and $c(0.01) \approx 1.628$. Compare PSI against its null expectation before calling a 0.02 reading drift, and treat the Kolmogorov-Smirnov $D$ as the effect size because at large $n$ everything is significant.

## Chapter 19: Security for LLM Systems

$$
\mathrm{Precision} = \frac{\pi\,\mathrm{TPR}}{\pi\,\mathrm{TPR} + (1-\pi)\mathrm{FPR}}, \quad \text{Miss rate} = \prod_i(1 - \mathrm{TPR}_i), \quad \text{Combined FPR} = 1 - \prod_i(1 - \mathrm{FPR}_i)
$$
**Guardrail precision and defense in depth.** Symbols: $\pi$ the prevalence of attacks in traffic, TPR and FPR the detector's rates, one factor per independent defense layer. A 90 percent detector at 2 percent false positives is right about one time in five when attacks are rare, and layering multiplies recall while accumulating false positives, which is why structural validators carry the load.

$$
\text{center} = \frac{\hat{p} + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}, \qquad \text{half-width} = \frac{z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
$$
**The Wilson score interval.** Symbols: $\hat{p}$ the observed proportion, $n$ the trials, $z = 1.96$. Use it for attack success rates and any proportion near 0 or 1, where the normal approximation runs off the end of the scale.

## Chapter 20: Reliability and FinOps

$$
\mathrm{SLI} = \frac{\text{good events}}{\text{valid events}}, \qquad \mathrm{SE} = \sqrt{\frac{\hat{q}(1-\hat{q})}{n}}, \qquad n \approx \frac{\left(z_\alpha\sqrt{q_0(1-q_0)} + z_\beta\sqrt{q_1(1-q_1)}\right)^2}{(q_0-q_1)^2}
$$
**Service level indicators and the quality SLO's sample size.** Symbols: $\hat{q}$ the judged pass rate over $n$ sampled responses, $q_0$ the objective and $q_1$ the drop you must detect, $z_\alpha = 1.645$ at 5 percent and $z_\beta = 0.84$ at 80 percent power. Say in the objectives document how finely the window can adjudicate the objective, and never promise a per-tenant SLO you cannot sample enough traffic to measure.

$$
E = (1 - \mathrm{SLO})N, \qquad b = \frac{\text{observed bad fraction}}{1 - \mathrm{SLO}}, \qquad \text{budget consumed} = b\frac{w}{T}, \qquad t_{\text{detect}} \approx w\frac{b(1-\mathrm{SLO})}{r}
$$
**Error budget and burn rate.** Symbols: $N$ valid events in the window, $E$ the allowed bad events, $b$ the burn rate ($b = 1$ spends the budget in exactly one window), $w$ the alert window inside an SLO window $T$, $r$ the fraction of traffic affected. Use a fast window and a slow window together, and quote $t_{\text{detect}}$ so nobody expects a page faster than the maths allows.

$$
b_{\text{threshold}} = \phi \, \frac{T}{w}
$$
**The burn-rate threshold for a chosen window.** Symbols: $\phi$ the fraction of the error budget you are willing to spend before alerting, $w$ the alert window, $T$ the SLO window. With $T = 30$ days, spending 2 percent in 1 hour gives $0.02 \times 720 = 14.4$; 5 percent in 6 hours gives 6; 10 percent in 1 day gives 3; 10 percent in 3 days gives 1. This is where the conventional 14.4, 6, 3, and 1 come from, so you can rederive them for any window instead of copying a table. Page on the two fast rows, ticket on the two slow ones, and pair each with a short window of about $w/12$ so the alert stops firing after recovery.

$$
\text{sleep} = \mathrm{Uniform}\left(0, \min\left(\text{cap}, \text{base} \times 2^{\text{attempt}}\right)\right), \qquad \mathbb{E}[\text{attempts}] = \frac{1 - f^{m+1}}{1-f}, \qquad \lambda_{\text{trigger}} = \lambda_{\max} - g\,t_{\text{lead}}
$$
**Full jitter, retry amplification, and the scaling trigger.** Symbols: $f$ the failure probability, $m$ the maximum retries, $\lambda_{\max}$ the measured knee in requests per second, $g$ the traffic growth rate, $t_{\text{lead}}$ the time to add capacity. Full jitter prevents synchronized retry waves, retries multiply load exactly when the backend can least absorb it (cap them with a retry budget), and the trigger must fire a lead time before the knee.

$$
\text{cost per correct answer} = \frac{\text{cost per query}}{\text{accuracy}}, \quad \mathrm{ROI} = \frac{\text{savings} - \text{infrastructure cost}}{\text{infrastructure cost}}, \quad z = \frac{0.6745(x - \tilde{x})}{\mathrm{MAD}}, \quad u = \frac{\text{output tokens in the window}}{R_{\text{out}} \times \text{seconds in the window}}
$$
**Unit economics, cache return, spend anomalies, and utilization.** Symbols: savings are $n_{\text{hits}}(c_{\text{avoided}} - c_{\text{lookup}})$, $\tilde{x}$ the median of the trailing window, $\mathrm{MAD}$ the median absolute deviation, $R_{\text{out}}$ the aggregate output tokens per second at the benchmarked knee. Cost per correct answer is the only unit that cannot be improved by routing traffic to a worse model; alert on $|z| > 3.5$ together with an absolute floor; and this $u$ is not the number `nvidia-smi` reports.

## Chapter 21: Agents in Production

$$
N_{in}(T) = \sum_{t=0}^{T-1}(c_0 + t\delta) = Tc_0 + \frac{\delta T(T-1)}{2}, \qquad R = w_e e(n) + w_m m(M) + w_r r + w_v v
$$
**Context growth and risk scoring.** Symbols: $T$ steps, $c_0$ initial context tokens, $\delta$ tokens added per step; $e(n) = \min(1, \log_{10}(1+n)/3)$ on entities affected, $m(M) = \min(1, \log_{10}(1+M)/5)$ on dollars at stake, $r = 1$ for irreversible actions, $v = 1$ for an unfamiliar action type, weights summing to 1. Token cost is quadratic in the step budget, so bound the steps; log the risk score with every action so the approval threshold can be tuned from data. Retries use Chapter 20's full-jitter backoff.

$$
\text{pass@}k = 1 - (1-p)^k, \quad \text{pass}^k = p^k, \quad \widehat{\text{pass}^k} = \frac{\binom{s}{k}}{\binom{n}{k}}, \quad \widehat{\text{pass@}k} = 1 - \frac{\binom{n-s}{k}}{\binom{n}{k}}, \quad C_{succ} = \frac{\sum_i c_i}{\sum_i s_i}
$$
**Agent evaluation metrics.** Symbols: $p$ the per-run success probability, $n$ runs of a scenario with $s$ successes, $c_i$ the dollar cost of run $i$, $s_i \in \{0,1\}$ its success. Use pass@k when a human picks the best attempt, pass^k when the agent acts alone, the unbiased estimators for reporting with $n \ge k+2$, and cost per successful task as the number a customer compares.

## Chapter 22: MCP in Depth

$$
\text{code\_challenge} = \text{BASE64URL}\big(\text{SHA-256}(\text{code\_verifier})\big)
$$
**PKCE.** Symbols: the code verifier is a high-entropy random string the client keeps; the challenge is what it sends with the authorization request. Use it in every OAuth 2.1 authorization-code flow, which for MCP means every remote server holding a user-delegated token.

## Chapter 23: Multimodal and Document AI

$$
N = \frac{H}{P}\cdot\frac{W}{P}, \qquad \text{sim}(a,b) = 1 - \frac{\text{lev}(a,b)}{\max(|a|,|b|)}
$$
**Image token count and normalized edit similarity.** Symbols: $H, W$ the image height and width in pixels, $P$ the patch size, $N$ the image tokens before merging; $\text{lev}$ the Levenshtein edit distance between two field values. Use the first to price a page before running anything, the second as the fuzzy-match rule for extracted fields, with a threshold near 0.9.

$$
P = \frac{TP}{TP+FP}, \quad R = \frac{TP}{TP+FN}, \quad F_1 = \frac{2PR}{P+R}, \qquad c = c_{auto} + q\,c_{review} + (1-q)\epsilon\,c_{error}
$$
**Field metrics and the true cost per page.** Symbols: per-field true positives, false positives, and false negatives; $c_{auto}$ the pipeline cost per page, $q$ the review fraction, $c_{review}$ one human review, $\epsilon$ the error rate among auto-accepted documents, $c_{error}$ the cost of a wrong value reaching the downstream system. Report precision and recall separately, never only F1, and use the cost model to choose the confidence threshold that sets $q$.

## Chapter 25: Engagements, from Discovery to Readout

$$
S_i = \sum_{j=1}^{5}w_j s_{ij}, \qquad n = \frac{z^2 p(1-p)}{E^2}
$$
**Use-case scoring and golden-set size.** Symbols: $s_{ij}$ use case $i$'s score on criterion $j$ and $w_j$ the criterion weight; $p$ the expected accuracy, $E$ the half-width you need, $z = 1.96$. Use the first to make a selection defensible in front of stakeholders, the second to justify a 300-item golden set rather than a 100-item one.

$$
B = \underbrace{aThc}_{\text{hours saved}} + \underbrace{aT(e_0-e_1)C_e}_{\text{errors avoided}} + R, \quad C_{month} = aTq + C_{infra} + C_{people} + C_{review}, \quad N = B - C_{month}, \quad P = \frac{C_0}{N}, \quad a^{*} = \frac{C_{infra} + C_{people} + C_{review}}{T\big(hc + (e_0-e_1)C_e - q\big)}
$$
**The ROI model.** Symbols: $a$ adoption, $T$ eligible tasks per month, $h$ hours saved per task, $c$ the loaded hourly cost, $e_0$ and $e_1$ error rates before and after, $C_e$ the cost of one error, $R$ any revenue effect, $q$ the blended model cost per task, $C_0$ the one-time delivery cost, $P$ the payback in months, $a^{*}$ the break-even adoption. Use numbers the customer gave you, state every assumption, and put $a^{*}$ in the readout: it names what the champion must deliver for the project to pay.
