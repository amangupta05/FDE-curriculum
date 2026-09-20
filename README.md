# The Forward Deployed AI Engineer Handbook

Self-contained learning material for [FDE_AI_Roadmap.md](../FDE_AI_Roadmap.md). The roadmap says what to build and when; the [guides](../guides/README.md) walk through each project; this handbook explains everything underneath so that you do not need to leave it: derivations, worked numerical examples on your own hardware, diagrams, short implementation listings where code is the clearest explanation, failure-mode tables, and exercises with solutions.

Read it by phase alongside the roadmap, or cover to cover as a course. Start with the front matter. A single-file build, `FDE_Handbook.md`, is generated from the chapters by `build_handbook.py` for viewers that prefer one document.

## Chapters

### Part I: Foundations (Phase 0)

| Chapter | File | Serves |
|---|---|---|
| Front matter: how to read, notation, chapter map | [00-Front-Matter.md](00-Front-Matter.md) | all |
| 1. Tokenization and Data Representation | [01-Tokenization-and-Data-Representation.md](01-Tokenization-and-Data-Representation.md) | P0.2, P1.1 |
| 2. The Transformer, Derived | [02-The-Transformer-Derived.md](02-The-Transformer-Derived.md) | P0.2, P1.1 |
| 3. Training Dynamics | [03-Training-Dynamics.md](03-Training-Dynamics.md) | P0.2, P1.1, P1.2 |
| 4. Memory, Compute, and Throughput | [04-Memory-Compute-and-Throughput.md](04-Memory-Compute-and-Throughput.md) | every training and serving project |
| 5. The Working Environment | [05-The-Working-Environment.md](05-The-Working-Environment.md) | P0.1 and all |

### Part II: Training and Fine-Tuning (Phase 1)

| Chapter | File | Serves |
|---|---|---|
| 6. Pretraining at Small Scale | [06-Pretraining-at-Small-Scale.md](06-Pretraining-at-Small-Scale.md) | P1.1 |
| 7. Supervised Fine-Tuning, LoRA, and QLoRA | [07-Supervised-Fine-Tuning-LoRA-and-QLoRA.md](07-Supervised-Fine-Tuning-LoRA-and-QLoRA.md) | P1.2, P1.5, P4.3 |
| 8. Preference Optimization and Reinforcement Learning | [08-Preference-Optimization-and-Reinforcement-Learning.md](08-Preference-Optimization-and-Reinforcement-Learning.md) | P1.3 |
| 9. Embeddings, Retrieval, and Rerankers | [09-Embeddings-Retrieval-and-Rerankers.md](09-Embeddings-Retrieval-and-Rerankers.md) | P1.4 |
| 10. Synthetic Data and Distillation | [10-Synthetic-Data-and-Distillation.md](10-Synthetic-Data-and-Distillation.md) | P1.5, P3.1 |
| 11. Evaluation and Statistics | [11-Evaluation-and-Statistics.md](11-Evaluation-and-Statistics.md) | P1.6 and all |

### Part III: Inference, Serving, and Infrastructure (Phase 2)

| Chapter | File | Serves |
|---|---|---|
| 12. Quantization | [12-Quantization.md](12-Quantization.md) | P2.1, P2.2 |
| 13. Serving Systems | [13-Serving-Systems.md](13-Serving-Systems.md) | P2.2, P2.5, P3.1 |
| 14. Kubernetes for Model Serving | [14-Kubernetes-for-Model-Serving.md](14-Kubernetes-for-Model-Serving.md) | P2.3, P2.4, P5.2 |
| 15. Infrastructure as Code and CI/CD | [15-Infrastructure-as-Code-and-CI-CD.md](15-Infrastructure-as-Code-and-CI-CD.md) | P2.4, P3.1, P5.2 |
| 16. Gateways, Routing, and Cost Engineering | [16-Gateways-Routing-and-Cost-Engineering.md](16-Gateways-Routing-and-Cost-Engineering.md) | P2.5, P3.1, P3.4 |

### Part IV: LLMOps (Phase 3)

| Chapter | File | Serves |
|---|---|---|
| 17. The Data Flywheel | [17-The-Data-Flywheel.md](17-The-Data-Flywheel.md) | P3.1, P5.2 |
| 18. Evaluation Platforms and Monitoring | [18-Evaluation-Platforms-and-Monitoring.md](18-Evaluation-Platforms-and-Monitoring.md) | P3.2, P3.4 |
| 19. Security for LLM Systems | [19-Security-for-LLM-Systems.md](19-Security-for-LLM-Systems.md) | P3.3, P4.2, P5.2 |
| 20. Reliability and FinOps | [20-Reliability-and-FinOps.md](20-Reliability-and-FinOps.md) | P3.4, P5.2 |

### Part V: Agents and Product (Phase 4)

| Chapter | File | Serves |
|---|---|---|
| 21. Agents in Production | [21-Agents-in-Production.md](21-Agents-in-Production.md) | P4.1, P5.2 |
| 22. MCP in Depth | [22-MCP-in-Depth.md](22-MCP-in-Depth.md) | P4.2, P5.2 |
| 23. Multimodal and Document AI | [23-Multimodal-and-Document-AI.md](23-Multimodal-and-Document-AI.md) | P4.3, P5.2 |
| 24. Demos and Front-Ends | [24-Demos-and-Front-Ends.md](24-Demos-and-Front-Ends.md) | P4.4, P5.2 |

### Part VI: FDE Craft (Phase 5)

| Chapter | File | Serves |
|---|---|---|
| 25. Engagements, from Discovery to Readout | [25-Engagements-Discovery-to-Readout.md](25-Engagements-Discovery-to-Readout.md) | P5.1, P5.2, P5.3 |

### Appendices

| Appendix | File |
|---|---|
| A. Formula Sheet | [A-Formula-Sheet.md](A-Formula-Sheet.md) |
| B. Sizing Tables | [B-Sizing-Tables.md](B-Sizing-Tables.md) |
| C. Glossary | [C-Glossary.md](C-Glossary.md) |
| D. Bibliography | [D-Bibliography.md](D-Bibliography.md) |

## How each chapter is organized

Every chapter follows the same template so that you always know where to look: an opener stating what you will be able to do, where it is used in the roadmap, and prerequisites; the problem the chapter solves; numbered concept sections with intuition, precise statement, worked example, and consequence for practice; implementation notes with labeled listings; a failure-modes table; an on-your-machine section with concrete numbers for the RTX 4060, Kaggle T4s, and a rented A100; exercises with solutions; a summary; further reading.

## How to study a chapter

Read it once without stopping. Read it again with a notebook and re-derive every formula that has a worked example. Run the listings on the 4060 with your own small inputs. Do the exercises before opening the solutions. Then do the roadmap project the chapter serves, with the chapter open as the reference. The guide's self-check questions are the exam.

## Maintaining the handbook

Authors follow [AUTHORING.md](AUTHORING.md) for form and [OUTLINE.md](OUTLINE.md) for substance. Run `verify_handbook.py` before committing a change; it checks the template, the prose rules, the Mermaid rules, and the math delimiters. Run `build_handbook.py` to regenerate the single-file build.
