# Claim and gap ledger

Access date: 2026-09-07. Scope: latest locally available image and user-specified serving paths. Planning API unavailable. Discovery complete; targeted follow-up complete to available evidence; synthesis and Markdown structural verification complete. Runtime matrix remains explicitly incomplete.

| Claim family | Source / publisher / date | Evidence and confidence | Gap / next test |
|---|---|---|---|
| Latest image composition | Host Docker image/files/package inventory, collected 2026-09-07; latest-image-packages.txt, packages.txt | Three MD5s match v125-fixes Dockerfiles and PLAN; high | Model weights and loaded runtime libraries need fresh manifest |
| Active baseline differs from latest | Host docker inspect/ps, live-inventory.txt, 16:54 UTC | v125 digest plus active master127 job; high | Freeze identity for each future measurement |
| Fresh deep throughput | Host bench127-partial.log, 2026-09-07 | DFlash/FP8 136191 and261757 prompts, one sample; high for logged result, low for generalization | Completed matrix and repetitions |
| Periodic output tail | Host loop8-partial.out, 2026-09-07 | Character detector found !!!! periodicity in1/4 selected cases; medium | Raw stream and paired nonspec/auto reproduction |
| Adaptive narrowing unsafe | v125-fixes/PLAN.md and llm_base_proposer_v53.py, local project, 2026-09-07 | Logged wedge and fixed-placeholder contract; high risk, mechanism not instruction-level proven | Scheduler-owned width protocol and transition suite |
| Ragged routing not observed | v125-fixes/PLAN.md, local project, 2026-09-07 | Reported2000 uniform calls; medium-high historical | Per-workload routing counter reproduction |
| TQ conversion overhead | turboquant_attn_v53.py, local project; source inspection2026-09-07 | Full-prefix dequant, rotate and concatenate; high | Worker allocation/timing profile |
| Dynamic SD compatibility | vLLM Dynamic Speculative Decoding, vLLM project, updated2026-07-07; https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/ | MRv1 piecewise/full MRv2 constraint explicitly stated; high upstream | Installed XPU compatibility is not established by docs |
| Speculation workload dependence | vLLM Speculative Decoding, vLLM project, accessed2026-09-07; https://docs.vllm.ai/en/latest/features/speculative_decoding/ | Low/moderate-QPS memory-bound purpose and workload-dependent gains; high | B70 measurements govern actual configuration |
| Chunk budget tradeoff | vLLM Optimization and Tuning, vLLM project, accessed2026-09-07; https://docs.vllm.ai/en/latest/configuration/optimization/ | Smaller budget favors ITL, larger favors TTFT; high general guidance | Local16384 historical regression requires controlled retest |
| KV scales | vLLM Quantized KV Cache, vLLM project, updated2026-08-03; https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/ | Calibration and hardware/backend specificity; high | Patched XPU scale-layout compatibility |
| Older model processor issue | Intel llm-scaler issue420, issue authors/Intel repository, accessed2026-09-07; https://github.com/intel/llm-scaler/issues/420 | Different0.14 image; discovery-only applicability | Regression fixture, not current defect attribution |

Searches: Intel llm-scaler issues TurboQuant/DFlash; official vLLM speculation, KV quantization and optimization; targeted dynamic SD follow-up. Local sources: all issue-family headings, relevant issue sections, existing report/evidence, v53 source/Dockerfiles/post-mortems; host live process and benchmark snapshots. Broad public issue enumeration was not completed. Two independent research workers were attempted under the skill; both ended with service usage-limit errors, and their unverified work is not treated as completed research. Parent directly verified consequential local and upstream claims.

Stop reason: further broad sources cannot resolve absent controlled runtime measurements; all material gaps are listed in the report rather than inferred as passing. No inference load was added to the concurrently running benchmark.
