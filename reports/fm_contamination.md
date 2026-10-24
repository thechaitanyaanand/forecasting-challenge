# Foundation-model contamination check

As of 2026-10-24. Question: did any benchmarked zero-shot model see Azure VM / cloud-cluster traces in pretraining? Only pages that were actually opened are cited.

| Model | Pretraining corpus summary | Cloud-trace data present? | Flagged | Sources |
|---|---|---|---|---|
| `amazon/chronos-bolt-base` | Card: "nearly 100 billion time series observations"; individual datasets not named. Original Chronos corpus (Monash, M4, energy, Wiki, mobility, weather, KernelSynth) names no cloud traces. Bolt corpus itself is undocumented. | Not documented (none in the original Chronos corpus) | No (unverified) | [model card](https://huggingface.co/amazon/chronos-bolt-base), [Chronos paper, App. B](https://arxiv.org/html/2403.07815), [Chronos datasets](https://huggingface.co/datasets/autogluon/chronos_datasets) |
| `amazon/chronos-2` | Subset of Chronos datasets + subset of GIFT-Eval Pretrain + synthetic (TSI, TCM, KernelSynth; multivariate fully synthetic). Paper Table 6 lists a "Cloud Ops" domain: Alibaba Cluster Trace (2018), Azure VM Traces (2017), Borg Cluster Data (2011), 100,000 series each. | Yes: Azure VM Traces 2017, Alibaba 2018, Borg 2011 | Yes | [model card](https://huggingface.co/amazon/chronos-2), [Chronos-2 paper](https://arxiv.org/html/2510.15821) |
| `google/timesfm-2.5-200m-pytorch` | Card: GiftEvalPretrain, Wikimedia pageviews, Google Trends, synthetic. GiftEvalPretrain sources include CloudOps TSF, which holds Azure VM Traces 2017, Borg 2011, Alibaba 2018 (GIFT-Eval paper, Table 14). | Yes, via GiftEvalPretrain | Yes | [model card](https://huggingface.co/google/timesfm-2.5-200m-pytorch), [GIFT-Eval paper](https://arxiv.org/html/2410.10393), [TimesFM paper](https://arxiv.org/html/2310.10688) (no cloud traces in the original TimesFM) |

Notes: the Azure trace in pretraining is Azure Public Dataset V1 (2017, via CloudOps TSF); this benchmark uses a different Azure Public Dataset release, but the same provider and workload family, so leakage of fleet-level patterns cannot be excluded. The TimesFM 2.5 flag is inferred from the card naming GiftEvalPretrain plus the GIFT-Eval paper's corpus composition; the card does not list the sources itself.

Conclusion: Chronos-2 and TimesFM 2.5 saw Azure VM, Borg and Alibaba cluster traces in pretraining and are flagged; they are benchmarked and reported but not eligible for selection. Chronos-Bolt-base documents no cloud-trace data, but its corpus is not public, so it is treated as unflagged with that caveat and its zero-shot result should be read with care.
