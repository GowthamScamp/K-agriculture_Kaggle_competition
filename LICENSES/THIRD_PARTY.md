# Third-party notices

This repository's original code and documentation are licensed under the
repository-level [MIT License](../LICENSE). That license does not replace the
notices that apply to included third-party material.

| Material | Location | License and source | Notes |
|---|---|---|---|
| `lonespear`-derived competitive heuristic | `src/agents/competitive_meta.py`, `submission/main.py` | MIT, Jonathan Day's `lonespear/kaggriculture` (`main_v18.py`) | The upstream MIT notice is retained in each derived source artifact. |
| `tetsutani_dp` tape chassis | `benchmarks/public/tetsutani_dp/main.py`, `submission/tape/main.py` | Apache-2.0, `warlord-2227/kaggle` commit `f8ddefbef8393011363746b158692ab2ca89d9aa` | The original Apache notice and modification notice are retained in the files. |

The ignored `benchmarks/public/` cache may contain additional locally fetched
agents. It is intentionally excluded from Git so this repository does not
redistribute code whose provenance has not been reviewed for publication.
