# Paper results

Only compact aggregate tables, figures, and machine-readable summaries used by the paper belong here. Raw generations,
rendered prompts, traces, caches, and intermediate reports stay under ignored `runs/` or the external artifact release.

`kg-benchmark analyze run` creates content-addressed `analysis_<hash>/` directories. Each contains a summary, endpoint
estimates, paired contrasts, diagnosis metrics, paper-ready Markdown tables, and a manifest binding every input and
output hash. Verify a package with `kg-benchmark analyze status --result-dir results/<analysis-id>`.
