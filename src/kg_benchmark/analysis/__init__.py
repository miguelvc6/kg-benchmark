"""Provider-free statistical analysis and paper result packaging."""

from kg_benchmark.analysis.statistics import (
    cluster_estimate,
    exact_mcnemar,
    holm_adjust,
    paired_cluster_contrast,
)
from kg_benchmark.analysis.workflow import (
    AnalysisWorkflowError,
    build_paper_results,
    replay_matrix_evaluations,
    verify_paper_results,
)

__all__ = [
    "AnalysisWorkflowError",
    "build_paper_results",
    "cluster_estimate",
    "exact_mcnemar",
    "holm_adjust",
    "paired_cluster_contrast",
    "replay_matrix_evaluations",
    "verify_paper_results",
]
