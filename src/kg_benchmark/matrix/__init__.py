"""Materialized, resumable experiment-matrix orchestration."""

from kg_benchmark.matrix.workflow import (
    MatrixWorkflowError,
    dry_run_matrix,
    execute_matrix,
    matrix_status,
    plan_matrix,
)

__all__ = [
    "MatrixWorkflowError",
    "dry_run_matrix",
    "execute_matrix",
    "matrix_status",
    "plan_matrix",
]
