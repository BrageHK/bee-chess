"""Bee Chess training and dataset generation.

This package is a stub for the bootstrap PR. Versioned training-example
and model-manifest schemas, dataset validation, and the PyTorch training
loop land in follow-up PRs (see `feat/training-schema`). Per
`docs/adr/0001-v1-engine-architecture.md`, Python owns training and
dataset generation; the Rust engine consumes exported model artifacts
through a versioned contract rather than depending on this package
directly.
"""

__version__ = "0.1.0"
