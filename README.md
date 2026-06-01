# Mathematical & MCQ Reasoning Alignment Pipeline

> **Solo Competitor:** Mohammad Abdullah
> Dual-stage SFT + Rejection Sampling pipeline for robust mathematical and multi-choice reasoning, optimized for strict LaTeX `\boxed{}` grading parsers.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Hardware & Performance](#hardware--performance)
4. [Directory Structure](#directory-structure)
5. [Installation](#installation)
6. [Reproducing Results](#reproducing-results)
7. [Pipeline Internals](#pipeline-internals)

---

## Overview

This repository implements a machine learning framework for solving complex mathematical challenges and multi-option reasoning problems. The system is built around a **dual-stage pipeline**:

- **Stage 1 — Supervised Fine-Tuning (SFT):** Trains the base model on curated reasoning traces.
- **Stage 2 — Rejection Sampling Alignment (RSA):** An offline self-correction protocol that eliminates token-formatting issues and aligns model outputs precisely with the strict LaTeX `\boxed{}` grading parser.

---

## Architecture

```
Input JSONL
    │
    ▼
┌─────────────────────────┐
│  Chat Template Formatter │  ← Parallel batch formatting
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  SFT Model Inference    │  ← Generates candidate reasoning traces
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Rejection Sampling     │  ← Filters & self-corrects generations
│  Alignment (RSA)        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Structural Fallback    │  ← Sanitizes malformed outputs
│  Sanitizer              │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Majority Voting        │  ← Aggregates across samples
└────────────┬────────────┘
             │
             ▼
    Output CSV (submission-ready)
```

---

## Hardware & Performance

| Property | Value |
|---|---|
| GPU | 1× NVIDIA H100 (Enterprise Accelerator) |
| Approximate Eval Time | ~15–20 minutes (full payload) |
| Scheduling | Optimized batch scheduling with majority voting |

---

## Directory Structure

```
.
├── data/
│   └── public.jsonl
├── results/
│   └── final_submission.csv
├── README.md
├── requirements.txt
├── submission_pipeline.py
├── judger.py
└── utils.p
```

---

## Installation

Install all required runtimes, parsers, and evaluation packages:

```bash
pip install torch transformers pandas datasets antlr4-python3-runtime==4.11.1 sympy
```

> **Note:** `antlr4-python3-runtime` must be pinned to `4.11.1` for compatibility with the symbolic math parser used in grading.

---

## Reproducing Results

There is a **single entry point** for end-to-end reproduction.

### Python API

Call `run_inference()` directly from a Python script or Jupyter notebook:

```python
from submission_pipeline import run_inference

# Running the pipeline dynamically streams the model weights from Hugging Face Hub
run_inference(
    input_jsonl_path="data/public.jsonl",
    output_csv_path="results/final_submission.csv"
)
```

### Shell / Script Mode

Alternatively, execute via the terminal:

```bash
python submission_pipeline.py
```

Both modes are equivalent. The entry point automatically handles:

- Environment setup and weight caching
- Parallel chat-template formatting
- Reasoning trace preservation
- Structural fallback sanitization
- Platform-compliant CSV output

---

## Pipeline Internals

| Component | Description |
|---|---|
| **Chat Template Formatter** | Applies model-specific prompt templates in parallel across the input batch |
| **SFT Inference** | Runs the fine-tuned model to produce candidate reasoning traces |
| **Rejection Sampling Alignment** | Offline self-correction — discards or repairs generations that fail the `\boxed{}` format constraint |
| **Fallback Sanitizer** | Catches residual structural issues before aggregation |
| **Majority Voting** | Aggregates across multiple sampled outputs to select the most consistent final answer |
| **CSV Exporter** | Writes a submission-ready file compatible with the Kaggle grading platform |
