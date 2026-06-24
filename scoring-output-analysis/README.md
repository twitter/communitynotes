# Scoring Output Analysis

Downstream analysis of Community Notes scoring model output. Examines how the multiple parallel scoring models (Core, Expansion, Group, MultiGroup) agree and disagree on note status, and which model ultimately determines the final rating.

## Background

The Community Notes scoring algorithm runs [multiple models in parallel](https://communitynotes.x.com/guide/en/under-the-hood/ranking-notes), each producing independent note helpfulness scores. A meta-scoring layer then reconciles these into a single final status. This analysis examines agreement patterns across models.

## Setup

```bash
pip install numpy pandas
```

No additional dependencies required beyond numpy and pandas.

## Data

This analysis works directly with the publicly downloadable `noteStatusHistory-00000.tsv` file -- no need to run the full scoring algorithm.

1. Download the data from https://x.com/i/communitynotes/download-data
2. Place `noteStatusHistory-00000.tsv` in this directory (or pass a path via `--note-status-history`)

## Usage

From the repository root:

```bash
python -m scoring-output-analysis.cross_model_agreement \
    --note-status-history scoring-output-analysis/noteStatusHistory-00000.tsv
```

## Analyses

### cross_model_agreement.py

Produces a text report covering:

1. **Status distribution** -- CRH/CRNH/NMR/FIRM_REJECT counts per model and overall
2. **Pairwise agreement** -- Agreement rates and Cohen's kappa between all model pairs
3. **Decided-by distribution** -- Which model's status becomes the final status
4. **Disagreement patterns** -- Most common patterns when models disagree, and how they are resolved
5. **Status flip analysis** -- How often notes change status after initial rating
6. **Modeling group outcomes** -- CRH/CRNH/NMR rates across regional/language modeling groups

## Example Output (Feb 2026 data, 2.56M notes)

```
Core and Expansion agree 93.5% of the time (kappa=0.91).
Group models agree with Core/Expansion only ~57% (kappa ~0.16).
CoreModel decides 77% of all notes.
Only 65 of 457K ever-rated notes flipped status.
87.3% of all notes are NMR; 8.5% reach CRH.
```
