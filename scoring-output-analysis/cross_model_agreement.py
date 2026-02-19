"""Cross-model agreement analysis for Community Notes scoring output.

Analyzes how the multiple scoring models (Core, Expansion, Group,
MultiGroup) agree and disagree on note status, and examines which
model ultimately determines the final rating.

Works with the publicly downloadable noteStatusHistory file -- no need
to run the full scoring algorithm.

Usage:
    python -m scoring-output-analysis.cross_model_agreement \
        --note-status-history path/to/noteStatusHistory-00000.tsv
"""

import argparse
import logging
import sys
from itertools import combinations
from typing import Dict, List

import pandas as pd

from . import constants as c

logger = logging.getLogger("scoring_output_analysis")
logger.setLevel(logging.INFO)


def load_note_status_history(path: str) -> pd.DataFrame:
    """Load noteStatusHistory-00000.tsv into a DataFrame.

    Args:
        path: Path to noteStatusHistory TSV file.

    Returns:
        DataFrame with note status history data.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    logger.info(f"Loaded {len(df):,} notes from {path}")

    missing = []
    for col in [c.noteIdKey, c.currentStatusKey, c.currentDecidedByKey]:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def _get_available_models(df: pd.DataFrame) -> Dict[str, str]:
    """Return the subset of NSH_MODELS whose status column exists in the DataFrame."""
    available = {}
    for name, status_col in c.NSH_MODELS.items():
        if status_col in df.columns:
            available[name] = status_col
    return available


def compute_status_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the overall and per-model status distribution.

    For each model, computes the count and percentage of notes in each
    status category (CRH, CRNH, NMR), considering only notes that the
    model actually scored (non-empty status).

    Args:
        df: Note status history DataFrame.

    Returns:
        DataFrame with columns [model, status, count, pct, total_scored].
    """
    models = _get_available_models(df)
    rows = []

    # Overall current status distribution
    final_counts = df[c.currentStatusKey].value_counts()
    total = len(df)
    for status in c.ALL_STATUSES:
        count = final_counts.get(status, 0)
        rows.append(
            {
                "model": "CurrentStatus",
                "status": status,
                "count": int(count),
                "pct": round(100 * count / total, 2) if total > 0 else 0,
                "total_scored": total,
            }
        )

    # Per-model distributions
    for model_name, status_col in models.items():
        scored = df[df[status_col].notna() & (df[status_col] != "")]
        total_scored = len(scored)
        if total_scored == 0:
            continue
        model_counts = scored[status_col].value_counts()
        for status in c.ALL_STATUSES:
            count = model_counts.get(status, 0)
            rows.append(
                {
                    "model": model_name,
                    "status": status,
                    "count": int(count),
                    "pct": round(100 * count / total_scored, 2),
                    "total_scored": total_scored,
                }
            )

    return pd.DataFrame(rows)


def compute_pairwise_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise agreement between all model pairs.

    For each pair of models, only considers notes where both models
    produced a non-empty status. Computes agreement rate and Cohen's
    kappa statistic.

    Args:
        df: Note status history DataFrame.

    Returns:
        DataFrame with columns [model_a, model_b, n_both_scored,
        agreement_rate, cohens_kappa, both_crh, both_crnh, both_nmr,
        disagree_count].
    """
    models = _get_available_models(df)
    rows = []

    for (name_a, col_a), (name_b, col_b) in combinations(models.items(), 2):
        both = df[
            df[col_a].notna() & (df[col_a] != "")
            & df[col_b].notna() & (df[col_b] != "")
        ]
        n = len(both)
        if n == 0:
            continue

        status_a = both[col_a]
        status_b = both[col_b]

        agree = (status_a == status_b).sum()
        both_crh = (
            (status_a == c.CURRENTLY_RATED_HELPFUL) & (status_b == c.CURRENTLY_RATED_HELPFUL)
        ).sum()
        both_crnh = (
            (status_a == c.CURRENTLY_RATED_NOT_HELPFUL)
            & (status_b == c.CURRENTLY_RATED_NOT_HELPFUL)
        ).sum()
        both_nmr = (
            (status_a == c.NEEDS_MORE_RATINGS) & (status_b == c.NEEDS_MORE_RATINGS)
        ).sum()
        disagree = n - agree

        kappa = _cohens_kappa(status_a, status_b, c.ALL_STATUSES)

        rows.append(
            {
                "model_a": name_a,
                "model_b": name_b,
                "n_both_scored": n,
                "agreement_rate": round(100 * agree / n, 2),
                "cohens_kappa": round(kappa, 4),
                "both_crh": int(both_crh),
                "both_crnh": int(both_crnh),
                "both_nmr": int(both_nmr),
                "disagree_count": int(disagree),
            }
        )

    return pd.DataFrame(rows)


def _cohens_kappa(
    series_a: pd.Series, series_b: pd.Series, labels: List[str]
) -> float:
    """Compute Cohen's kappa for two categorical series."""
    n = len(series_a)
    if n == 0:
        return 0.0

    observed_agreement = (series_a == series_b).sum() / n

    expected_agreement = 0.0
    for label in labels:
        p_a = (series_a == label).sum() / n
        p_b = (series_b == label).sum() / n
        expected_agreement += p_a * p_b

    if expected_agreement >= 1.0:
        return 0.0

    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def analyze_decided_by(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze the distribution of which model decides the final status.

    The `currentDecidedBy` column indicates which scoring model's status
    was selected as the final note status by the meta-scorer.

    Args:
        df: Note status history DataFrame.

    Returns:
        DataFrame with columns [decided_by, count, pct, crh_count,
        crnh_count, nmr_count].
    """
    valid = df[df[c.currentDecidedByKey].notna() & (df[c.currentDecidedByKey] != "")]
    total = len(valid)
    if total == 0:
        return pd.DataFrame()

    rows = []
    decided_counts = valid.groupby(c.currentDecidedByKey).agg(
        count=(c.noteIdKey, "count"),
    ).reset_index()

    for _, row in decided_counts.iterrows():
        decided_by = row[c.currentDecidedByKey]
        subset = valid[valid[c.currentDecidedByKey] == decided_by]
        status_counts = subset[c.currentStatusKey].value_counts()
        rows.append(
            {
                "decided_by": decided_by,
                "count": int(row["count"]),
                "pct": round(100 * row["count"] / total, 2),
                "crh_count": int(status_counts.get(c.CURRENTLY_RATED_HELPFUL, 0)),
                "crnh_count": int(status_counts.get(c.CURRENTLY_RATED_NOT_HELPFUL, 0)),
                "nmr_count": int(status_counts.get(c.NEEDS_MORE_RATINGS, 0)),
            }
        )

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result = result.sort_values("count", ascending=False).reset_index(drop=True)
    return result


def find_disagreement_cases(
    df: pd.DataFrame, max_patterns: int = 20
) -> pd.DataFrame:
    """Identify and categorize notes where models disagree.

    Groups disagreements by the pattern of status assignments across
    models. For example, "Core=CRH, Expansion=NMR" would be one pattern.

    Args:
        df: Note status history DataFrame.
        max_patterns: Maximum number of disagreement patterns to return.

    Returns:
        DataFrame with columns [pattern, count, pct_of_disagreements,
        decided_by_distribution].
    """
    models = _get_available_models(df)
    if len(models) < 2:
        return pd.DataFrame()

    model_names = list(models.keys())
    status_cols = list(models.values())

    # Only look at notes scored by at least 2 models
    scored_counts = sum(
        (df[col].notna() & (df[col] != "")).astype(int) for col in status_cols
    )
    multi_scored = df[scored_counts >= 2].copy()

    if len(multi_scored) == 0:
        return pd.DataFrame()

    abbrev = {
        "CURRENTLY_RATED_HELPFUL": "CRH",
        "CURRENTLY_RATED_NOT_HELPFUL": "CRNH",
        "NEEDS_MORE_RATINGS": "NMR",
    }

    def _make_pattern(row):
        parts = []
        for name in model_names:
            col = models[name]
            val = row[col]
            if pd.notna(val) and val != "":
                parts.append(f"{name}={abbrev.get(val, val)}")
        return " | ".join(parts)

    def _has_disagreement(row):
        statuses = set()
        for name in model_names:
            col = models[name]
            val = row[col]
            if pd.notna(val) and val != "":
                statuses.add(val)
        return len(statuses) > 1

    multi_scored["_pattern"] = multi_scored.apply(_make_pattern, axis=1)
    disagreements = multi_scored[multi_scored.apply(_has_disagreement, axis=1)]

    logger.info(
        f"{len(disagreements):,} notes ({100*len(disagreements)/len(multi_scored):.1f}% "
        f"of multi-scored) have model disagreements"
    )

    if len(disagreements) == 0:
        return pd.DataFrame()

    pattern_groups = disagreements.groupby("_pattern")
    rows = []
    total_disagreements = len(disagreements)

    for pattern, group in pattern_groups:
        decided_by_dist = group[c.currentDecidedByKey].value_counts().head(3).to_dict()
        rows.append(
            {
                "pattern": pattern,
                "count": len(group),
                "pct_of_disagreements": round(
                    100 * len(group) / total_disagreements, 2
                ),
                "decided_by_distribution": decided_by_dist,
            }
        )

    result = pd.DataFrame(rows).sort_values("count", ascending=False).head(max_patterns)
    return result.reset_index(drop=True)


def analyze_status_flips(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze notes that changed status (flipped between CRH/CRNH/NMR).

    Uses firstNonNMRStatus vs mostRecentNonNMRStatus to detect flips.

    Args:
        df: Note status history DataFrame.

    Returns:
        DataFrame with columns [flip_type, count, pct].
    """
    has_first = df[c.firstNonNMRStatusKey].notna() & (df[c.firstNonNMRStatusKey] != "")
    has_latest = df[c.mostRecentNonNMRStatusKey].notna() & (df[c.mostRecentNonNMRStatusKey] != "")
    ever_rated = df[has_first & has_latest]

    if len(ever_rated) == 0:
        return pd.DataFrame()

    flipped = ever_rated[
        ever_rated[c.firstNonNMRStatusKey] != ever_rated[c.mostRecentNonNMRStatusKey]
    ]

    rows = []
    total = len(ever_rated)
    rows.append(
        {
            "flip_type": "TOTAL EVER RATED (non-NMR)",
            "count": total,
            "pct": 100.0,
        }
    )
    rows.append(
        {
            "flip_type": "Flipped status",
            "count": len(flipped),
            "pct": round(100 * len(flipped) / total, 2),
        }
    )
    rows.append(
        {
            "flip_type": "Never flipped",
            "count": total - len(flipped),
            "pct": round(100 * (total - len(flipped)) / total, 2),
        }
    )

    if len(flipped) > 0:
        flip_types = (
            flipped[c.firstNonNMRStatusKey] + " -> " + flipped[c.mostRecentNonNMRStatusKey]
        ).value_counts()
        for flip_type, count in flip_types.items():
            rows.append(
                {
                    "flip_type": f"  {flip_type}",
                    "count": int(count),
                    "pct": round(100 * count / len(flipped), 2),
                }
            )

    return pd.DataFrame(rows)


def analyze_modeling_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze the distribution of modeling groups and their outcomes.

    Args:
        df: Note status history DataFrame.

    Returns:
        DataFrame with columns [modeling_group, count, crh_pct, crnh_pct, nmr_pct].
    """
    if c.currentModelingGroupKey not in df.columns:
        return pd.DataFrame()

    has_group = df[df[c.currentModelingGroupKey].notna() & (df[c.currentModelingGroupKey] != 0)]
    if len(has_group) == 0:
        return pd.DataFrame()

    rows = []
    for group_id, group_df in has_group.groupby(c.currentModelingGroupKey):
        total = len(group_df)
        status_counts = group_df[c.currentStatusKey].value_counts()
        rows.append(
            {
                "modeling_group": int(group_id),
                "count": total,
                "crh_count": int(status_counts.get(c.CURRENTLY_RATED_HELPFUL, 0)),
                "crh_pct": round(
                    100 * status_counts.get(c.CURRENTLY_RATED_HELPFUL, 0) / total, 1
                ),
                "crnh_pct": round(
                    100 * status_counts.get(c.CURRENTLY_RATED_NOT_HELPFUL, 0) / total, 1
                ),
                "nmr_pct": round(
                    100 * status_counts.get(c.NEEDS_MORE_RATINGS, 0) / total, 1
                ),
            }
        )

    result = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    return result


def print_summary_report(df: pd.DataFrame) -> None:
    """Print a human-readable summary report to stdout."""
    models = _get_available_models(df)

    print("=" * 72)
    print("COMMUNITY NOTES: CROSS-MODEL AGREEMENT ANALYSIS")
    print("=" * 72)
    print(f"\nTotal notes: {len(df):,}")
    print(f"Models available: {', '.join(models.keys())}")

    # 1. Status distribution
    print("\n" + "-" * 72)
    print("1. STATUS DISTRIBUTION")
    print("-" * 72)
    dist = compute_status_distribution(df)
    for model_name in ["CurrentStatus"] + list(models.keys()):
        model_dist = dist[dist["model"] == model_name]
        if len(model_dist) == 0:
            continue
        total = model_dist["total_scored"].iloc[0]
        print(f"\n  {model_name} ({total:,} notes scored):")
        for _, row in model_dist.iterrows():
            bar = "#" * int(row["pct"] / 2)
            print(
                f"    {row['status']:>30s}: {row['count']:>10,} "
                f"({row['pct']:>5.1f}%) {bar}"
            )

    # 2. Pairwise agreement
    print("\n" + "-" * 72)
    print("2. PAIRWISE MODEL AGREEMENT")
    print("-" * 72)
    agreement = compute_pairwise_agreement(df)
    if len(agreement) > 0:
        print(
            f"\n  {'Model A':<14s} {'Model B':<14s} {'N':>10s} "
            f"{'Agree%':>8s} {'Kappa':>8s} {'Disagree':>10s}"
        )
        print("  " + "-" * 64)
        for _, row in agreement.iterrows():
            print(
                f"  {row['model_a']:<14s} {row['model_b']:<14s} "
                f"{row['n_both_scored']:>10,} {row['agreement_rate']:>7.1f}% "
                f"{row['cohens_kappa']:>8.4f} {row['disagree_count']:>10,}"
            )
    else:
        print("  No model pairs found with jointly-scored notes.")

    # 3. Decided-by distribution
    print("\n" + "-" * 72)
    print("3. DECIDED-BY DISTRIBUTION")
    print("-" * 72)
    decided = analyze_decided_by(df)
    if len(decided) > 0:
        print(
            f"\n  {'Decided By':<35s} {'Count':>8s} {'Pct':>7s} "
            f"{'CRH':>8s} {'CRNH':>8s} {'NMR':>8s}"
        )
        print("  " + "-" * 74)
        for _, row in decided.iterrows():
            print(
                f"  {row['decided_by']:<35s} {row['count']:>8,} "
                f"{row['pct']:>6.1f}% {row['crh_count']:>8,} "
                f"{row['crnh_count']:>8,} {row['nmr_count']:>8,}"
            )

    # 4. Disagreement patterns
    print("\n" + "-" * 72)
    print("4. TOP DISAGREEMENT PATTERNS")
    print("-" * 72)
    disagreements = find_disagreement_cases(df)
    if len(disagreements) > 0:
        for i, row in disagreements.head(10).iterrows():
            print(
                f"\n  Pattern #{i+1} "
                f"({row['count']:,} notes, "
                f"{row['pct_of_disagreements']:.1f}% of disagreements):"
            )
            print(f"    {row['pattern']}")
            print(f"    Decided by: {row['decided_by_distribution']}")
    else:
        print("  No disagreement patterns found.")

    # 5. Status flips
    print("\n" + "-" * 72)
    print("5. STATUS FLIP ANALYSIS")
    print("-" * 72)
    flips = analyze_status_flips(df)
    if len(flips) > 0:
        for _, row in flips.iterrows():
            print(f"  {row['flip_type']:<45s} {row['count']:>10,} ({row['pct']:>5.1f}%)")
    else:
        print("  No status flip data available.")

    # 6. Modeling groups
    print("\n" + "-" * 72)
    print("6. MODELING GROUP OUTCOMES")
    print("-" * 72)
    groups = analyze_modeling_groups(df)
    if len(groups) > 0:
        print(
            f"\n  {'Group':>6s} {'Notes':>10s} {'CRH':>8s} "
            f"{'CRH%':>7s} {'CRNH%':>7s} {'NMR%':>7s}"
        )
        print("  " + "-" * 46)
        for _, row in groups.iterrows():
            print(
                f"  {row['modeling_group']:>6.0f} {row['count']:>10,} "
                f"{row['crh_count']:>8,} {row['crh_pct']:>6.1f}% "
                f"{row['crnh_pct']:>6.1f}% {row['nmr_pct']:>6.1f}%"
            )
    else:
        print("  No modeling group data available.")

    print("\n" + "=" * 72)
    print("END OF REPORT")
    print("=" * 72)


def main():
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description="Cross-model agreement analysis for Community Notes scoring output."
    )
    parser.add_argument(
        "--note-status-history",
        required=True,
        help="Path to noteStatusHistory-00000.tsv (from public data download).",
    )
    args = parser.parse_args()

    df = load_note_status_history(args.note_status_history)
    print_summary_report(df)


if __name__ == "__main__":
    main()
