import argparse
import logging
import os
import sys

from . import constants as c
from .enums import scorers_from_csv
from .process_data import (
  LocalDataLoader,
  _restore_participant_ids,
  apply_participant_id_mapping,
  tsv_reader,
  write_parquet_local,
  write_tsv_local,
)
from .run_scoring import (
  run_contributor_scoring,
  run_final_note_scoring,
  run_prescoring,
  run_rater_clustering,
  run_scoring,
)

import joblib
import pandas as pd


logger = logging.getLogger("birdwatch.runner")
logger.setLevel(logging.INFO)


# Subdirectory under --outdir for intermediate results between phases.
_INTERMEDIATES_DIR = "_intermediates"


def parse_args():
  parser = argparse.ArgumentParser("Community Notes Scoring")
  parser.add_argument(
    "--check-flips",
    dest="check_flips",
    help="Validate that note statuses align with prior runs (disable for testing)",
    action="store_true",
  )
  parser.add_argument(
    "--nocheck-flips",
    help="Disable validation that note statuses align with prior runs (use for testing)",
    action="store_false",
    dest="check_flips",
  )
  parser.set_defaults(check_flips=False)
  parser.add_argument(
    "--enforce-types",
    dest="enforce_types",
    help="Raise errors when types in Pandas operations do not meet expectations.",
    action="store_true",
  )
  parser.add_argument(
    "--noenforce-types",
    dest="enforce_types",
    help="Log to stderr when types in Pandas operations do not meet expectations.",
    action="store_false",
  )
  parser.set_defaults(enforce_types=False)
  parser.add_argument(
    "-e", "--enrollment", default=c.enrollmentInputPath, help="note enrollment dataset"
  )
  parser.add_argument(
    "--epoch-millis",
    default=None,
    type=float,
    dest="epoch_millis",
    help="timestamp in milliseconds since epoch to treat as now",
  )
  parser.add_argument(
    "--headers",
    dest="headers",
    help="First row of input files should be a header",
    action="store_true",
  )
  parser.add_argument(
    "--noheaders",
    dest="headers",
    help="First row of input files should be data.  There should be no headers.",
    action="store_false",
  )
  parser.set_defaults(headers=True)
  parser.add_argument("-n", "--notes", default=c.notesInputPath, help="note dataset")
  parser.add_argument(
    "--previous-scored-notes", default=None, help="previous scored notes dataset path"
  )
  parser.add_argument(
    "--previous-aux-note-info", default=None, help="previous aux note info dataset path"
  )
  parser.add_argument(
    "--previous-rating-cutoff-millis",
    default=None,
    type=int,
    help="previous rating cutoff millis",
  )
  parser.add_argument("-o", "--outdir", default=".", help="directory for output files")
  parser.add_argument(
    "--pseudoraters",
    dest="pseudoraters",
    help="Include calculation of pseudorater intervals",
    action="store_true",
  )
  parser.add_argument(
    "--nopseudoraters",
    dest="pseudoraters",
    help="Exclude calculation of pseudorater intervals (faster)",
    action="store_false",
  )
  parser.set_defaults(pseudoraters=True)
  parser.add_argument("-r", "--ratings", default=c.ratingsInputPath, help="rating dataset")
  parser.add_argument(
    "--scorers",
    default=None,
    type=scorers_from_csv,
    help="CSV list of scorers to enable.",
  )
  parser.add_argument(
    "--seed", default=None, type=int, help="set to an int to seed matrix factorization"
  )
  parser.add_argument(
    "-s",
    "--status",
    default=c.noteStatusHistoryInputPath,
    help="note status history dataset",
  )
  parser.add_argument(
    "--strict-columns",
    dest="strict_columns",
    help="Explicitly select columns and require that expected columns are present.",
    action="store_true",
  )
  parser.add_argument(
    "--nostrict-columns",
    help="Disable validation of expected columns and allow unexpected columns.",
    action="store_false",
    dest="strict_columns",
  )
  parser.set_defaults(strict_columns=True)
  parser.add_argument(
    "--parallel",
    help="Enable parallel run of algorithm.",
    action="store_true",
    dest="parallel",
  )
  parser.set_defaults(parallel=False)
  parser.add_argument(
    "--max-workers",
    type=int,
    default=None,
    dest="maxWorkers",
    help="Maximum number of parallel worker processes. Defaults to 6 for prescoring, "
    "4-6 for final scoring. Lower values reduce memory usage.",
  )

  parser.add_argument(
    "--no-parquet",
    help="Disable writing parquet files.",
    default=False,
    action="store_true",
    dest="no_parquet",
  )

  parser.add_argument(
    "--cutoff-timestamp-millis",
    default=None,
    type=int,
    dest="cutoffTimestampMillis",
    help="filter notes and ratings created after this time.",
  )
  parser.add_argument(
    "--exclude-ratings-after-a-note-got-first-status-plus-n-hours",
    default=None,
    type=int,
    dest="excludeRatingsAfterANoteGotFirstStatusPlusNHours",
    help="Exclude ratings after a note got first status plus n hours",
  )
  parser.add_argument(
    "--days-in-past-to-apply-post-first-status-filtering",
    default=14,
    type=int,
    dest="daysInPastToApplyPostFirstStatusFiltering",
    help="Days in past to apply post first status filtering",
  )
  parser.add_argument(
    "--prescoring-delay-hours",
    default=None,
    type=int,
    dest="prescoring_delay_hours",
    help="Filter prescoring input to simulate delay in hours",
  )
  parser.add_argument(
    "--sample-ratings",
    default=0.0,
    type=float,
    dest="sample_ratings",
    help="Set to sample ratings at random.",
  )
  parser.add_argument(
    "--skip-pss",
    action="store_true",
    default=False,
    dest="skip_pss",
    help=(
      "Skip the Post Selection Similarity (run_rater_clustering) step and feed an empty "
      "PSS DataFrame to prescoring/final scoring. Useful for fast validation runs that "
      "want to exercise the rest of the pipeline without the multi-hour PSS pair-counts "
      "step. Output scores differ slightly because PSS-based correlated-rater "
      "suppression is disabled."
    ),
  )
  parser.add_argument(
    "--phase",
    default="all",
    choices=["all", "pss", "prescoring", "final", "contributor"],
    help=(
      "Run a single scoring phase instead of all phases in one process. Each phase is a "
      "separate OS process invoked sequentially by the caller, so memory is fully "
      "reclaimed between phases (matching production where each phase is its own binary). "
      "Intermediate results are written to --outdir/_intermediates/. "
      "Use 'all' (default) to run the entire pipeline in one process; useful for tests "
      "and for environments without enough disk for intermediates."
    ),
  )
  return parser.parse_args()


def _intermediates_path(args):
  p = os.path.join(args.outdir, _INTERMEDIATES_DIR)
  os.makedirs(p, exist_ok=True)
  return p


def _load_previous_outputs(args):
  """Load --previous-scored-notes and --previous-aux-note-info if both are set."""
  if args.previous_scored_notes is None:
    return None, None
  previousScoredNotes = tsv_reader(
    args.previous_scored_notes,
    c.noteModelOutputTSVTypeMapping,
    c.noteModelOutputTSVColumns,
    header=False,
    convertNAToNone=False,
  )
  assert (
    args.previous_aux_note_info is not None
  ), "previous_aux_note_info must be available if previous_scored_notes is available"
  previousAuxiliaryNoteInfo = tsv_reader(
    args.previous_aux_note_info,
    c.auxiliaryScoredNotesTSVTypeMapping,
    c.auxiliaryScoredNotesTSVColumns,
    header=False,
    convertNAToNone=False,
  )
  return previousScoredNotes, previousAuxiliaryNoteInfo


def _load_data(args, dataLoader=None, extraFrames=None):
  """Load input data with participant IDs already normalized for scoring."""
  if args.epoch_millis:
    c.epochMillis = args.epoch_millis
    c.useCurrentTimeInsteadOfEpochMillisForNoteStatusHistory = False

  if dataLoader is None:
    dataLoader = LocalDataLoader(
      args.notes,
      args.ratings,
      args.status,
      args.enrollment,
      args.headers,
      normalizeParticipantIds=True,
    )
  notes, ratings, statusHistory, userEnrollment = dataLoader.get_data()
  if not hasattr(dataLoader, "get_participant_id_reverse_mapping"):
    raise ValueError("Public runner requires a LocalDataLoader with participant ID normalization")
  reverseIdMapping = dataLoader.get_participant_id_reverse_mapping()
  if reverseIdMapping is not None and extraFrames:
    apply_participant_id_mapping(extraFrames, reverseIdMapping)

  if args.sample_ratings:
    origSize = len(ratings)
    ratings = ratings.sample(frac=args.sample_ratings)
    logger.info(f"ratings reduced from {origSize} to {len(ratings)}")

  return notes, ratings, statusHistory, userEnrollment, reverseIdMapping, dataLoader


def _restore_outputs(reverseIdMapping, *dfs_and_cols):
  """Restore original public string IDs in output DataFrames when needed."""
  if reverseIdMapping is None:
    return
  outputs = [(df, col) for df, col in dfs_and_cols if col in df.columns]
  if outputs:
    _restore_participant_ids(outputs, reverseIdMapping)


# ---------------------------------------------------------------------------
# Per-phase runners.  Each re-loads data from TSV (fresh process, clean memory)
# and reads/writes intermediates from --outdir/_intermediates/.
# ---------------------------------------------------------------------------


def _run_pss_phase(args, dataLoader=None):
  """Phase 1: Post Selection Similarity + Quasi-Clique Detection.

  Honors --skip-pss by writing an empty PSS DataFrame (same shape as a real
  PSS output) so downstream phases can read it without branching.
  """
  logger.info("=== Phase: PSS ===")
  idir = _intermediates_path(args)
  pss_path = os.path.join(idir, "pss_output.parquet")

  if args.skip_pss:
    logger.info("skipPss=True: writing empty PSS DataFrame.")
    postSelectionSimilarityValues = pd.DataFrame(
      columns=[c.raterParticipantIdKey, c.postSelectionValueKey, c.quasiCliqueValueKey]
    )
  else:
    notes, ratings, _, _, _, _ = _load_data(args, dataLoader)
    postSelectionSimilarityValues = run_rater_clustering(notes=notes, ratings=ratings)

  postSelectionSimilarityValues.to_parquet(pss_path)
  logger.info(f"PSS phase complete. Wrote {pss_path}")


def _run_prescoring_phase(args, dataLoader=None):
  """Phase 2: Prescoring (parallel scorers + pflip + pcrh + empirical prior)."""
  logger.info("=== Phase: Prescoring ===")
  notes, ratings, statusHistory, userEnrollment, reverseIdMapping, dataLoader = _load_data(
    args, dataLoader
  )

  idir = _intermediates_path(args)
  postSelectionSimilarityValues = pd.read_parquet(os.path.join(idir, "pss_output.parquet"))

  previousScoredNotes, previousAuxiliaryNoteInfo = _load_previous_outputs(args)

  (
    prescoringNoteModelOutput,
    prescoringRaterModelOutput,
    prescoringNoteTopicClassifier,
    prescoringPflipClassifier,
    prescoringPcrhClassifier,
    prescoringMetaOutput,
    _prescoringScoredNotes,
    empiricalTotals,
  ) = run_prescoring(
    args,
    notes=notes,
    ratings=ratings,
    noteStatusHistory=statusHistory,
    userEnrollment=userEnrollment,
    postSelectionSimilarityValues=postSelectionSimilarityValues,
    seed=args.seed,
    enabledScorers=args.scorers,
    runParallel=args.parallel,
    dataLoader=dataLoader if args.parallel else None,
    checkFlips=False,
    previousRatingCutoffTimestampMillis=getattr(args, "previous_rating_cutoff_millis", None),
    maxWorkers=args.maxWorkers,
  )

  # Phase boundary: intermediates must hold the original public IDs so the next
  # phase's fresh _load_data can produce matching normalized IDs.
  _restore_outputs(
    reverseIdMapping,
    (prescoringRaterModelOutput, c.raterParticipantIdKey),
  )

  prescoringNoteModelOutput.to_parquet(os.path.join(idir, "prescoring_note_model_output.parquet"))
  prescoringRaterModelOutput.to_parquet(os.path.join(idir, "prescoring_rater_model_output.parquet"))
  joblib.dump(prescoringNoteTopicClassifier, os.path.join(idir, "note_topic_classifier.joblib"))
  joblib.dump(prescoringPflipClassifier, os.path.join(idir, "pflip_classifier.joblib"))
  joblib.dump(prescoringPcrhClassifier, os.path.join(idir, "pcrh_classifier.joblib"))
  joblib.dump(prescoringMetaOutput, os.path.join(idir, "prescoring_meta_output.joblib"))
  empiricalTotals.to_parquet(os.path.join(idir, "empirical_totals.parquet"))

  logger.info("Prescoring phase complete.")


def _run_final_phase(args, dataLoader=None):
  """Phase 3: Final note scoring (parallel scorers)."""
  logger.info("=== Phase: Final Scoring ===")

  # Load prescoring intermediates before input data so they can be mapped with
  # the same participant ID mapping produced while loading TSV inputs.
  idir = _intermediates_path(args)
  prescoringNoteModelOutput = pd.read_parquet(
    os.path.join(idir, "prescoring_note_model_output.parquet")
  )
  prescoringRaterModelOutput = pd.read_parquet(
    os.path.join(idir, "prescoring_rater_model_output.parquet")
  )
  prescoringNoteTopicClassifier = joblib.load(os.path.join(idir, "note_topic_classifier.joblib"))
  prescoringPflipClassifier = joblib.load(os.path.join(idir, "pflip_classifier.joblib"))
  prescoringPcrhClassifier = joblib.load(os.path.join(idir, "pcrh_classifier.joblib"))
  prescoringMetaOutput = joblib.load(os.path.join(idir, "prescoring_meta_output.joblib"))
  empiricalTotals = pd.read_parquet(os.path.join(idir, "empirical_totals.parquet"))

  notes, ratings, statusHistory, userEnrollment, reverseIdMapping, dataLoader = _load_data(
    args,
    dataLoader,
    extraFrames=[(prescoringRaterModelOutput, c.raterParticipantIdKey)],
  )

  previousScoredNotes, previousAuxiliaryNoteInfo = _load_previous_outputs(args)

  scoredNotes, newStatus, auxNoteInfo, _ = run_final_note_scoring(
    args,
    notes=notes,
    ratings=ratings,
    noteStatusHistory=statusHistory,
    userEnrollment=userEnrollment,
    seed=args.seed,
    pseudoraters=args.pseudoraters,
    enabledScorers=args.scorers,
    strictColumns=args.strict_columns,
    runParallel=args.parallel,
    dataLoader=dataLoader if args.parallel else None,
    prescoringNoteModelOutput=prescoringNoteModelOutput,
    prescoringRaterModelOutput=prescoringRaterModelOutput,
    noteTopicClassifier=prescoringNoteTopicClassifier,
    pflipClassifier=prescoringPflipClassifier,
    pcrhClassifier=prescoringPcrhClassifier,
    prescoringMetaOutput=prescoringMetaOutput,
    checkFlips=args.check_flips,
    previousScoredNotes=previousScoredNotes,
    previousAuxiliaryNoteInfo=previousAuxiliaryNoteInfo,
    previousRatingCutoffTimestampMillis=getattr(args, "previous_rating_cutoff_millis", None),
    empiricalTotals=empiricalTotals,
    maxWorkers=args.maxWorkers,
  )

  # Restore public participant IDs before writing final outputs and contributor intermediates.
  _restore_outputs(
    reverseIdMapping,
    (scoredNotes, c.noteAuthorParticipantIdKey),
    (newStatus, c.noteAuthorParticipantIdKey),
    (auxNoteInfo, c.noteAuthorParticipantIdKey),
  )

  write_tsv_local(scoredNotes, os.path.join(args.outdir, "scored_notes.tsv"))
  write_tsv_local(newStatus, os.path.join(args.outdir, "note_status_history.tsv"))
  write_tsv_local(auxNoteInfo, os.path.join(args.outdir, "aux_note_info.tsv"))
  if not args.no_parquet:
    write_parquet_local(scoredNotes, os.path.join(args.outdir, "scored_notes.parquet"))
    write_parquet_local(newStatus, os.path.join(args.outdir, "note_status_history.parquet"))
    write_parquet_local(auxNoteInfo, os.path.join(args.outdir, "aux_note_info.parquet"))

  # Hand intermediates to the contributor phase.
  scoredNotes.to_parquet(os.path.join(idir, "scored_notes.parquet"))
  auxNoteInfo.to_parquet(os.path.join(idir, "aux_note_info.parquet"))
  newStatus.to_parquet(os.path.join(idir, "new_note_status_history.parquet"))

  logger.info("Final scoring phase complete.")


def _run_contributor_phase(args, dataLoader=None):
  """Phase 4: Contributor scoring."""
  logger.info("=== Phase: Contributor Scoring ===")

  # Load all parquet intermediates first so they can be mapped with the same
  # participant ID mapping produced while loading TSV inputs.
  idir = _intermediates_path(args)
  scoredNotes = pd.read_parquet(os.path.join(idir, "scored_notes.parquet"))
  auxNoteInfo = pd.read_parquet(os.path.join(idir, "aux_note_info.parquet"))
  newStatus = pd.read_parquet(os.path.join(idir, "new_note_status_history.parquet"))
  prescoringRaterModelOutput = pd.read_parquet(
    os.path.join(idir, "prescoring_rater_model_output.parquet")
  )

  _, ratings, _, userEnrollment, reverseIdMapping, _ = _load_data(
    args,
    dataLoader,
    extraFrames=[
      (prescoringRaterModelOutput, c.raterParticipantIdKey),
      (scoredNotes, c.noteAuthorParticipantIdKey),
      (auxNoteInfo, c.noteAuthorParticipantIdKey),
      (newStatus, c.noteAuthorParticipantIdKey),
    ],
  )

  # Contributor scoring runs on normalized IDs; restore public IDs on final output.
  helpfulnessScores = run_contributor_scoring(
    ratings=ratings,
    scoredNotes=scoredNotes,
    auxiliaryNoteInfo=auxNoteInfo,
    prescoringRaterModelOutput=prescoringRaterModelOutput,
    noteStatusHistory=newStatus,
    userEnrollment=userEnrollment,
    strictColumns=args.strict_columns,
  )

  _restore_outputs(
    reverseIdMapping,
    (helpfulnessScores, c.raterParticipantIdKey),
  )

  write_tsv_local(helpfulnessScores, os.path.join(args.outdir, "helpfulness_scores.tsv"))
  if not args.no_parquet:
    write_parquet_local(helpfulnessScores, os.path.join(args.outdir, "helpfulness_scores.parquet"))

  logger.info("Contributor scoring phase complete.")


# ---------------------------------------------------------------------------
# Single-process runner (--phase all, backward compatible default).
# ---------------------------------------------------------------------------


def _run_scorer(
  args=None,
  dataLoader=None,
  extraScoringArgs={},
):
  """Run all phases in a single process. Equivalent to the pre-phase-split runner.

  Normalizes IDs at load time so the same memory savings as the phase-split apply
  inside the parallel scorers. Tradeoff vs the phase-split: TSV-parse and PSS
  fragmentation are not reclaimed between phases. Recommended for small datasets
  and tests; use --phase pss|prescoring|final|contributor for full-scale runs.
  """
  logger.info("beginning scorer execution")
  assert args is not None, "args must be available"
  notes, ratings, statusHistory, userEnrollment, reverseIdMapping, dataLoader = _load_data(
    args, dataLoader
  )

  previousScoredNotes, previousAuxiliaryNoteInfo = _load_previous_outputs(args)

  scoredNotes, helpfulnessScores, newStatus, auxNoteInfo = run_scoring(
    args,
    notes,
    ratings,
    statusHistory,
    userEnrollment,
    seed=args.seed,
    pseudoraters=args.pseudoraters,
    enabledScorers=args.scorers,
    strictColumns=args.strict_columns,
    runParallel=args.parallel,
    dataLoader=dataLoader if args.parallel == True else None,
    cutoffTimestampMillis=args.cutoffTimestampMillis,
    excludeRatingsAfterANoteGotFirstStatusPlusNHours=args.excludeRatingsAfterANoteGotFirstStatusPlusNHours,
    daysInPastToApplyPostFirstStatusFiltering=args.daysInPastToApplyPostFirstStatusFiltering,
    filterPrescoringInputToSimulateDelayInHours=args.prescoring_delay_hours,
    checkFlips=args.check_flips,
    previousScoredNotes=previousScoredNotes,
    previousAuxiliaryNoteInfo=previousAuxiliaryNoteInfo,
    previousRatingCutoffTimestampMillis=args.previous_rating_cutoff_millis,
    maxWorkers=args.maxWorkers,
    skipPss=args.skip_pss,
    **extraScoringArgs,
  )

  # Restore original public IDs on all single-process outputs before writing.
  _restore_outputs(
    reverseIdMapping,
    (scoredNotes, c.noteAuthorParticipantIdKey),
    (helpfulnessScores, c.raterParticipantIdKey),
    (newStatus, c.noteAuthorParticipantIdKey),
    (auxNoteInfo, c.noteAuthorParticipantIdKey),
  )

  write_tsv_local(scoredNotes, os.path.join(args.outdir, "scored_notes.tsv"))
  write_tsv_local(helpfulnessScores, os.path.join(args.outdir, "helpfulness_scores.tsv"))
  write_tsv_local(newStatus, os.path.join(args.outdir, "note_status_history.tsv"))
  write_tsv_local(auxNoteInfo, os.path.join(args.outdir, "aux_note_info.tsv"))

  if not args.no_parquet:
    write_parquet_local(scoredNotes, os.path.join(args.outdir, "scored_notes.parquet"))
    write_parquet_local(helpfulnessScores, os.path.join(args.outdir, "helpfulness_scores.parquet"))
    write_parquet_local(newStatus, os.path.join(args.outdir, "note_status_history.parquet"))
    write_parquet_local(auxNoteInfo, os.path.join(args.outdir, "aux_note_info.parquet"))


_PHASE_RUNNERS = {
  "pss": _run_pss_phase,
  "prescoring": _run_prescoring_phase,
  "final": _run_final_phase,
  "contributor": _run_contributor_phase,
}


def main(
  args=None,
  dataLoader=None,
  extraScoringArgs={},
):
  if args is None:
    args = parse_args()
  logger.info(f"scorer python version: {sys.version}")
  logger.info(f"scorer pandas version: {pd.__version__}")

  phase = getattr(args, "phase", "all")
  if phase == "all":
    return _run_scorer(args=args, dataLoader=dataLoader, extraScoringArgs=extraScoringArgs)
  return _PHASE_RUNNERS[phase](args, dataLoader)


if __name__ == "__main__":
  main()
