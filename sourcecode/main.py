"""Invoke Community Notes scoring and user contribution algorithms.

Example Usage:
  python main.py \
    --enrollment data/user_enrollment_tsv \
    --notes data/all_birdwatch_notes.tsv \
    --output data/scored_notes.tsv \
    --ratings data/all_birdwatch_note_ratings.tsv \
    --status data/note_status_history_public.tsv
"""

import argparse
import os

import scoring.constants as c
from scoring.process_data import get_data, write_tsv_local
from scoring.run_scoring import run_scoring


def parse_args():
  parser = argparse.ArgumentParser("Community Notes Scoring")
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
  parser.add_argument("-n", "--notes", default=c.notesInputPath, help="note dataset")
  parser.add_argument(
    "-o", "--outdir", default=c.scoredNotesOutputPath, help="directory for output files"
  )
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
    "--seed", default=None, type=int, help="set to an int to seed matrix factorization"
  )
  parser.add_argument(
    "-s", "--status", default=c.noteStatusHistoryInputPath, help="note status history dataset"
  )

  return parser.parse_args()


def main():
  args = parse_args()
  if args.epoch_millis:
    c.epochMillis = args.epoch_millis

  _, ratings, statusHistory, userEnrollment = get_data(
    args.notes, args.ratings, args.status, args.enrollment
  )

  scoredNotes, helpfulnessScores, newStatus, auxNoteInfo = run_scoring(
    ratings, statusHistory, userEnrollment, seed=args.seed, pseudoraters=args.pseudoraters
  )

  write_tsv_local(scoredNotes, os.path.join(args.outdir, "scored_notes.tsv"))
  write_tsv_local(helpfulnessScores, os.path.join(args.outdir, "helpfulness_scores.tsv"))
  write_tsv_local(newStatus, os.path.join(args.outdir, "note_status_history.tsv"))
  write_tsv_local(auxNoteInfo, os.path.join(args.outdir, "aux_note_info.tsv"))


if __name__ == "__main__":
  main()
