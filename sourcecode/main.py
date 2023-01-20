import argparse

import algorithm, constants as c, process_data


"""
Example Usage:
  python main.py \
    --enrollment userEnrollment-00000.tsv \
    --notes_path notes-00000.tsv  \
    --ratings_path ratings-00000.tsv \
    --note_status_history_path noteStatusHistory-00000.tsv
"""


def get_args():
  """Parse command line arguments for running on command line.

  Returns:
      args: the parsed arguments
  """
  parser = argparse.ArgumentParser(description="Birdwatch Algorithm.")
  parser.add_argument(
    "-e", "--enrollment", default=c.enrollmentInputPath, help="user enrollment dataset"
  )
  parser.add_argument("-n", "--notes_path", default=c.notesInputPath, help="note dataset")
  parser.add_argument("-r", "--ratings_path", default=c.ratingsInputPath, help="rating dataset")
  parser.add_argument(
    "-s",
    "--note_status_history_path",
    default=c.noteStatusHistoryInputPath,
    help="note status history dataset",
  )
  parser.add_argument("-o", "--output_path", default=c.scoredNotesOutputPath, help="output path")
  return parser.parse_args()


def run_scoring():
  """
  Run the complete Birdwatch algorithm, including parsing command line args,
  reading data and writing scored output; mean to be invoked from main.
  """
  args = get_args()
  _, ratings, noteStatusHistory, userEnrollment = process_data.get_data(
    args.notes_path, args.ratings_path, args.note_status_history_path, args.enrollment
  )
  scoredNotes, _, _, _ = algorithm.run_algorithm(ratings, noteStatusHistory, userEnrollment)
  process_data.write_tsv_local(scoredNotes, c.scoredNotesOutputPath)
  print("Finished.")


if __name__ == "__main__":
  run_scoring()
