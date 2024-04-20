"""Assign notes to a set of predetermined topics.

The topic assignment process is seeded with a small set of terms which are indicative of
the topic.  After preliminary topic assignment based on term matching, a logistic regression
trained on bag-of-words features model expands the set of in-topic notes for each topic.
Note that the logistic regression modeling excludes any tokens containing seed terms.

This approach represents a prelimiary approach to topic assignment while Community Notes
evaluates the efficacy of per-topic note scoring.
"""

from typing import List, Tuple

from . import constants as c
from .enums import Topics

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline


class TopicModel(object):
  def __init__(self):
    """Initialize a list of seed terms for each topic."""
    self._seedTerms = {
      Topics.UkraineConflict: {
        "ukrain",  # intentionally shortened for expanded matching
        "russia",
        "kiev",
        "kyiv",
        "moscow",
        "zelensky",
        "putin",
      },
      Topics.GazaConflict: {
        "israel",
        "palestin",  # intentionally shortened for expanded matching
        "gaza",
        "jerusalem",
      },
      Topics.MessiRonaldo: {
        "messi ",  # intentional whitespace to prevent prefix matches
        "ronaldo",
      },
    }

  def _make_seed_labels(self, texts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Produce a label vector based on seed terms.

    The label vector has type np.int64 with values corresponding to the enum value for
    each topic.  Any text which matches seed terms from multiple topics is left unassigned.

    Args:
      texts: array containing strings for topic assignment

    Returns:
      Tuple[0]: array specifing topic labels for texts
      Tuple[1]: array specifying texts that are unassigned due to conflicting matches.
    """
    texts = np.array([text.lower() for text in texts])
    labels = np.zeros(texts.shape[0], dtype=np.int64)
    conflictedTexts = np.zeros(texts.shape[0])
    for topic in Topics:
      if topic == Topics.Unassigned:
        continue
      topicMatches = np.array(
        [any(term in text for term in self._seedTerms[topic]) for text in texts]
      )
      labels[topicMatches] = topic.value
      conflictedTexts += topicMatches.astype(np.int64)
    labels[conflictedTexts > 1] = Topics.Unassigned.value
    print(f"  Notes unassigned due to multiple matches: {conflictedTexts.sum()}")
    return labels, conflictedTexts > 1

  def _get_stop_words(self, texts: np.ndarray) -> List[str]:
    """Identify tokens in the extracted vocabulary that contain seed terms.

    Any token containing a seed term will be treated as a stop word (i.e. removed
    from the extracted features).  This prevents the model from training on the same
    tokens used to label the data.

    Args:
      texts: array containing strings for topic assignment

    Returns:
      List specifying which tokens to exclude from the features.
    """
    # Extract vocabulary
    cv = CountVectorizer(strip_accents="unicode")
    cv.fit(texts)
    rawVocabulary = cv.vocabulary_.keys()
    print(f"  Initial vocabulary length: {len(rawVocabulary)}")
    # Identify stop words
    blockedTokens = set()
    for terms in self._seedTerms.values():
      blockedTokens |= {t.strip() for t in terms}
    print(f"  Total tokens to filter: {len(blockedTokens)}")
    stopWords = [v for v in rawVocabulary if any(t in v for t in blockedTokens)]
    print(f"  Total identified stopwords: {len(stopWords)}")
    return stopWords

  def _merge_predictions_and_labels(
    self, predictions: np.ndarray, labels: np.ndarray
  ) -> np.ndarray:
    """Update predictions based on defined labels when the label is not Unassigned.

    Args:
      predictions: 1D matrix specifying the class label as an int64 assigned to each row.

    Returns:
      Updated predictions based on keyword matches when available.
    """
    for label in range(1, len(Topics)):
      predictions[labels == label] = label
    return predictions

  def _prepare_post_text(self, notes: pd.DataFrame) -> pd.DataFrame:
    """Concatenate all notes within each post into a single row associated with the post.

    Args:
      notes: dataframe containing raw note text

    Returns:
      DataFrame with one post per row containing note text
    """
    postNoteText = (
      notes[[c.tweetIdKey, c.summaryKey]]
      .fillna({c.summaryKey: ""})
      .groupby(c.tweetIdKey)[c.summaryKey]
      .apply(lambda postNotes: " ".join(postNotes))
      .reset_index(drop=False)
    )
    # Default tokenization for CountVectorizer will not split on underscore, which
    # results in very long tokens containing many words inside of URLs.  Removing
    # underscores allows us to keep default splitting while fixing that problem.
    postNoteText[c.summaryKey] = [
      text.replace("_", " ") for text in postNoteText[c.summaryKey].values
    ]
    return postNoteText

  def get_note_topics(self, notes: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame specifying each {note, topic} pair.

    Notes that are not assigned to a topic do not appear in the dataframe.

    Args:
      notes: DF containing all notes to potentially assign to a topic
    """
    print("Assigning notes to topics:")
    # Obtain aggregate post text, seed labels and stop words
    postText = self._prepare_post_text(notes)
    seedLabels, conflictedTexts = self._make_seed_labels(postText[c.summaryKey].values)
    stopWords = self._get_stop_words(postText[c.summaryKey].values)
    # Define and fit model
    pipe = Pipeline(
      [
        (
          "UnigramEncoder",
          CountVectorizer(
            strip_accents="unicode",
            stop_words=stopWords,
            min_df=25,
            max_df=max(1000, int(0.25 * len(postText))),
          ),
        ),
        ("tfidf", TfidfTransformer()),
        ("Classifier", LogisticRegression(max_iter=1000, verbose=1)),
      ],
      verbose=True,
    )
    pipe.fit(
      # Notice that we omit posts with an unclear label from training.
      postText[c.summaryKey].values[~conflictedTexts],
      seedLabels[~conflictedTexts],
    )
    # Predict notes.  Notice that in effect we are looking to see which notes in the
    # training data the model felt were mis-labeled after the training process
    # completed, and generating labels for any posts which were omitted from the
    # original training.
    pred = pipe.predict(postText[c.summaryKey].values)
    balancedAccuracy = balanced_accuracy_score(seedLabels[~conflictedTexts], pred[~conflictedTexts])
    print(f"  Balanced accuracy on raw predictions: {balancedAccuracy}")
    assert balancedAccuracy > 0.5, f"Balanced accuracy too low: {balancedAccuracy}"
    # Validate that any conflicted text is Unassigned
    assert all(seedLabels[conflictedTexts] == Topics.Unassigned.value)
    pred = self._merge_predictions_and_labels(pred, seedLabels)
    print(f"  Topic assignment results: {np.bincount(pred)}")

    # Assign topics to notes based on aggregated note text, and drop any
    # notes on posts that were unassigned.
    postText[c.noteTopicKey] = [Topics(t).name for t in pred]
    postText = postText[postText[c.noteTopicKey] != Topics.Unassigned.name]
    noteTopics = notes[[c.noteIdKey, c.tweetIdKey]].merge(postText[[c.tweetIdKey, c.noteTopicKey]])
    return noteTopics.drop(columns=c.tweetIdKey)
