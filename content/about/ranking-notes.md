---
title: Note ranking
aliases:
  [
    "/ranking-notes",
    "/note-ranking",
    "/about/note-ranking",
    "/about/ranking-notes",
  ]
geekdocBreadcrumb: false
---

# Note ranking

When displaying notes written on a Tweet, Birdwatch initially shows notes in reverse-chronological order (most recent on top). In some cases — if notes have received enough ratings from other contributors, and certain criteria are met — notes might be raised to the top of the list and highlighted with a header showing that they are **“currently rated helpful.”**

To help people understand how Birdwatch works, this page and code snippet below explain how notes are identified as “currently rated helpful.”

{{< figure src="../../images/5-good-notes.png">}}

In short, Birdwatch computes a “helpfulness ratio” for each note, which is simply the proportion of ratings that say the note is helpful. Then out of all notes that have received at least 5 ratings and a minimum helpfulness ratio of 0.84, the top 3 by “helpfulness ratio” are annotated as “currently rated helpful”. This ranking mechanism is knowingly basic, and we only intend to use it for a short time during the program’s earliest phase.

During the pilot, “currently rated helpful” notes are only computed at periodic intervals, so there is a time delay from when a note meets the “currently rated helpful” criteria and when it rises to the top of the list. This delay allows Birdwatch to collect a set of independent ratings from people who haven’t yet been influenced by seeing the “currently rated helpful” flag on certain notes.

Here’s a Python code snippet one can run to reproduce how we compute our “Currently Rated Helpful” annotations. It uses as input the notes and ratings files made available on the Birdwatch [Data Download](https://twitter.com/i/birdwatch/download-data) page.

{{< highlight python >}}

import pandas as pd
notes = pd.read_csv('notes-00000.tsv', sep='\t')
ratings = pd.read_csv('ratings-00000.tsv', sep='\t')

ratingsWithNotes = notes.set_index('noteId').join(ratings.set_index('noteId'), lsuffix="_note", rsuffix="_rating", how='inner')
ratingsWithNotes['numRatings'] = 1

def getCurrentlyRatedHelpfulNotesForTweet(
  tweetId,
  maxCurrentlyRatedHelpfulNotes = 3,
  minRatingsNeeded = 5,
  minHelpfulnessRatioNeeded = 0.84
):
  ratingsWithNotesForTweet = ratingsWithNotes[ratingsWithNotes['tweetId']==tweetId]
  scoredNotes = ratingsWithNotesForTweet.groupby('noteId').sum()
  scoredNotes['helpfulnessRatio'] = scoredNotes['helpful']/scoredNotes['numRatings']
  filteredNotes = scoredNotes[(scoredNotes['numRatings'] >= minRatingsNeeded) & (scoredNotes['helpfulnessRatio'] >= minHelpfulnessRatioNeeded)]
  filteredNotes = filteredNotes.join(notes.set_index('noteId').drop('tweetId',axis=1), lsuffix="_note", rsuffix="_rating", how='inner')  # join in note info
  return filteredNotes.sort_values(by='helpfulnessRatio', ascending=False)[:maxCurrentlyRatedHelpfulNotes]

{{< / highlight >}}

These notes that are labeled “currently rated helpful” are used to determine what Tweets are added to the “Rated helpful” tab on the [Birdwatch Home page](https://twitter.com/i/birdwatch/). For a Tweet to appear in that tab, it must have at least one note “currently rated helpful”, at least one of those “currently rated helpful” notes must have labeled the Tweet “misinformed or potentially misleading”, and a majority of those “currently rated helpful” notes had to have not labeled the Tweet as satire (either “It is a joke or satire that might be misinterpreted as a fact” or “It is clearly satirical/joking”). The Tweets that pass these filters are sorted chronologically by the timestamp of the Tweet’s first-created “currently rated helpful” note.

{{< figure src="../../images/home-two-tabs.png">}}
