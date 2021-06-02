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

When displaying notes written on a Tweet, Birdwatch initially shows most recent notes on top and displays them with a status indicator that the note **Needs More Ratings**. In some cases — if notes have received enough ratings from other contributors, and certain criteria are met — notes might be moved to the top or bottom of the list and annotated with a header showing that they are **Currently Rated Helpful** or **Currently Not Rated Helpful**.

To help people understand how Birdwatch works, this page and code snippet below explain how notes are assigned these statuses and ranked.

{{< figure src="../../images/note-statuses.png">}}

1. All Birdwatch notes start out with the **Needs More Ratings** status until they receive at least 5 ratings.

2. Then, Birdwatch computes a **Helpfulness Ratio** for each note, which is simply the proportion of ratings that say the note is helpful.

3. For a note to be labeled as **Currently Rated Helpful** or **Currently Not Rated Helpful**, raters must have selected at least two of the corresponding
   reasons why a note is (or isn’t) helpful — e.g. “cites high-quality sources,” “nonjudgmental and/or empathetic,” etc — at least twice each.
4. Notes with a **Helpfulness Ratio** of 0.29 and below are assigned **Currently Not Rated Helpful**, and notes with a ratio of 0.84 and above are assigned **Currently Rated Helpful**. Notes with scores in between 0.29 and 0.84 remain labeled as **Needs more Ratings**.

5. In addition to **Currently Rated / Not Rated Helpful** status, labels also show the most commonly chosen reasons why it is helpful/unhelpful. To break ties between reasons that are chosen an equal number of times by raters, we pick the reason that is used least frequently by Birdwatch raters in general (with the exception of Other, which loses all tiebreaks).

6. Notes with the status **Needs More Ratings** remain sorted by recency (newest first), and notes with a **Currently Rated / Not Rated Helpful** status are sorted by their **Helpfulness Ratio**.

This ranking mechanism is knowingly basic, and we only intend to use it for a short time during the program’s earliest phases.

During the pilot, rating statuses are only computed at periodic intervals, so there is a time delay from when a note meets the **Currently Rated / Not Rated Helpful** criteria and when it jumps to the top or bottom of the list. This delay allows Birdwatch to collect a set of independent ratings from people who haven’t yet been influenced by seeing status annotations on certain notes.

### Tie breaking

For tie-breaking, we use the following priority order of reasons:

{{< columns >}} <!-- begin columns block -->

**Currently Rated Helpful Reasons**

- UniqueContext
- Empathetic
- GoodSources
- Clear
- Informative
- Other

<---> <!-- magic sparator, between columns -->

**Currently Not Rated Helpful Reasons**

- Outdated
- SpamHarassmentOrAbuse
- HardToUnderstand
- OffTopic
- Incorrect
- ArgumentativeOrInflammatory
- MissingKeyPoints
- SourcesMissingOrUnreliable
- OpinionSpeculationOrBias
- Other

{{< /columns >}}

### Ranking Code

Here’s a Python code snippet one can run to reproduce how we compute notes’ rating statuses and sorting. It uses as input the notes and ratings files made available on the Birdwatch [Data Download](https://twitter.com/i/birdwatch/download-data) page.

{{< include file="static/sourcecode/ranking-notes.py" language="python" markdown=false >}}

{{< button href="/birdwatch/sourcecode/ranking-notes.py" >}}See source file{{< /button >}}

## Tabs: Needs Your Help, New, and Rated Helpful

Birdwatch participants are able to view three different tabs on the Birdwatch [Home](https://twitter.com/i/birdwatch) page. Each tab contains different sets of Tweets in different orders. In order to appear in any of these tabs, a Tweet must have received at least 100 total likes plus Retweets.

{{< figure src="../../images/home-three-tabs.png">}}

### New

Contains Tweets sorted by reverse chronological order of when its latest note was written (Tweets are bumped to the top of the list when a new note is written on them).

### Rated Helpful

Tweets in this tab have at least one note labeled **Currently Rated Helpful**, at least one of those **Currently Rated Helpful** notes must have labeled the Tweet “misinformed or potentially misleading”, and a majority of those **Currently Rated Helpful** notes had to have not labeled the Tweet as satire (either “It is a joke or satire that might be misinterpreted as a fact” or “It is clearly satirical/joking”). The Tweets that pass these filters are sorted reverse-chronologically by the timestamp of the Tweet’s first-created **Currently Rated Helpful** note.

### Needs Your Help

This tab is only visible to pilot participants. It is designed to increase the likelihood that people from diverse perspectives rate each note, so that Birdwatch can ultimately elevate notes that people from a wide range of perspectives will find helpful. It also gives Birdwatch contributors an easy way to have immediate impact.

It contains a set of 5 Tweets that have notes that need more ratings (although there may be fewer than 5 Tweets if one of the Tweets was recently deleted by the author). Tweets in this tab are filtered to those that the contributor hasn’t rated any of the notes on, and Tweets with notes from the past day, unless no Tweets pass those filters for you (that will only happen if you’re a very active rater!). The tab is updated roughly every couple hours, so when the contributor has rated notes in the tab, they can come back later to see fresh Tweets.

### Rater similarity score for Needs Your Help

Tweets on the **Needs Your Help** tab are sorted by a ranking score, where Tweets are ranked higher based on the proportion of notes on the Tweet that are labeled **Needs More Ratings**, and Tweets are ranked lower if your rating history is more similar to the other raters who’ve already rated notes on this Tweet: this is a mechanism to help increase the likelihood that people from diverse perspectives rate each note.

Specifically, the rater similarity score between a pair of raters is defined as the number of notes that both raters rated, divided by the minimum number of total notes each rater has rated(over each of the two raters).

Example: if rater A rated notes 1, 2, and 3, while rater B rated notes 1, 3, 4, 5, and 6, then the similarity score between A and B = 2 / min(3, 5) = 2/3, because they both rated notes 1 and 3, for a total of 2, and the minimum number of total notes either rater has rated is 3. If two raters haven’t rated any of the same notes, the score is defined as 0.01 instead of 0. Then, the average rater similarity between a contributor and each other rater who’s rated a particular Tweet is computed, in order to compute the average rater similarity between the contributor and the raters of that Tweet. Then, Tweets are ranked in the **Needs your help** tab using the following score: 0.3 \* proportionOfNotesOnTweetThatNeedMoreRatings - averageRaterSimilarityScore.

The factor of 0.3 is applied to help to balance between the two scores fairly equally. We may in the future experiment with using other forms of rater similarity, e.g. based on whether two contributors tend to agree when they rate the same notes. This version primarily reflects contributors’ interest in what they choose to rate.

## Birdwatch cards on Tweets

When browsing Twitter on Android, iOS, or at Twitter.com, Birdwatch pilot participants may see cards on Tweets that have Birdwatch notes. Birdwatch cards may include a specific Birdwatch note or reference the number of Birdwatch notes on the Tweet, and link to the Birdwatch site where contributors can rate the notes.

### Cards showing a specific note

{{< figure src="../../images/notes-on-twitter-specific.png">}}

If a Tweet has a Birdwatch note deemed **Currently rated helpful** by contributors (determined as described above in **Note Ranking**) then the note will be shown in the card, along with a button to rate the note. If the Tweet has multiple notes with **Currently rated helpful** status, the card will show one note, randomly cycling between them at periodic intervals in order to gather additional rating inputs on each.

### Cards showing a count of notes

{{< figure src="../../images/notes-on-twitter-count.png">}}

If a Tweet has notes but none yet have **Currently rated helpful** status, the card will show the number of notes, and allow people to tap to read and rate those notes on the Birdwatch site.

If all notes on a Tweet have **Currently not rated helpful** status, no card will be shown. Pilot participants will still be able to access these notes via the Birdwatch icon displayed in the Tweet details page.
