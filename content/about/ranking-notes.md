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

When displaying notes written on a Tweet, Birdwatch initially shows notes in reverse-chronological order (most recent on top), and displays them with a status indicator that the note “Needs More Ratings”. In some cases — if notes have received enough ratings from other contributors, and certain criteria are met — notes might be moved to the top or bottom of the list and annotated with a header showing that they are “Currently Rated Helpful” or “Currently Not Rated Helpful.”

To help people understand how Birdwatch works, this page and code snippet below explain how notes are assigned these statuses and ranked.

{{< figure src="../../images/note-statuses.png">}}

All Birdwatch notes start out with the “Needs More Ratings” status until they receive at least 5 ratings. Then, Birdwatch computes a “helpfulness ratio” for each note, which is simply the proportion of ratings that say the note is helpful. To be assigned a status of “Currently Rated Helpful” or “Currently Not Rated Helpful”, two other criteria must be met. First, there is a minimum helpfulness rating of 0.84 to be “Currently Rated Helpful” and a maximum helpfulness rating of 0.29 to be “Currently Not Rated Helpful”. Second, raters must have selected at least two of the corresponding reasons why a note is (or isn’t) helpful — e.g. “cites high-quality sources,” “nonjudgmental and/or empathetic,” etc — at least twice each. Once a note receives at least 5 ratings and meets the helpfulness ratio threshold as well as receives two reasons why, the note is assigned the corresponding label and assigned the most commonly chosen reasons why. To break ties between reasons that are chosen an equal number of times by raters, we pick the reason that is used least frequently by Birdwatch raters in general (with the exception of Other, which loses all tiebreaks).

While notes with the status “Needs More Ratings” are sorted reverse-chronologically (newest first), notes with a “Currently Rated Helpful” or “Currently Not Rated Helpful” status are sorted by their “helpfulness ratio”. This ranking mechanism is knowingly basic, and we only intend to use it for a short time during the program’s earliest phases.

During the pilot, rating statuses are only computed at periodic intervals, so there is a time delay from when a note meets the “Currently Rated Helpful” or “Currently Not Rated Helpful” criteria and when it jumps to the top or bottom of the list. This delay allows Birdwatch to collect a set of independent ratings from people who haven’t yet been influenced by seeing status annotations on certain notes.

Priority Order of “Currently Rated Helpful” Reasons (For tie-breaking only):

- UniqueContext
- Empathetic
- GoodSources
- Clear
- Informative
- Other  
  <br/>

Priority Order of “Currently Not Rated Helpful” Reasons (For tie-breaking only):

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
  <br/>

Here’s a Python code snippet one can run to reproduce how we compute notes’ rating statuses and sorting. It uses as input the notes and ratings files made available on the Birdwatch [Data Download](https://twitter.com/i/birdwatch/download-data) page.

{{< include file="static/sourcecode/ranking-notes.py" language="python" markdown=false >}}

{{< button href="/birdwatch/sourcecode/ranking-notes.py" >}}See source file{{< /button >}}

<br/>
<br/>

## Tabs: Needs Your Help, New, and Rated Helpful

Birdwatch participants are able to view three different tabs on the Birdwatch [Home](https://twitter.com/i/birdwatch) page. Each tab contains different sets of Tweets in different orders. In order to appear in any of these tabs, a Tweet must have received at least 100 total likes plus Retweets.

{{< figure src="../../images/home-three-tabs.png">}}

The “New” tab contains Tweets sorted by reverse chronological order of when its latest note was written (Tweets are bumped to the top of the list when a new note is written on them).

The notes that are labeled “Currently Rated Helpful” are used to determine what Tweets are added to the “Rated helpful” tab. For a Tweet to appear in that tab, it must have at least one note labeled “Currently Rated Helpful”, at least one of those “Currently Rated Helpful” notes must have labeled the Tweet “misinformed or potentially misleading”, and a majority of those “Currently Rated Helpful” notes had to have not labeled the Tweet as satire (either “It is a joke or satire that might be misinterpreted as a fact” or “It is clearly satirical/joking”). The Tweets that pass these filters are sorted reverse-chronologically by the timestamp of the Tweet’s first-created “Currently Rated Helpful” note.

The “Needs Your Help” tab is designed to increase the likelihood that people from diverse perspectives rate each note, so that Birdwatch can ultimately elevate notes that people from a wide range of perspectives will find helpful. It also gives Birdwatch contributors an easy way to have immediate impact. The tab contains a set of 5 Tweets that have notes that need more ratings (although there may be fewer than 5 Tweets if one of the Tweets was recently deleted by the author). Tweets in this tab are filtered to those that the contributor hasn’t rated any of the notes on, and Tweets with notes from the past day, unless no Tweets pass those filters for you (that will only happen if you’re a very active rater!). Tweets are sorted by a ranking score, where Tweets are ranked higher based on the proportion of notes on the Tweet that are labeled “Needs More Ratings”, and Tweets are ranked lower if your rating history is more similar to the other raters who’ve already rated notes on this Tweet: this is a mechanism to help increase the likelihood that people from diverse perspectives rate each note. The tab is updated roughly every couple hours, so when the contributor has rated notes in the tab, they can come back later to see fresh Tweets.

Specifically, the rater similarity score between a pair of raters is defined as the number of notes that both raters rated, divided by the minimum number of total notes each rater has rated(over each of the two raters). E.g. if rater A rated notes 1, 2, and 3, while rater B rated notes 1, 3, 4, 5, and 6, then the similarity score between A and B = 2 / min(3, 5) = 2/3, because they both rated notes 1 and 3, for a total of 2, and the minimum number of total notes either rater has rated is 3. If two raters haven’t rated any of the same notes, the score is defined as 0.01 instead of 0. Then, the average rater similarity between a contributor and each other rater who’s rated a particular Tweet is computed, in order to compute the average rater similarity between the contributor and the raters of that Tweet. Then, Tweets are ranked in the “Needs your help” tab using the following score: 0.3 \* proportionOfNotesOnTweetThatNeedMoreRatings - averageRaterSimilarityScore. The factor of 0.3 is applied to help to balance between the two scores fairly equally. We may in the future experiment with using other forms of rater similarity, e.g. based on whether two contributors tend to agree when they rate the same notes. This version primarily reflects contributors’ interest in what they choose to rate.
