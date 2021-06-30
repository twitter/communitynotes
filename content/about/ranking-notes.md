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
geekdocToc: 1
katex: true
---

Birdwatch notes are submitted and rated by contributors. Ratings are used to determine [Note Status](./#note-status) (e.g. “Currently Rated Helpful”), which notes are displayed on each of the [Birdwatch Site's timelines](./#birdwatch-timelines), and which notes are displayed as [Birdwatch Cards](./#birdwatch-cards-on-tweets) on the Twitter timelines.

To help people understand how Birdwatch's ranking system works, the sections below explain how notes are assigned these statuses, ranked, and displayed.

{{< toc >}}

## Helpful Ranking Mapping

{{< figure src="../../images/helpful-ranking.png">}}

When rating notes, Birdwatch contributors answer the question “Is this note helpful?” Answers to that question are then used to rank notes. When Birdwatch launched in January 2021, people could answer `yes` or `no` to that question. An update on June 30, 2021 allows people to choose between `yes`, `somewhat` and `no`. We map these responses to continuous values from 0.0 to 1.0, hereafter referred to as `helpful scores`:

- `Yes` maps to `1.0`
- `Somewhat` maps to `0.5`.
- `No` maps to `0.0`.
- `Yes` from the original 2-option version of the rating form maps to `1.0`.
- `No` from the original 2-option version of the rating form maps to `0.0`.

Specific values in the mapping may change, particularly once the `somewhat` option has been available for a period of time, and data exists about how it’s used relative to `yes` and `no` responses.

## Note Status

{{< figure src="../../images/note-statuses.png">}}

All Birdwatch notes start out with the Needs More Ratings status until they receive at least 5 total ratings and at least 2 weighted ratings (the sum of the [Combined Helpfulness Scores](./#combined-helpfulness-score) of the raters that rated the note, where [Combined Helpfulness Scores](./#combined-helpfulness-score) are between `0` and `1`).

Then, Birdwatch computes a Note Helpfulness Score for each note, which is the average helpful score from ratings on that note, where each rating is weighted by the [Combined Helpfulness Scores](./#combined-helpfulness-score) of the rater.

Notes with a Note Helpfulness Score of `0.29` and below are assigned Currently Not Rated Helpful, and notes with a score of `0.84` and above are assigned Currently Rated Helpful. Notes with scores in between `0.29` and `0.84` remain labeled as Needs more Ratings.

In addition to Currently Rated / Not Rated Helpful status, labels also show the most commonly chosen reasons why it is helpful/unhelpful. To break ties between reasons that are chosen an equal number of times by raters, we pick the reason that is used least frequently by Birdwatch raters in general (with the exception of `other`, which loses all tiebreaks).

Notes with the status Needs More Ratings remain sorted by recency (newest first), and notes with a Currently Rated / Not Rated Helpful status are sorted by their Helpfulness Score.

This ranking mechanism is knowingly imperfect, and we are actively developing it with the aim that Birdwatch consistently identifies notes that are found helpful to people from a wide variety of perspectives.

During the pilot, rating statuses are only computed at periodic intervals, so there is a time delay from when a note meets the Currently Rated / Not Rated Helpful criteria and when that designation appears on the Birdwatch site. This delay allows Birdwatch to collect a set of independent ratings from people who haven’t yet been influenced by seeing status annotations on certain notes.

## Combined Helpfulness Score

At a high level, Combined Helpfulness Scores are a way to give more weight to people whose contributions are consistently found helpful by others:

- If you write notes that people consistently find helpful, the ratings you give other notes will have greater weight.
- If the ratings you give often align with the ultimate rating outcome from the community, your ratings will have greater weight.

Each Birdwatch contributor earns an [Author Helpfulness Score](./#author-helpfulness-score) score, and a [Rater Helpfulness Score](./#rater-helpfulness-score). These two scores, in turn, are averaged together into a Combined Helpfulness Score, which is used to weight the ratings that each contributor gives to notes. The purpose of using Combined Helpfulness Score is to give more influence to contributors who have a track record of helpful contributions.

These scores are a simple first approach to improve the quality of Birdwatch’s note scoring and ranking, and a step towards making manipulation of Birdwatch more difficult. More improvements are on the way, including factoring in not just how many ratings a note has received, but also how diverse the perspectives of those raters are — i.e. do those raters always agree with each other, or sometimes/often disagree?

### Author Helpfulness Score

Any contributor who has written at least one note that’s been rated by other participants receives an Author Helpfulness Score. The score starts out low, and can increase (or decrease) as the author receives ratings from more raters.

The way scores are computed is inspired by [PageRank](https://en.wikipedia.org/wiki/PageRank), where ratings from raters who have higher helpfulness scores are given larger weight, rather than weighting incoming ratings equally from all raters. The underlying assumption is that contributors who have written helpful notes themselves will make more accurate ratings about whether other notes will be found helpful.

The score is based on the weighted average of ratings they’ve received from other contributors on the notes that they’ve authored. Each rater’s ratings of any particular author are only counted once in this weighted average, using the average helpfulness rating from that particular rater of the particular author. For example: if a rater rated 10 notes from the same author, those will count as 1 author-level rating instead of 10 ratings.

Then, the score is smoothed by adding 2 pseudocounts to the numerator and 6 pseudocounts to the denominator — this has the effect of scores starting out low and increasing over time as authors receive ratings from more raters. It also effectively means that the default score is ⅓. So we then transform the score with the function `max(0, 1.5s - .5)` in order to set ratios of ⅓ or less to 0 and then renormalize the range back to [0, 1].

We initially assign the Author Helpfulness Score of all contributors to be 1, then repeat the following reweighting update step defined below until the scores converge. Intuitively, the process starts off assuming all raters have equal weight, and calculates each note author’s Helpfulness Score solely based on the average of the ratings they’ve received from each rater. That gives a starting point of which authors’ contributions are found most helpful by others. For the purpose of computing these scores, the process assumes those more helpful authors are also better raters, and re-computes all Author Helpfulness Scores, giving the ratings of helpful authors more weight. Repeat until this process hits a maximum number of iterations, at which point the author helpfulness scores should have converged (stopped changing much between iterations):

\\[a_i(u) = max(0, \frac{3}{2} * \frac{2 + \sum_{rater \in R(u)} a_{i-1}(rater) * rating(rater,u)}{6 + \sum_{rater \in R(u)} a_{i-1} (rater)} - \frac{1}{2})
\\]

In the update equation:

- `a_i(u)` is the Author Helpfulness Score of contributor u at the i-th iteration.
- `i` indicates the iteration number.
- `R(u)` is the set of contributors that rated author u.
- `Rating(v, u)` is contributor `v`’s average rating of contributor `u`, averaged over all notes that were authored by contributor `u` and rated by contributor `v`, with helpful ratings = 1 and not-helpful ratings = 0.

### Preliminary Note Scoring

The next step is to compute preliminary Note Helpfulness Scores, which are only used to compute [Rater Helpfulness Score](./#rater-helpfulness-score) as defined below (they are not used as final Note Helpfulness Scores). The preliminary Note Helpfulness Scores are computed by weighting each rater’s [Author Helpfulness Scores](./#author-helpfulness-score) from the previous section as in this equation here:

\\[preliminary\text{\textunderscore}note\text{\textunderscore}score(n) = \frac{\sum_{rater \in R(n)} a(rater) * rating(rater,n)}{\sum_{rater \in R(n)} a(rater)}\\]

Where `a` is the [Author Helpfulness Score](./#author-helpfulness-score).

### Rater Helpfulness Score

Rater Helpfulness Score reflects how similar a contributor’s ratings are to the ratings on notes that were eventually labeled “Currently Rated Helpful” or “Currently Not Rated Helpful” (indicating clear consensus among raters, and not labeled “Needs More Ratings”).

Currently only the first 5 ratings on each note that were made within 48 hours of the note’s creation are used when evaluating a Rater Helpfulness Score (hereafter called “valid ratings”). This is done to both reward quick rating, and also so that retroactively rating old notes with clear labels doesn’t boost Rater Helpfulness Score.

Rater Helpfulness is initially 0 until the contributor has made at least one valid rating (one of the first 5 ratings within 48 hours on a note that got a label besides “Needs More Ratings”). Then the Rater Helpfulness Score is based on the fraction of their valid ratings that match the consensus label of whether the note was rated helpful or not rated helpful. When computing this fraction, we determine what the note’s label would’ve been without the rating from the rater whose score is being computed.

Then, this ratio is transformed in an identical way to how the Author Helpfulness Scores are smoothed: first by adding 2 pseudocounts to the numerator and 6 pseudocounts to the denominator (so that scores start out low and increase as raters make more valid ratings), and finally the score s is transformed with `max(0, 1.5s - .5)` in order to set ratios of ⅓ or less to 0 and then renormalize the range back to [0, 1].

\\[r(u) = max(0, \frac{3}{2} * \frac{2 + num\text{\textunderscore}valid\text{\textunderscore}ratings\text{\textunderscore}that\text{\textunderscore}match\text{\textunderscore}consensus(u)}{6 + num\text{\textunderscore}valid\text{\textunderscore}ratings(u)} - \frac{1}{2})\\]

Where `r` is the Rater Helpfulness Score of user `u`.

### Final Note Scoring

Next, [Author Helpfulness Score](./#author-helpfulness-score) and [Rater Helpfulness Score](./#rater-helpfulness-score) are averaged to get a combined weight for each contributor’s ratings, called [Combined Helpfulness Score](./#combined-helpfulness-score):

\\[c(u) = .5 * a(u) + .5 * r(u)\\]

Where `c` is the Combined Helpfulness Score, `a` is the Author Helpfulness Score, and `r` is the Rater Helpfulness Score of user `u`.

[Combined Helpfulness Scores](./#combined-helpfulness-score) are then used to weight each contributor’s ratings when scoring and ranking notes:
​
\\[note\text{\textunderscore}score(n) = \frac{\sum_{rater \in R(n)} c(rater) * rating(rater,n)}{\sum_{rater \in R(n)} c(rater)}\\]

Then, as long as notes received at least 5 ratings and 2 “weighted ratings” (the sum of the Combined Helpfulness Scores of the contributors who rated the note), notes are given the “Currently Rated Helpful” label if their final score is at least 0.84, and “Currently Rated Not Helpful” if the final score is at most 0.29. Otherwise, they get the label “Needs More Ratings”.

### Tie breaking of reasons

For tie-breaking, we use the following priority order of reasons:

Currently Rated Helpful Reasons:

1. `UniqueContext`
2. `Empathetic`
3. `GoodSources`
4. `Clear`
5. `Informative`
6. `Other`

<br>

Currently Not Rated Helpful Reasons:

1. `Outdated`
2. `SpamHarassmentOrAbuse`
3. `HardToUnderstand`
4. `OffTopic`
5. `Incorrect`
6. `ArgumentativeOrInflammatory`
7. `MissingKeyPoints`
8. `SourcesMissingOrUnreliable`
9. `OpinionSpeculationOrBias`
10. `Other`

<br>

## Ranking Code

Here you can find links to the real code that we are using to determine contributor’s helpfulness scores currently in the pilot. Although the code is currently only runnable inside the Twitter environment, we are releasing this code in the spirit of transparency, and we hope to be able to release code that will be runnable outside of Twitter’s production environment in the near future.

- [HelpfulnessScoresUtil.scala](/birdwatch/sourcecode/HelpfulnessScoresUtil.scala)

- [RatingAggregate.scala](/birdwatch/sourcecode/RatingAggregate.scala)

Here’s a Python code snippet one can run to reproduce how we compute notes’ rating statuses and sorting. It uses as input the notes and ratings files made available on the Birdwatch [Data Download](https://twitter.com/i/birdwatch/download-data) page.

- [Ranking-notes.py](/birdwatch/sourcecode/ranking-notes.py)

## Birdwatch Timelines

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

Example: if rater `A` rated notes 1, 2, and 3, while rater `B` rated notes 1, 3, 4, 5, and 6, then the `similarity score between A and B = 2 / min(3, 5) = 2/3`, because they both rated notes 1 and 3, for a total of 2, and the minimum number of total notes either rater has rated is 3. If two raters haven’t rated any of the same notes, the score is defined as 0.01 instead of 0. Then, the average rater similarity between a contributor and each other rater who’s rated a particular Tweet is computed, in order to compute the average rater similarity between the contributor and the raters of that Tweet. Then, Tweets are ranked in the **Needs your help** tab using the following score: `0.3 \* proportionOfNotesOnTweetThatNeedMoreRatings - averageRaterSimilarityScore`.

The factor of 0.3 is applied to help to balance between the two scores fairly equally. We may in the future experiment with using other forms of rater similarity, e.g. based on whether two contributors tend to agree when they rate the same notes. This version primarily reflects contributors’ interest in what they choose to rate.

## Birdwatch Cards on Tweets

When browsing Twitter on Android, iOS, or at Twitter.com, Birdwatch pilot participants may see cards on Tweets that have Birdwatch notes. Birdwatch cards may include a specific Birdwatch note or reference the number of Birdwatch notes on the Tweet, and link to the Birdwatch site where contributors can rate the notes.

### Cards showing a specific note

{{< figure src="../../images/notes-on-twitter-specific.png">}}

If a Tweet has a Birdwatch note deemed **Currently rated helpful** by contributors (determined as described above in **Note Ranking**) then the note will be shown in the card, along with a button to rate the note. If the Tweet has multiple notes with **Currently rated helpful** status, the card will show one note, randomly cycling between them at periodic intervals in order to gather additional rating inputs on each.

### Cards showing a count of notes

{{< figure src="../../images/notes-on-twitter-count.png">}}

If a Tweet has notes but none yet have **Currently rated helpful** status, the card will show the number of notes, and allow people to tap to read and rate those notes on the Birdwatch site.

If all notes on a Tweet have **Currently not rated helpful** status, no card will be shown. Pilot participants will still be able to access these notes via the Birdwatch icon displayed in the Tweet details page.
