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

Birdwatch notes are submitted and rated by contributors. Ratings are used to determine note status, which notes are displayed on each of the [Birdwatch Site's timelines](./#birdwatch-timelines), and which notes are displayed as [Birdwatch Cards](./#birdwatch-cards-on-tweets) on the Twitter timelines.

To help people understand how Birdwatch's ranking system works, the sections below explain how notes are assigned these statuses, ranked, and displayed.

{{< toc >}}

## Note Status

{{< figure src="../../images/note-statuses.png">}}

All Birdwatch notes start out with the **Needs More Ratings** status until they receive at least 5 total ratings and at least 2.5 weighted ratings (the sum of the Contributor Scores of the raters that rated the note, where Contributor Scores are between 0 and 1).

All Birdwatch notes start out with the **Needs More Ratings** status until they receive at least 2 weighted ratings (the sum of the Contributor Scores that rated the note). Since contributor weights are between 0 and 1, and are around 0.5 on average for contributors with nonzero weights, this means in practice that about 4 contributors with nonzero weights will need to have rated the note.

Then, Birdwatch computes a **Helpfulness Ratio** for each note, which is simply the proportion of ratings that say the note is helpful.

For a note to be labeled as **Currently Rated Helpful** or **Currently Not Rated Helpful**, raters must have selected at least two of the corresponding
reasons why a note is (or isn’t) helpful — e.g. “cites high-quality sources,” “nonjudgmental and/or empathetic,” etc — at least twice each.

Notes with a **Helpfulness Ratio** of 0.29 and below are assigned **Currently Not Rated Helpful**, and notes with a ratio of 0.84 and above are assigned **Currently Rated Helpful**. Notes with scores in between 0.29 and 0.84 remain labeled as **Needs more Ratings**.

In addition to **Currently Rated / Not Rated Helpful** status, labels also show the most commonly chosen reasons why it is helpful/unhelpful. To break ties between reasons that are chosen an equal number of times by raters, we pick the reason that is used least frequently by Birdwatch raters in general (with the exception of Other, which loses all tiebreaks).

Notes with the status **Needs More Ratings** remain sorted by recency (newest first), and notes with a **Currently Rated / Not Rated Helpful** status are sorted by their **Helpfulness Ratio**.

This ranking mechanism is knowingly basic, and we only intend to use it for a short time during the program’s earliest phases.

During the pilot, rating statuses are only computed at periodic intervals, so there is a time delay from when a note meets the **Currently Rated / Not Rated Helpful** criteria and when it jumps to the top or bottom of the list. This delay allows Birdwatch to collect a set of independent ratings from people who haven’t yet been influenced by seeing status annotations on certain notes.

## Contributor Scores

For each contributor, we will compute two scores: the Author Helpfulness Score score, and the Contributor Ratings Score. These two scores, in turn, are averaged together into the Average Contributor Score, which is used to weight the ratings each contributor produces when evaluating notes. The purpose of using Contributor Scores is to give more influence to contributors who have a track record of helpful contributions.

At a high level, Contributor Scores are a way to give more weight to people whose contributions are consistently found helpful by others:

- If you write notes that people consistently find helpful, the ratings you give other notes will have greater weight.
- If the ratings you give often align with the ultimate rating outcome from the community, your ratings will have greater weight.

<br>

This is a simple start to improve the quality of Birdwatch’s note scoring and ranking, and is a step towards making manipulation of Birdwatch more difficult.. More improvements are on their way, including factoring in not just how many ratings a note has received, but also how diverse the perspectives of those raters are — i.e. do they always agree with each other, or sometimes/often disagree?

### Author Helpfulness Score

Author Helpfulness Score
Any contributor who has written at least one note that’s been rated by other participants receives an Author Helpfulness Score. This score is defined as the weighted average of ratings they’ve received from other contributors on the notes that they’ve authored, with 10 “pseudocounts” . We initially assign the Author Helpfulness Score of all contributors to be 1, then repeat the following reweighting update step defined below until the scores converge.

Intuitively, the process starts off assuming all raters have equal weight, and calculates each note author’s Helpfulness Score solely based on the average of all the ratings they’ve received. That gives a starting point of which authors’ contributions are found most helpful by others. For the purpose of computing these scores, the process assumes those more helpful writers are also better raters, and re-computes all Author Helpfulness Scores, giving the ratings of helpful writers more weight. Repeat until this process hits a maximum number of iterations, at which point the helpfulness scores should have converged (stopped changing much between iterations):

\\[helpfulness_i(u) = \frac{\sum_{rater \in R(u)} helpfulness_{i-1}(rater) * rating(rater,u)}{\sum_{rater \in R(u)} helpfulness_{i-1} (rater)}\\]

### Preliminary Note Scoring

The next step is to compute preliminary Note Helpfulness Scores, which are only used to compute Contributor Rating Scores as defined below (they are not used as final Note Helpfulness Scores). The preliminary scores are computed by weighting each rater’s Author Helpfulness Scores from the previous section as in this equation here:

\\[helpfulness_i(u) = \frac{\sum_{rater \in R(u)} helpfulness_{i-1}(rater) * rating(rater,u)}{\sum_{rater \in R(u)} helpfulness_{i-1} (rater)}\\]

### Contributor Rating Score

To determine Contributor Rating Score, we measure how similar the contributor’s ratings are to the ratings on notes that were eventually labeled “Currently Rated Helpful” or “Currently Not Rated Helpful” (indicating clear consensus among raters and they aren’t labeled “Needs More Ratings”).

We currently only use the first 5 ratings on each note and within 48 hours of the note’s creation when evaluating a Contributor’s Rating Score (which we will call “valid ratings”), both as an incentive to reward quick rating but also so that retroactively rating old notes with clear labels doesn’t boost Contributor Rating Score.

Until a contributor has made valid ratings on at least 10 notes that aren’t labeled “Needs More Ratings” and they didn’t rate every note the exact same way, that contributor’s Contributor Rating Score is 0. Once they have, then the Contributor Rating Score is defined as the correlation of that rater’s ratings with the preliminary Note Helpfulness Scores for the notes that have labels besides “Needs More Ratings”, after excluding the rater’s own rating from the preliminary Note Helpfulness Score.

\\[helpfulness_i(u) = \frac{\sum_{rater \in R(u)} helpfulness_{i-1}(rater) * rating(rater,u)}{\sum_{rater \in R(u)} helpfulness_{i-1} (rater)}\\]

Because it’s impossible to have a correlation of 1 if you ever rate notes with preliminary Note Helpfulness Scores that aren’t exactly 1.0, we then normalize this score by the maximum possible correlation from rating the set of notes that they chose to rate. This means raters aren’t unfairly penalized for rating notes where there isn’t perfect consensus.

### Final Note Scoring

Then, we average the Author Helpfulness Score and Contributor Rating Score to get a combined weight for each contributor (Average Contributor Score):

Average Contributor Scores are then used to weight each contributor’s ratings when scoring and ranking notes:

\\[helpfulness_i(u) = \frac{\sum_{rater \in R(u)} helpfulness_{i-1}(rater) * rating(rater,u)}{\sum_{rater \in R(u)} helpfulness_{i-1} (rater)}\\]

Then, as long as notes received at least 5 ratings and 2.5 “weighted ratings” (the sum of the Average Contributor Scores of the contributors who rated the note), notes are given the “Currently Rated Helpful” label if their final score is at least 0.84, and “Currently Rated Not Helpful” if the final score is at most 0.29. Otherwise, they get the label “Needs More Ratings”.

Helpfulness
\\[helpfulness_i(u) = \frac{\sum_{rater \in R(u)} helpfulness_{i-1}(rater) * rating(rater,u)}{\sum_{rater \in R(u)} helpfulness_{i-1} (rater)}\\]

Note Score
\\[score(n) = \frac{\sum_{rater \in R(n)} weight(rater) * rating(rater,n)}{\sum_{rater \in R(n)} weight(rater)}\\]

Preliminary Note Score
\\[preliminary\_note\_score(n) = \frac{\sum_{rater \in R(n)} helpfulness(rater) * rating(rater,n)}{\sum_{rater \in R(n)} helpfulness(rater)}\\]

Weight
\\[
weight(rater) = .5 * helpfulness(rater) + .5 * reliability(rater)
\\]

Reliability
\\[
reliability(u) = \rho(rating(u,n), preliminary\textunderscorenote\text{\textunderscore}score(n))
\\]

Reliability

​

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

## Ranking Code

Here’s a Python code snippet one can run to reproduce how we compute notes’ rating statuses and sorting. It uses as input the notes and ratings files made available on the Birdwatch [Data Download](https://twitter.com/i/birdwatch/download-data) page.

{{< include file="static/sourcecode/ranking-notes.py" language="python" markdown=false >}}

{{< button href="/birdwatch/sourcecode/ranking-notes.py" >}}See source file{{< /button >}}

## Birdwatch timelines

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
