---
title: Timeline Tabs
aliases: ["/timeline-tabs"]
geekdocBreadcrumb: false
geekdocToc: 1
---

Birdwatch participants are able to view three different tabs on the [Birdwatch Home](http://birdwatch.twitter.com) page. Each tab contains different sets of Tweets in different orders. In order to appear in any of these tabs, a Tweet must have received at least 100 total likes plus Retweets.

{{< figure src="../images/home-three-tabs.png">}}

### New

Contains Tweets sorted by reverse chronological order of when its latest note was written (Tweets are bumped to the top of the list when a new note is written on them).

### Rated Helpful

Tweets in this tab have at least one note labeled Currently Rated Helpful, at least one of those Currently Rated Helpful notes must have labeled the Tweet “misinformed or potentially misleading”, and a majority of those Currently Rated Helpful notes had to have not labeled the Tweet as satire (either “It is a joke or satire that might be misinterpreted as a fact” or “It is clearly satirical/joking”). The Tweets that pass these filters are sorted reverse-chronologically by the timestamp of the Tweet’s first-created Currently Rated Helpful note.

### Needs Your Help

This tab is only visible to pilot participants. It is designed to increase the likelihood that people from diverse perspectives rate each note, so that Birdwatch can ultimately elevate notes that people from a wide range of perspectives will find helpful. It also gives Birdwatch contributors an easy way to have immediate impact.

It contains a set of 5 Tweets that have notes that need more ratings (although there may be fewer than 5 Tweets if one of the Tweets was recently deleted by the author). Tweets in this tab are filtered to those that the contributor hasn’t rated any of the notes on, and Tweets with notes from the past day, unless no Tweets pass those filters for you (that will only happen if you’re a very active rater!). The tab is updated roughly every couple hours, so when the contributor has rated notes in the tab, they can come back later to see fresh Tweets.

Additionally, Birdwatch offers a way for Tweet authors to [request additional review](../additional-review) on notes on their Tweets. If an author requests additional review, the relevant Tweet will appear in all contributors’ Needs Your Help tabs. If there are more than 5 active requests for additional review, Tweets will be sorted by rater similar scores so as to broaden the diversity of raters who will see a Tweet.

### Rater similarity score for Needs Your Help

Tweets on the Needs Your Help tab are sorted by a ranking score, where Tweets are ranked higher based on the proportion of notes on the Tweet that are labeled Needs More Ratings, and Tweets are ranked lower if your rating history is more similar to the other raters who’ve already rated notes on this Tweet: this is a mechanism to help increase the likelihood that people from diverse perspectives rate each note.

Specifically, the rater similarity score between a pair of raters is defined as the number of notes that both raters rated, divided by the minimum number of total notes each rater has rated(over each of the two raters).

Example: if rater A rated notes 1, 2, and 3, while rater B rated notes 1, 3, 4, 5, and 6, then the similarity score between A and B = 2 / min(3, 5) = 2/3, because they both rated notes 1 and 3, for a total of 2, and the minimum number of total notes either rater has rated is 3. If two raters haven’t rated any of the same notes, the score is defined as 0.01 instead of 0. Then, the average rater similarity between a contributor and each other rater who’s rated a particular Tweet is computed, in order to compute the average rater similarity between the contributor and the raters of that Tweet. Then, Tweets are ranked in the Needs your help tab using the following score: 0.3 \* proportionOfNotesOnTweetThatNeedMoreRatings - averageRaterSimilarityScore.

The factor of 0.3 is applied to help to balance between the two scores fairly equally. We may in the future experiment with using other forms of rater similarity, e.g. based on whether two contributors tend to agree when they rate the same notes. This version primarily reflects contributors’ interest in what they choose to rate.
