---
title: Needs Your Help ranking
aliases: ["/timeline-tabs"]
geekdocBreadcrumb: false
geekdocToc: 1
description: How Community Notes are ranked to become eligible for features and notifications
---

Community Notes participants are able to see a tab of notes that could use their ratings, and receive notifications about notes that need help.

{{< figure src="../images/home.png" alt="Community Notes home page, showing Tweets with notes to be rated">}}

### Needs Your Help tab

This tab is only visible to contributors. It is designed to increase the likelihood that people from diverse perspectives rate each note, so that Community Notes can elevate notes that people from a wide range of perspectives will find helpful. It gives contributors an easy way to have immediate impact.

It contains a set of 10 Tweets that have notes that need more ratings (although there may be fewer than 10 Tweets if one of the Tweets was recently deleted by the author, or if not enough Tweets have new notes that meet the criteria to appear in the NYH tab). Tweets in this tab are filtered to those that the contributor hasn’t rated any of the notes on, and Tweets with notes from the past day, unless no Tweets pass those filters for you (that will only happen if you’re a very active rater!). The tab is updated roughly every hour or two, so when the contributor has rated notes in the tab, they can come back later to see fresh Tweets.

In order to appear in any of the tabs in the Community Notes site, a Tweet must have received at least 100 total likes plus Retweets.

Additionally, Community Notes offers a way for Tweet authors to [request additional review](../additional-review) on notes on their Tweets. If an author requests additional review, the relevant Tweet will appear in all contributors’ Needs Your Help tabs. If there are more than 10 active requests for additional review, Tweets will be sorted by by the ranking score described below, which incorporates the viewpoints of raters.

### Needs your help alerts

To ensure contributors don't miss the opportunity to rate notes on Tweets getting a lot of attention, Community Notes sends alerts requesting help from time to time.

{{< figure src="../images/alerts-settings.png" alt="One screenshot showing a Community Notes Contributor profile page, highlighting the settings button on the top right. Another screenshot showing the settings screen where contributors can edit their alert frequency">}}

### How does Community Notes decide which notes trigger alerts for help?

Alerts are optimized to increase the chance that potentially helpful notes on Tweets with high predicted visibility get rated by enough people that they have the chance to earn a status of Helpful quickly. Here's our current approach:

**Tweets are chosen based on:**

- Projected future Likes and Retweets the Tweet will receive.
- The rater can see the Tweet (for example, excludes Tweets from authors you've blocked)

**Notes are chosen based on meeting some of the following:**

- Written by an author with high writing impact, or high average helpfulness score of notes they authored.
- Currently have a status of "Needs More Ratings"
- Currently have a high helpfulness score, nearing the threshold to earn status of "Helpful"
- Do not have a large number of ratings (such that more ratings could change the note's status)

Notes are sent to a random selection of contributors, excluding the note author and those who have already rated the note. Notifications are also limited by the recipient's notification frequency setting.

We will continue to experiment with this logic to help ensure Needs Your Help notifications feel high impact and satisfying for contributors.

### Rater similarity score for Needs Your Help

Tweets on the Needs Your Help tab are sorted by a ranking score, where Tweets are ranked higher based on the proportion of notes on the Tweet that are labeled Needs More Rating. Tweets are also ranked higher if they have notes with moderately high intercept scores (>=0.25) but which don’t have at least 3 raters with a similar viewpoint as yours (measured by whether your viewpoint factor is positive or negative). This is a mechanism to increase the likelihood that people from diverse perspectives rate each note.
