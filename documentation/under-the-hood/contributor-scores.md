---
title: Contributor helpfulness scores
enableMathJax: true
description: Helpfulness scores are a way to give more influence to people with a track record of making high-quality contributions to Community Notes.
navWeight: 5
---
# Contributor helpfulness scores

Helpfulness scores are a way to give more influence to people with a track record of making high-quality contributions to Community Notes. There are currently two types of author helpfulness scores and one rater helpfulness score.

In order to get enough data from new raters to be able to assess how similarly they rate notes to others, we require a minimum of 10 ratings made before helpfulness scores are computed and ratings may be counted. Additionally, to help mitigate misuse of Community Notes, contributors with helpfulness scores that are too low are filtered out, since those contributors are consistently not found helpful by a [diverse set of raters](../contributing/diversity-of-perspectives.md).

## Author Helpfulness Scores

### Author Helpful vs. Not Helpful Ratio

This score is the proportion of notes you’ve written (that have gotten at least 5 ratings) that have reached the status of Helpful ("Currently Rated Helpful", or CRH), minus 5 times the proportion of notes you wrote that reached the status of Not Helpful ("Currently Rated Not Helpful", or CRNH).

Contributors must have a ratio of at least 0.0 to be included in the [second round of note scoring](./ranking-notes.md) (contributors need to write at least 5 CRH notes for every 1 CRNH note they write in order for their ratings to count); this filters out a small percentage of raters. Labels on notes that have been deleted after May 19, 2022 continue to affect this score, so that the score can’t be trivially changed by deleting CRNH notes.

### Author Mean Note Score

This score is the average score of notes you’ve written (that have gotten at least 5 ratings). Ratings are filtered out from the small percentage of users whose scores are less than 0.05. Labels on notes that have been deleted after May 19, 2022 continue to affect this score, so that the score can’t be trivially changed by deleting CRNH notes.

## Rater Helpfulness Score

The Rater Helpfulness Score reflects how similar a contributor’s ratings are to the ratings on notes that eventually reached the status of “Helpful” or "Not Helpful” (indicating clear widespread consensus among raters, and not labeled “Needs More Ratings”).

First, we compute each raters' [Valid Ratings](#valid-ratings) are used in computing rater helpfulness scores. This is done to both reward quick rating, and to prevent one form of artificially gaming this score (retroactively rating old notes with clear labels).

Rater Helpfulness is not defined until the contributor has made at least one valid rating (defined below). Then the Rater Helpfulness Score is equal to 

$$\frac{s - (10 * h)}{t}$$
With the terms defined as:
* (s): the number of successful valid ratings they've made (valid ratings that matched the final note status label)
* (-10 * h): minus a penalty of 10 for note they rated helpful (valid or not) with an extremely [Tag-Consensus Harassment-Abuse Note Score](./ranking-notes.md#tag-consensus-harassment-abuse-note-score)
* (t): divided by the total number of valid ratings they've made

## Valid Ratings

A “valid” rating is a rating that’s eligible to be used in determining rater helpfulness scores. The idea is that to prevent manipulation, only ratings that were made before the rater could’ve known what the final note status label is are eligible. To be specific, valid ratings must be:

- Made within the first 48 hours of the note’s creation (because we publicly release all rating data after 48 hours)
- A rating on a note that eventually ended up getting a Helpful or Not Helpful status, so we can compute whether your rating matched the final note status
- If the note being rated was created before May 18, 2022:
  - Only the first 5 ratings on the note are valid (since note status labels aren’t computed until the note has at least 5 ratings)
- If the note being rated was created on or after May 18, 2022:
  - To be valid, the rating must be made before the time of when the note received its first status besides Needs More Ratings. Or, if the note’s status changed from Helpful to Not Helpful or vice versa, then all ratings will be valid as long as they are made before the timestamp of when that most recent status change occurred.

Ratings have the same impact on the note’s final status label whether they are “valid” or not. Whether a rating is valid is only relevant for the computation of rater helpfulness scores.

## Filtering Ratings Based on Helpfulness Scores

Community Notes gives more weight to contributors who are good at identifying which notes will be helpful (or unhelpful) to people from different points of view. This helps improve note scoring and ranking, and makes manipulation of Community Notes more difficult.

Community Notes does this by incorporating a subset of ratings in a second round of note scoring. Contributors’ ratings are only included in the second round of note scoring if:

- They have made at least 10 total ratings (on notes that have at least 5 ratings) and have made at least 1 [valid rating](#valid-ratings).
- Their rater helpfulness score must be at least 0.66
- If they have written any notes that have received at least 5 ratings (contributors who haven’t written any such notes are included):
  - The CRH-vs.-CRNH ratio of notes they’ve authored must be at least 0.0
  - The mean note score of all notes they’ve written must be at least 0.05

These scores are a simple first approach to improving quality and inhibiting manipulation, and more improvements are on the way.
