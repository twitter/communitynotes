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
math: true
description: How are Community Notes ranked?
---

Community Notes are submitted and rated by contributors. Ratings are used to determine note statuses (“Helpful”, “Not Helpful”, or “Needs More Ratings”). Note statuses determine which notes are displayed on each of the [Community Notes Site’s timelines](../birdwatch-timelines/), and which notes are displayed [on Tweets](../notes-on-twitter/).

At this time, only notes that indicate a Tweet is “potentially misleading” are eligible to be displayed on Tweets.
We observed that notes marking a Tweet as "not misleading" were often rated as “Unhelpful - Tweet doesn’t need a note”.
On March 9, 2022 we paused assigning status to notes marking Tweets as "not misleading" pending improvements to the rating form.
On October 3, 2022 we updated the rating form to better capture the strengths of notes which add context without indicating the Tweet is misleading.
As discussed in Note Status below, we have resumed assigning status to notes marking Tweets as "not misleading" in select circumstances as we evaluate ranking quality and utility to users.

The sections below describe how notes are assigned statuses, which determines how notes are ranked and displayed in the product:

{{< toc >}}

## Note Status

{{< figure src="../images/note-statuses.png" alt="Three Community notes in different statuses">}}

All Community Notes start with the Needs More Ratings status until they receive at least 5 total ratings.
Notes with 5 or more ratings may be assigned a status of Helpful or Not Helpful according to the algorithm described below.
If a note is deleted, the algorithm will still score it (using all non-deleted ratings of that note) and the note will receive a status if it’s been rated more than 5 times, although since it is deleted it will not be shown on Twitter even if its status is Helpful.

Notes marking Tweets as "potentially misleading" with a Note Helpfulness Score of 0.40 and above earn the status of Helpful.
Notes with a Note Helpfulness Score less than -0.05 -0.8 \* abs(noteFactorScore) are assigned Not Helpful, where noteFactorScore is described in [Matrix Factorization](#matrix-factorization).
Notes with scores in between remain with a status of Needs more Ratings.

Notes marking Tweets as "not misleading" with a Note Helpfulness Score below -0.15 earn the status of Not Helpful.
Identifying notes as Not Helpful improves contributor helpfulness scoring and reduces time contributors spend reviewing low quality notes.
At this time, any note marking a Tweet as "not misleading" with a Note Helpfulness Score above -0.15 remains in the status of Needs More Ratings.
We plan to enable Helpful statuses for notes marking Tweets as "not misleading" as we continue to evaluate ranking quality and utility to users.

When a note reaches a status of Helpful / Not Helpful, they're shown alongside the two most commonly chosen explanation tags which describe the reason the note was rated helpful or unhelpful.
Notes with the status Needs More Ratings remain sorted by recency (newest first), and notes with a Helpful or Not Helpful status are sorted by their Helpfulness Score.

This ranking mechanism is subject to continuous development and improvement with the aim that Community Notes consistently identifies notes that are found helpful to people from a wide variety of perspectives.

Rating statuses are computed at periodic intervals, so there is a time delay from when a note meets the Helpful / Not Helpful criteria and when that designation appears on the Community Notes site.
This delay allows Community Notes to collect a set of independent ratings from people who haven’t yet been influenced by seeing status annotations on certain notes.

## Helpful Rating Mapping

{{< figure src="../images/helpful-ranking.png" alt="Mockup of the rating prompt for rating notes. It reads: ”Is this note helpful?” And presents three options: Yes, Somewhat, or No">}}

When rating notes, contributors answer the question “Is this note helpful?” Answers to that question are then used to rank notes. When Community Notes (formerly called Birdwatch) launched in January 2021, people could answer “yes” or “no” to that question. An update on June 30, 2021 allows people to choose between “yes,” “somewhat” and “no.” We map these responses to continuous values from 0.0 to 1.0, hereafter referred to as “helpful scores”:

- `Yes` maps to `1.0`
- `Somewhat` maps to `0.5`.
- `No` maps to `0.0`.
- `Yes` from the original 2-option version of the rating form maps to `1.0`.
- `No` from the original 2-option version of the rating form maps to `0.0`.

Specific values in the mapping may change in the future, and will be updated here.

## Matrix Factorization

The main technique we use to determine which notes are helpful or unhelpful is matrix factorization on the note-rater matrix, a sparse matrix that encodes, for each note and rater, whether that rater found the note to be helpful or unhelpful. This approach, originally made famous by Funk in the 2006 Netflix prize recommender system competition, seeks a latent representation (embedding) of users and items which can explain the affinity of certain users for certain items [[1](https://sifter.org/~simon/journal/20061211.html)] [[2](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)]. In our application, this representation space identifies whether notes may appeal to raters with specific viewpoints, and as a result we are able to identify notes with broad appeal across viewpoints.

One challenge is that not all raters evaluate all notes - in fact most raters do not rate most notes - and this sparsity leads to outliers and noise in the data. Regularization techniques are a common solution to these issues; a key distinction in our approach is that we use much higher regularization on the intercept terms, which capture the helpfulness of a note or rater that is not explained by viewpoint agreement, relative to the embedding factors. This encourages a representation that uses user and note embeddings to explain as much variation in the ratings as possible before fitting additional note- and user-specific intercepts. As a result, for a note to achieve a high intercept term (which is the note’s helpfulness score), it must be rated helpful by raters with a diversity of viewpoints (factor embeddings). Notes are given a single global score-- their intercept term-- rather than using this algorithm in the traditional way to personalize content as in a recommender system.

We predict each rating as:

$$ \hat{r}\_{un} = \mu + i_u + i_n + f_u \cdot f_n $$

Where the prediction is the sum of three intercept terms: $\mu$ is the global intercept term, $i_u$ is the user’s intercept term, and $i_n$ is the note’s intercept term, added to the dot product of the user and notes’ factor vectors $f_u$ and $f_n$ (note that when user and note factors are close, a user is expected to give a higher rating to the note).

To fit the model parameters, we minimize the following regularized least squared error loss function via gradient descent over the dataset of all observed ratings $r_{un}$:

$$ \sum*{r*{un}} (r*{un} - \hat{r}*{un})^2 + \lambda_i (i_u^2 + i_n^2 + \mu^2) + \lambda_f (||f_u||^2 + ||f_n||^2) $$

Where $\lambda_i=0.03$, the regularization on the intercept terms, is currently 5 times higher than $\lambda_f=0.15$, the regularization on the factors.

The resulting scores that we use for each note are the note intercept terms $i_n$. These scores on our current data give an approximately Normal distribution, where notes with the highest and lowest intercepts tend to have factors closer to zero.

In general, we set the thresholds to achieve a “Helpful” status at 0.40, including less than 10% of the notes, and our threshold to achieve a “Not Helpful” status at $-0.05 - 0.8 \* abs(f_n)$. The [Tag Outlier Filtering](#tag-outlier-filtering) section describes an extension to the general thresholds.

This approach has a few nice properties:

- Extra regularization on the intercept terms in practice requires that notes are rated by raters with diverse factors before a note gets a label (the note intercept term becomes very large or very small)
- We can represent multidimensional viewpoint spaces by increasing the dimensionality of the factors, without changing the algorithm itself
- Rater-specific intercept terms capture how lenient or generous each rater is with their helpful ratings
- We are able to include somewhat helpful ratings naturally as 0.5s

Note: for now, to avoid overfitting on our very small dataset, we only use 1-dimensional factors. We expect to increase this dimensionality as our dataset size grows significantly.

## Tag Outlier Filtering

In some cases, a note may appear helpful but miss key points about the tweet or lack sources.
Reviewers who rate a note as "Not Helpful" can associate [tag](examples#helpful-attributes) with their review to identify specific shortcomings of the note.
When a note has receives high levels of a "Not Helpful" tag, we require a higher intercept before rating the note as "Helpful".
This approach helps us to maintain data quality by recognizing when there is a troubling pattern on an otherwise strong note.

We define the quantity $a_{un}$ to represent the _weight_ given to tag $a$ identified by reviewer (user) $u$ on note $n$:

$$ a*{un} = \mathbb{1}*{aun} \left( 1 + \left( {{||f_u - f_n||} \over {\tilde{f}}} \right)^2 \right) ^{-1} $$

Where:

- $\tilde{f} = median_{r_{un}}(||f_n - f_r||)$ indicates the median distance between the reviewer and note latent factors over all observable reviews $r_{un}$
- $\mathbb{1}_{aun}$ is 1 if reviewer $u$ assigned tag $a$ to note $n$ and 0 otherwise.

We define the total weight of an tag $a$ on note $n$ as:

$$ n*{a} = \sum*{r*{un}} a*{un} $$

Notice the following:

- No single review can achieve an tag weight $a_{un} > 1$.
- Reviews where the reviewer factor and note factor are equal will achieve the maximum weight of 1.0, reviews at a median distance will achieve a weight of 0.5, and reviews at 2x the median distance will have a weight of 0.2. All reviews will have positive weight.
- Assigning higher weights to tags in reviews where the reviewer and note are closer in the embedding space effectively lends greater weight to critical reviews from reviewers who tend to share the same perspective as the note.

Given the quantities defined above, we modify scoring as follows:

- When the total weight $a_n$ of an tag exceeds 1.5 _and_ is in the 95th percentile of all notes with an intercept greater than 0.4, we require the intercept to exceed 0.5 before marking the note as helpful.
- We disregard the "Typos or unclear language" and "Note not needed on this Tweet" tags, which do not relate to note accuracy.

## CRH Inertia

The scoring algorithm updates the Helpful status of each note during every invocation.
Re-computing the Helpful status ensures that as new ratings emerge the notes shown to users continue to reflect a broad consensus among raters.
In some cases, small variations in the note intercept $i_n$ can cause notes to lose and potentially re-acquire Helpful status.

To help ensure that changes in Helpful status reflect a clear shift in consensus, we require that the note intercept $i_n$ drops below the applicable threshold by more than 0.01 before the note loses Helpful status.
For example, if a note achieved Helpful status with note intercept $i_n>0.40$, then the note would need $i_n<0.39$ before losing Helpful status.
Similarly, if a note was impacted by tag outlier filter and required note intercedpt $i_n>0.50$ to achieve Helpful status, the note would need $i_n<0.49$ to lose Helpful status.

## Status Stabilization

As Community Notes has scaled from inception to global availability we've seen an increasing number of notes and ratings spanning a widening array of topics.
With the increased volume of community contributions, ranking data for older and newer notes has diverged: newer notes are able to receive more ratings from a wider range of contributors while the available ranking data for older notes remains more limited.
As older data comprise an increasingly small fraction of the dataset, ranking results have tended to fluctuate and some notes have lost Helpful status.

To maintain Helpful note quality as Community Notes continues to grow, we are adding logic which stabilizes the status of a note once the note is two weeks old.
This approach allows us to continue optimizing the ranking algorithm with a focus on the impact on current data while persisting helpful community contributions on older topics.
Before a note is two weeks old, the helpfulness status will continue to be updated each time time the ranking algorithm is run.
After a note turns two weeks old we store the helpfulness status for that note and use the stored status in the future, including for displaying notes on Twitter and calcualting user contribution statistics.

## Determining Note Status Explanation Tags

When notes reach a status of Helpful or Not Helpful, they're displayed alongside the top two explanation tags that were given by raters to explain why they rated the note helpful or not.

This is done by counting the number of times each explanation tag was given filtered to explanation tags that match the final note status (e.g., if the note status is Helpful we only count helpful explanation tags). Importantly, each explanation tag must be used by at least two different raters. If there aren’t two different tags that are each used by two different raters, then the note’s status is reverted to “Needs More Ratings” (this is rare).

We break ties between multiple explanation tags by picking the less commonly used reasons, given in order below (#1 is the least commonly used and therefore wins all tiebreaks).

For helpful notes:

```
1. UnbiasedLanguage
2. UniqueContext
3. Empathetic
4. GoodSources
5. AddressesClaim
6. ImportantContext
7. Clear
8. Informative
9. Other
```

<br/>

For not-helpful notes:

```
1. Outdated
2. SpamHarassmentOrAbuse
3. HardToUnderstand
4. OffTopic
5. Incorrect
6. ArgumentativeOrInflammatory
7. NoteNotNeeded
8. MissingKeyPoints
9. OpinionSpeculation
10. SourcesMissingOrUnreliable
11. OpinionSpeculationOrBias
12. Other
```

</br>

## Complete Algorithm Steps:

1. <div>Pre-filter the data: to address sparsity issues, only raters with at least 10 ratings and notes with at least 5 ratings are included (although we don’t recursively filter until convergence).</div>
2. <div>Fit matrix factorization model, then assign intermediate note status labels for notes whose intercept terms (scores) are above or below thresholds.</div>
3. <div>Compute Author and Rater Helpfulness Scores based on the results of the first matrix factorization, then filter out raters with low helpfulness scores from the ratings data as described in <a href="../contributor-scores/#filtering-ratings-based-on-helpfulness-scores">Filtering Ratings Based on Helpfulness Scores</a>.</div>
4. <div>Re-fit the matrix factorization model on the ratings data that’s been filtered further in step 3.</div>
5. <div>Update status labels for any notes written within the last two weeks based the intercept terms (scores) and ratings tags.  Stabilize helpfulness status for any notes older than two weeks.</div>
6. <div>Assign the top two explanation tags that match the note’s final status label as in <a href="./#determining-note-status-explanation-tags">Determining Note Status Explanation Tags</a>, or if two such tags don’t exist, then revert the note status label to “Needs More Ratings”.</div>

<br/>

## What’s New?

**January 20, 2022**

- Updated the helpfulness status of all notes to the historical status on August 15, 2022 or two weeks after note creation, whichever was later. Notes created within the last two weeks were unimpacted. We selected August 15, 2022 to include as many scoring improvements as possible while still predating data changes caused by scaling Community Notes.

**January 17, 2022**

- Introduced logic stabilizing helpfulness status once a note is two weeks old.
- Added an experimental extension to the matrix factorization approach to quantify how much a note intercept may fluctuate.

**December 6, 2022**

- Corrected calculation of how many notes a contributor has rated after a status decision was made.
- Corrected authorship attribution within noteStatusHistory to eliminate `nan` author values in note scoring output.
- Improved usage of Pandas to eliminate warnings.
- Removed logging from contributor_state which is no longer necessary.
- Refactored insufficient tag scoring logic to fit within updated scoring framework.

**November 30, 2022**

- Improved thresholding logic to help ensure notes only lose CRH status when there is a clear change in rater consensus.

**November 25, 2022**

- Resumed assigning statuses to notes that indicate the Tweet is “not misleading.” Only such notes written after October 3, 2022 will be eligible to receive statuses, as on that date we [updated the rating form](https://twitter.com/CommunityNotes/status/1576981914296102912) to better capture the helpfulness of notes indicating the Tweet is not misleading.

**November 10, 2022**

- Launched scoring logic adjusting standards for "Helpful" notes based on tags assigned in reviews labeling the note as "Not Helpful."

**July 13, 2022**

- To prevent manipulation of helpfulness scores through deletion of notes, notes that are deleted will continue to be assigned note statuses based on the ratings they received. These statuses are factored into author helpfulness scores.
- Valid Ratings Definition Update: instead of just the first 5 ratings on a note, all ratings will be valid if they are within the first 48 hours after note creation and were created before the note first received its status of Helpful or Not Helpful (or if its status flipped between Helpful and Not Helpful, then all ratings will be valid up until that flip occurred).
- To make the above two changes possible, we are releasing a new dataset, note status history, which contains timestamps for when each note received statuses, and the timestamp and hashed participant ID of the author of a note. This data file is being populated now and will be available on the Community Notes [Data Download](https://twitter.com/i/communitynotes/download-data) page beginning Monday July 18, 2022.

**Mar 09, 2022**

- Temporarily paused assigning statuses to notes that indicate the Tweet is “not misleading”
- Adjusted thresholds for notes statuses

**Feb 28, 2022**

- Launched new algorithm to compute note statuses (Helpful, Not Helpful), which looks for agreement across different viewpoints using a matrix factorization method. Updates contributor helpfulness scores to reflect helpfulness so that contributors whose contributions are helpful to people from a wide range of viewpoints earn higher scores. Uses helpfulness scores to identify a subset of contributor ratings to include in a final round of note scoring. This replaces the previous algorithm which weighted ratings by raters’ helpfulness scores.

**June 30, 2021**

- Added a new “somewhat” option to the “is this note helpful?” question in the rating form.

**June 14, 2021**

- New algorithm to compute note status, which weights ratings by raters’ helpfulness scores, rather than taking a direct average of ratings.

**January 28, 2021**

- Community Notes (formerly called Birdwatch) pilot begins, with note ranking done via simply computing the raw ratio of helpful ratings.
