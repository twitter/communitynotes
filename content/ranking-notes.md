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

Birdwatch notes are submitted and rated by contributors. Ratings are used to determine note status labels (“Currently Rated Helpful”, “Currently Rated Not Helpful”, or “Needs More Ratings”). Note statuses also determine which notes are displayed on each of the [Birdwatch Site’s timelines](../birdwatch-timelines/), and which notes are displayed as [Birdwatch Cards](../notes-on-twitter/) on Tweets.

Only notes that indicate the Tweet as “potentially misleading” are eligible to be displayed as Birdwatch Cards on Tweets; notes that indicate the Tweet is “not misleading” are not displayed as Birdwatch Cards on Tweets. As of March 9, 2022, we have temporarily paused assigning statuses to notes that indicate the Tweet is “not misleading.” Why? People have been confused about how to rate them. As these notes are effectively making the case that the Tweet does not need a note, raters often rated them as “Unhelpful - Tweet doesn’t need a note” so as to indicate the note should not appear on the Tweet. We are planning an improvement to the rating form to resolve this confusion, and plan to resume assigning statuses to “not misleading” notes once that’s in place.

The sections below describe how notes are assigned statuses, which determines how notes are ranked and displayed in the product:

{{< toc >}}

## Note Status

{{< figure src="../images/note-statuses.png">}}

All Birdwatch notes start out with the Needs More Ratings status until they receive at least 5 total ratings. Once a note has received at least 5 ratings, it is assigned a Note Helpfulness Score according to the algorithm described below.

Notes with a Note Helpfulness Score of –0.08 and below are assigned Currently Not Rated Helpful, and notes with a score of 0.40 and above are assigned Currently Rated Helpful. Notes with scores in between –0.08 and 0.40 remain labeled as Needs more Ratings.

In addition to Currently Rated / Not Rated Helpful status, labels also show the two most commonly chosen explanation tags which describe the reason the note was rated helpful/unhelpful.

Notes with the status Needs More Ratings remain sorted by recency (newest first), and notes with a Currently Rated / Not Rated Helpful status are sorted by their Helpfulness Score.

This ranking mechanism is subject to continuous development and improvement with the aim that Birdwatch consistently identifies notes that are found helpful to people from a wide variety of perspectives.

During the pilot phase, rating statuses are only computed at periodic intervals, so there is a time delay from when a note meets the Currently Rated / Not Rated Helpful criteria and when that designation appears on the Birdwatch site. This delay allows Birdwatch to collect a set of independent ratings from people who haven’t yet been influenced by seeing status annotations on certain notes.

## Helpful Rating Mapping

{{< figure src="../images/helpful-ranking.png">}}

When rating notes, Birdwatch contributors answer the question “Is this note helpful?” Answers to that question are then used to rank notes. When Birdwatch launched in January 2021, people could answer “yes” or “no” to that question. An update on June 30, 2021 allows people to choose between “yes,” “somewhat” and “no.” We map these responses to continuous values from 0.0 to 1.0, hereafter referred to as “helpful scores”:

- `Yes` maps to `1.0`
- `Somewhat` maps to `0.5`.
- `No` maps to `0.0`.
- `Yes` from the original 2-option version of the rating form maps to `1.0`.
- `No` from the original 2-option version of the rating form maps to `0.0`.

Specific values in the mapping may change in the future, and will be updated here if so.

## Matrix Factorization

The main technique we use to determine which notes are helpful or unhelpful is matrix factorization on the note-rater matrix, a sparse matrix that encodes, for each note and rater, whether that rater found the note to be helpful or unhelpful. This approach, originally made famous by Funk in the 2006 Netflix prize recommender system competition, seeks a latent representation (embedding) of users and items which can explain the affinity of certain users for certain items [[1](https://sifter.org/~simon/journal/20061211.html)] [[2](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)]. In our application, this representation space identifies whether notes may appeal to raters with specific viewpoints, and as a result we are able to identify notes with broad appeal across viewpoints.

One challenge is that not all raters evaluate all notes - in fact most raters do not rate most notes - and this sparsity leads to outliers and noise in the data. Regularization techniques are a common solution to these issues; a key distinction in our approach is that we use much higher regularization on the intercept terms, which capture the helpfulness of a note or rater that is not explained by viewpoint agreement, relative to the embedding factors. This encourages a representation that uses user and note embeddings to explain as much variation in the ratings as possible before fitting additional note- and user-specific intercepts. As a result, for a note to achieve a high intercept term (which is the note’s helpfulness score), it must be rated helpful by raters with a diversity of viewpoints (factor embeddings). Notes are given a single global score-- their intercept term-- rather than using this algorithm in the traditional way to personalize content as in a recommender system.

We predict each rating as:

\\[\hat{r}\_{un} = \mu + i_u + i_n + f_u \cdot f_n\\]

Where the prediction is the sum of three intercept terms: mu is the global intercept term, i_u is the user’s intercept term, and i_n is the note’s intercept term, added to the dot product of the user and notes’ factor vectors f_u and f_n (note that when user and note factors are close, a user is expected to give a higher rating to the note).

To fit the model parameters, we minimize the following regularized least squared error loss function via gradient descent over the dataset of all observed ratings r_un:

\\[\sum_{r_{un}} (r_{un} - \hat{r}_{un})^2 + \lambda_i (i_u^2 + i_n^2 + \mu^2) + \lambda_f (||f_u||^2 + ||f_n||^2)\\]

Where lambda_i (0.03), the regularization on the intercept terms, is currently 5 times higher than lambda_f (0.15), the regularization on the factors.

The resulting scores that we use for each note are the note intercept terms i_n. These scores on our current data give an approximately Normal distribution, where notes with the highest and lowest intercepts tend to have factors closer to zero.

We currently set the thresholds to achieve a “Currently Rated Helpful” label at 0.40, including less than 10% of the notes, and our threshold to achieve a “Currently Rated Not Helpful” label at –0.08. However, these are far from set in stone and the way we generate status labels from note scores will evolve over time.

This approach has a few nice properties:

- Extra regularization on the intercept terms in practice requires that notes are rated by raters with diverse factors before a note gets a label (the note intercept term becomes very large or very small)
- We can represent multidimensional viewpoint spaces by increasing the dimensionality of the factors, without changing the algorithm itself
- Rater-specific intercept terms capture how lenient or generous each rater is with their helpful ratings
- We are able to include somewhat helpful ratings naturally as 0.5s

Note: for now, to avoid overfitting on our very small dataset, we only use 1-dimensional factors. We expect to increase this dimensionality as our dataset size grows significantly.

### Determining Note Status Explanation Tags

When notes are labeled with a status (either Currently Rated Helpful or Currently Rated Not Helpful), we display the top two explanation tags that were given by raters to explain why they rated the note helpful or not.

This is done by simply counting the number of times each explanation tag was given filtered to explanation tags that match the final note status label (e.g., if the note status is Currently Rated Helpful we only count helpful explanation tags). Importantly, each explanation tag must be used by at least two different raters. If there aren’t two different tags that are each used by two different raters, then the note’s status label is reverted back to “Needs More Ratings” (this is very rare).

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

### Complete Algorithm Steps:

1. <div>Pre-filter the data: to address sparsity issues, only raters with at least 10 ratings and notes with at least 5 ratings are included (although we don’t recursively filter until convergence)</div>
2. <div>Fit matrix factorization model, then assign intermediate note status labels for notes whose intercept terms (scores) are above or below thresholds.</div>
3. <div>Compute Author and Rater Helpfulness Scores based on the results of the first matrix factorization, then filter out raters with low helpfulness scores from the ratings data as described in <a href="../contributor-scores/#filtering-ratings-based-on-helpfulness-scores">Filtering Ratings Based on Helpfulness Scores</a></div>
4. <div>Re-fit the matrix factorization model on the ratings data that’s been filtered further in step 3</div>
5. <div>Assign final note status labels to notes based on whether their intercept terms (scores) are above or below thresholds</div>

6. <div>Assign the top two explanation tags that match the note’s final status label as in <a href="./#determining-note-status-explanation-tags">Determining Note Status Explanation Tags</a>, or if two such tags don’t exist, then revert the note status label to “Needs More Ratings”.</div>

<br/>

## What’s New?

**March 09, 2022**

- Temporarily paused assigning statuses to notes that indicate the Tweet is “not misleading”
- Adjusted thresholds for notes statuses

**Mar 1, 2022**

- Launched entirely new algorithm to compute note statuses (Currently Rated Helpful, Currently Not Rated Helpful), which looks for agreement across different viewpoints using a matrix factorization method. Updates contributor helpfulness scores to reflect helpfulness so that contributors whose contributions are helpful to people from a wide range of viewpoints earn higher scores. Uses helpfulness scores to identify a subset of contributor ratings to include in a final round of note scoring. This entirely replaces the previous algorithm which weighted ratings by raters’ helpfulness scores.

**June 30, 2021**

- Added a new “somewhat” option to the “is this note helpful?” question in the rating form.

**June 14, 2021**

- New algorithm to compute note status, which weights ratings by raters’ helpfulness scores, rather than just taking a direct average of ratings.

**January 28, 2021**

- Birdwatch pilot begins, with note ranking done via simply computing the raw ratio of helpful ratings.
