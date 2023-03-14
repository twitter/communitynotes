---
title: Note ranking algorithm
aliases:
  [
    "/ranking-notes",
    "/note-ranking",
    "/about/note-ranking",
    "/about/ranking-notes",
  ]
geekdocBreadcrumb: false
geekdocToc: 1
enableMathJax: true
description: How are Community Notes ranked? Learn more about our open-source algorithm
---

The algorithm used to rank Community Notes and compute their statuses is open-source, so anyone can help us identify bugs, biases, and opportunities for improvement. This page describes in detail how that algorithm works and how we've improved it over time. The algorithm source code can be [found here](https://github.com/twitter/communitynotes/tree/main/sourcecode).

{{< toc >}}

## Note Status

![Three Community notes in different statuses](../images/note-statuses.png)

Community Notes are submitted and rated by contributors. Ratings are used to determine note statuses (“Helpful”, “Not Helpful”, or “Needs More Ratings”). Note statuses determine which notes are displayed on each of the [Community Notes Site’s timelines](../birdwatch-timelines/), and which notes are displayed [on Tweets](../notes-on-twitter/).

All Community Notes start with the Needs More Ratings status until they receive at least 5 total ratings.
Notes with 5 or more ratings may be assigned a status of Helpful or Not Helpful according to the algorithm described below.
If a note is deleted, the algorithm will still score it (using all non-deleted ratings of that note) and the note will receive a status if it’s been rated more than 5 times, although since it is deleted it will not be shown on Twitter even if its status is Helpful.

Notes marking Tweets as "potentially misleading" with a Note Helpfulness Score of 0.40 and above earn the status of Helpful. At this time, only notes that indicate a Tweet is “potentially misleading” and earn the status of Helpful are eligible to be displayed on Tweets.
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

![Mockup of the rating prompt for rating notes. It reads: ”Is this note helpful?” And presents three options: Yes, Somewhat, or No](../images/helpful-ranking.png)

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

$$ \hat{r}_{un} = \mu + i_u + i_n + f_u \cdot f_n $$

Where the prediction is the sum of three intercept terms: $\mu$ is the global intercept term, $i_u$ is the user’s intercept term, and $i_n$ is the note’s intercept term, added to the dot product of the user and notes’ factor vectors $f_u$ and $f_n$ (note that when user and note factors are close, a user is expected to give a higher rating to the note).

To fit the model parameters, we minimize the following regularized least squared error loss function via gradient descent over the dataset of all observed ratings $r_{un}$:

$$ \sum_{r_{un}} (r_{un} - \hat{r}_{un})^2 + \lambda_i (i_u^2 + i_n^2 + \mu^2) + \lambda_f (||f_u||^2 + ||f_n||^2) $$

Where $\lambda_i=0.15$, the regularization on the intercept terms, is currently 5 times higher than $\lambda_f=0.03$, the regularization on the factors.

The resulting scores that we use for each note are the note intercept terms $i_n$. These scores on our current data give an approximately Normal distribution, where notes with the highest and lowest intercepts tend to have factors closer to zero.

In general, we set the thresholds to achieve a “Helpful” status at 0.40, including less than 10% of the notes, and our threshold to achieve a “Not Helpful” status at $-0.05 - 0.8 \* abs(f_n)$. The [Tag Outlier Filtering](#tag-outlier-filtering) section describes an extension to the general thresholds.

This approach has a few nice properties:

- Extra regularization on the intercept terms in practice requires that notes are rated by raters with diverse factors before a note gets a label (the note intercept term becomes very large or very small)
- We can represent multidimensional viewpoint spaces by increasing the dimensionality of the factors, without changing the algorithm itself
- Rater-specific intercept terms capture how lenient or generous each rater is with their helpful ratings
- We are able to include somewhat helpful ratings naturally as 0.5s

Note: for now, to avoid overfitting on our very small dataset, we only use 1-dimensional factors. We expect to increase this dimensionality as our dataset size grows significantly.

Additionally, because the matrix factorization is re-trained from scratch every hour, we have added additional logic to detect if the loss is more than expected (currently by detecting if the loss is above a hard threshold of 0.09) that may have resulted from an unlucky initialization and local mode, and then re-fit the model if so.

## Tag Outlier Filtering

In some cases, a note may appear helpful but miss key points about the tweet or lack sources.
Reviewers who rate a note as "Not Helpful" can associate [tag](examples#helpful-attributes) with their review to identify specific shortcomings of the note.
When a note has receives high levels of a "Not Helpful" tag, we require a higher intercept before rating the note as "Helpful".
This approach helps us to maintain data quality by recognizing when there is a troubling pattern on an otherwise strong note.

We define the quantity $a_{un}$ to represent the _weight_ given to tag $a$ identified by reviewer (user) $u$ on note $n$:

$$ a_{un} = \frac{\mathbb{1}_{a_{un}}}{ 1 + \left( {{||f_u - f_n||} \over {\tilde{f}}} \right)^5  }  $$

Where:

- $\tilde{f} = \eta_{40}^{r_{un}}(||f_n - f_||)$ indicates the 40th percentile of the distances between the rater (user) and note latent factors over all observable ratings $r_{un}$
- $\mathbb{1}_{a_{un}}$ is 1 if rater $u$ assigned tag $a$ to note $n$ and 0 otherwise.

We define the total weight of an tag $a$ on note $n$ as:

$$ n_{a} = \sum_{r_{un}} a_{un} $$

Notice the following:

- No single rating can achieve an tag weight $a_{un} > 1$.
- Ratings where the rater factor and note factor are equal will achieve the maximum weight of 1.0, ratings at a 40th percentile distance will achieve a weight of 0.5, and reviews at 2x the 40th percentile distance will have a weight of ~0.03. All ratings will have positive weight.
- Assigning higher weights to tags in ratings where the rater and note are closer in the embedding space effectively lends greater weight to critical ratings from raters who tend to share the same perspective as the note.

Given the quantities defined above, we modify scoring as follows:

- When the total weight $a_n$ of an tag exceeds 2.5 _and_ is in the 95th percentile of all notes with an intercept greater than 0.4, we require the intercept to exceed 0.5 before marking the note as helpful.
- We disregard the "Typos or unclear language" and "Note not needed on this Tweet" tags, which do not relate to note accuracy.

## CRH Inertia

The scoring algorithm updates the Helpful status of each note during every invocation.
Re-computing the Helpful status ensures that as new ratings emerge the notes shown to users continue to reflect a broad consensus among raters.
In some cases, small variations in the note intercept $i_n$ can cause notes to lose and potentially re-acquire Helpful status.

To help ensure that changes in Helpful status reflect a clear shift in consensus, we require that the note intercept $i_n$ drops below the applicable threshold by more than 0.01 before the note loses Helpful status.
For example, if a note achieved Helpful status with note intercept $i_n>0.40$, then the note would need $i_n<0.39$ before losing Helpful status.
Similarly, if a note was impacted by tag outlier filter and required note intercedpt $i_n>0.50$ to achieve Helpful status, the note would need $i_n<0.49$ to lose Helpful status.

## Multi-Model Note Ranking

Multi-Model ranking allows Community Notes to run multiple ranking algorithms before reconciling the results to assign final note status.
We use this ability to test new models, refine current approaches and support expanding the Community Notes contributor base.
We currently run three note ranking models:

- The _Core_ model runs the matrix factorization approach described above to determine status for notes with most ratings from geographical areas where Community Notes is well established (e.g. the US, where Community Notes has been available for multiple years).  We refer to established areas as _Core_ areas and areas where Community Notes has recently launched as _Expansion_ areas. The Core model includes ratings from users in Core areas on notes where the majority of ratings also came from users in Core areas.
- The _Expansion_ model runs the same ranking algorithm with the same parameters as the Core model, with the difference that the Expansion model includes all notes with all ratings across Core and Expansion areas.
- The _Coverage_ model runs the same ranking algorithm and processes the same notes and ratings as the Core model, except the intercept regularization $\lambda_i$ and Helpful note threshold have been [tuned differently](https://github.com/twitter/communitynotes/blob/main/sourcecode/scoring/mf_coverage_scorer.py) to increase the number of Helpful notes.

In cases where a note is ranked by both the Core and Expansion models the Core model is always authoritative.
This approach allows us to grow Community Notes as quickly as possible in experimental Expansion areas without the risk of compromising quality in Core areas where Community Notes is well established.
In cases where the Core and Coverage models disagree, a Helpful rating from the Core model always takes precedence.
If a note is only rated as Helpful by the Coverage model, then the note must surpass a safeguard threshold for the Core model intercept to receive a final Helpful rating.
We have initialized the Core model safeguard threshold to 0.38, 0.02 below the Core model default Helpfulness threshold of 0.40, and will lower the safeguard threshold as the Coverage model continues to launch.

When using Twitter, you can see which model computed the status a given note by looking at the Note Details screen.
It might list one of the following models:
- CoreModel (vX.X). The _Core_ model described above.
- ExpansionModel (vX.X). The _Expansion_ model described above.
- CoverageModel (vX.X). The _Coverage_ model described above.
- ScoringDriftGuard. This is a scoring rule that locks note statuses after two weeks. See the [next section](#status-stabilization) for more details.

## Status Stabilization

As Community Notes has scaled from inception to global availability we've seen an increasing number of notes and ratings spanning a widening array of topics.
With the increased volume of community contributions, ranking data for older and newer notes has diverged: newer notes are able to receive more ratings from a wider range of contributors while the available ranking data for older notes remains more limited.
As older data comprise an increasingly small fraction of the dataset, ranking results have tended to fluctuate and some notes have lost Helpful status.

To maintain Helpful note quality as Community Notes continues to grow, we are adding logic which stabilizes the status of a note once the note is two weeks old.
This approach allows us to continue optimizing the ranking algorithm with a focus on the impact on current data while persisting helpful community contributions on older topics.
Before a note is two weeks old, the helpfulness status will continue to be updated each time time the ranking algorithm is run.
After a note turns two weeks old we store the helpfulness status for that note and use the stored status in the future, including for displaying notes on Twitter and calcualting user contribution statistics.

While a note may be scored by the Core, Expansion and Coverage models, we only finalize note status based on the Core model.
Notes that are only ranked by the Expansion model are not eligible for stabilization since the Expansion model is under development and may be revised to improve quality at any time.
Similarly, if a note is rated Helpful by the Coverage model and Needs More Ratings by the Core model, we will allow the note status to remain at Helpful even after the note is two weeks old.
If at any point both models agree and the Core model scores the note as Helpful or the Coverage model scores the note as Needs More Ratings, then the status will be finalized in agreement with both models.

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
5. <div>Reconcile scoring results from the Core, Expansion and Coverage models to generate final status for each note.</div>
6. <div>Update status labels for any notes written within the last two weeks based the intercept terms (scores) and ratings tags.  Stabilize helpfulness status for any notes older than two weeks.</div>
7. <div>Assign the top two explanation tags that match the note’s final status label as in <a href="./#determining-note-status-explanation-tags">Determining Note Status Explanation Tags</a>, or if two such tags don’t exist, then revert the note status label to “Needs More Ratings”.</div>

<br/>

## What’s New?

**March 13, 2023**

- Start labeling notes that were scored by the Expansion model, which has resulted in multiple notes from the expansion countries (e.g. Brazil) becoming CRH and CRNH.
- Update tag weighting formula for tag filtering, to reduce weights for raters further from the note.
- Decrease CRH threshold of the coverage model from 0.39 to 0.38, thereby increasing the coverage of the coverage model.
- Add logic to improve the stability of the optimization between re-trainings, to prevent status flipping when the optimization occasionally gets stuck in a bad local mode.
- Release a new modelingPopulation column in the user enrollment file, which indicates whether a user is included in the core or expansion model.

**February 24, 2023**

- Introduced support for running multiple ranking models.
- Launched ranking support for Community Notes global expansion, partitioning notes and ratings by Core and Expansion to maintain note ranking quality while growing globally.
- Launched Coverage model with increased intercept regularization.  This model will run along-side the Core note ranking model to increase Helpful note coverage.


**January 20, 2023**

- Updated the helpfulness status of all notes to the historical status on August 15, 2022 or two weeks after note creation, whichever was later. Notes created within the last two weeks were unimpacted. We selected August 15, 2022 to include as many scoring improvements as possible while still predating data changes caused by scaling Community Notes.

**January 17, 2023**

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

**October 3, 2022**
- Updated the rating form to better capture the strengths of notes which add context without indicating the Tweet is misleading. We have resumed assigning status to notes marking Tweets as "not misleading" in select circumstances as we evaluate ranking quality and utility to users.

**July 13, 2022**

- To prevent manipulation of helpfulness scores through deletion of notes, notes that are deleted will continue to be assigned note statuses based on the ratings they received. These statuses are factored into author helpfulness scores.
- Valid Ratings Definition Update: instead of just the first 5 ratings on a note, all ratings will be valid if they are within the first 48 hours after note creation and were created before the note first received its status of Helpful or Not Helpful (or if its status flipped between Helpful and Not Helpful, then all ratings will be valid up until that flip occurred).
- To make the above two changes possible, we are releasing a new dataset, note status history, which contains timestamps for when each note received statuses, and the timestamp and hashed participant ID of the author of a note. This data file is being populated now and will be available on the Community Notes [Data Download](https://twitter.com/i/communitynotes/download-data) page beginning Monday July 18, 2022.

**Mar 09, 2022**

- Temporarily paused assigning statuses to notes that indicate the Tweet is “not misleading”. We observed that notes marking a Tweet as "not misleading" were often rated as “Unhelpful - Tweet doesn’t need a note”, so we paused assigning status to notes marking Tweets as "not misleading" pending improvements to the rating form.
- Adjusted thresholds for notes statuses

**Feb 28, 2022**

- Launched new algorithm to compute note statuses (Helpful, Not Helpful), which looks for agreement across different viewpoints using a matrix factorization method. Updates contributor helpfulness scores to reflect helpfulness so that contributors whose contributions are helpful to people from a wide range of viewpoints earn higher scores. Uses helpfulness scores to identify a subset of contributor ratings to include in a final round of note scoring. This replaces the previous algorithm which weighted ratings by raters’ helpfulness scores.

**June 30, 2021**

- Added a new “somewhat” option to the “is this note helpful?” question in the rating form.

**June 14, 2021**

- New algorithm to compute note status, which weights ratings by raters’ helpfulness scores, rather than taking a direct average of ratings.

**January 28, 2021**

- Community Notes (formerly called Birdwatch) pilot begins, with note ranking done via simply computing the raw ratio of helpful ratings.
