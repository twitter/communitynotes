---
title: Evaluation
description: Community Notes tracks a number of quality measures, with a corresponding set of operational guardrails.
navWeight: 6
---
#  Evaluation

Community Notes aims to create a better-informed world, by empowering people on X to collaboratively add helpful context to posts that might be misleading or missing important context.

It’s important that the context that is [elevated to viewers from Community Notes](../contributing/notes-on-twitter.md) be helpful, informative, and accurate.

The [note ranking algorithm](./note-ranking-code.md) is designed to help meet these quality standards, along with a number of [additional safeguards](../about/challenges.md) built over the course of the program’s development. As Community Notes and its approach are novel, we expect there will be challenges and bumps in the road that could impact note quality.

## Guardrails and Circuit Breakers

To identify potential problems, Community Notes tracks a number of quality measures, with a corresponding set of operational “guardrails” and “circuit breaker” thresholds and procedures to respond to issues. We anticipate that we may need to occasionally trigger these mechanisms as we learn and grow. Here’s how they work:

### Quality Measures

Community Notes measures and monitors three top-line metrics to understand the quality of notes and identify potential issues to mitigate along the way. These measures are used for monitoring purposes, and do not impact the note status outcomes or visibility for individual notes. At present, these measures focus on notes that earn the status of Helpful (i.e., those that are [shown to viewers on X](../contributing/notes-on-twitter.md), beyond enrolled contributors).

Evidence of consistent or systematic problems in top rated notes that earn the status of Helpful on any of the following measures can trigger a guardrail or circuit breaker procedure:

1. **Accuracy: Whether notes that earn the status of Helpful contain accurate, high-quality information**

   This is measured via partnerships with [professional reviewers](https://blog.x.com/en_us/topics/company/2021/bringing-more-reliable-context-to-conversations-on-twitter) who provide evaluations of note accuracy. These evaluations provide a tracking measure of how often Helpful notes that are rated as accurate vs. inaccurate by reviewers over time; they don't impact notes’ ratings/status.

   We don't expect perfect consensus among expert reviewers that all notes are accurate, nor do professionals always [agree with one another on accuracy ratings](https://www.science.org/doi/10.1126/sciadv.abf4393). However, this gives us an indicator of whether there might be broad or consistent issues emerging with note accuracy.

2. **Informativeness: Whether notes that earn the status of Helpful help inform people’s understanding of the subject matter in posts**

   This is measured via survey experiments of random samples of users in the US, comparing whether people who see potentially misleading posts with notes tend to have a different view of the post’s main claim vs those who see posts without notes.

   While notes may vary in their ability to inform, we want to ensure that on average notes that earn the status of Helpful are effective at helping people understand the topics in posts (vs having no impact or an adverse effect).

3. **Helpfulness: Whether users beyond the contributor pool tend to find notes that earn the status of Helpful to be helpful**

   This is measured via surveys of random samples of users in the US. The surveys allow these users to give feedback on the helpfulness of Community Notes.

   We don’t expect all notes to be perceived as helpful by all people all the time. Instead, the goal is to ensure that on average notes that earn the status of Helpful are likely to be seen as helpful by a wide range of people from different points of view, and not only be seen as helpful by people from one viewpoint.

In addition to the three top-line metrics listed above, Community Notes monitors additional operational quality metrics, like whether any notes violate X's [Rules](https://help.x.com/rules-and-policies/twitter-rules).

### Thresholds

For each of these measures and metrics, there are thresholds to trigger ‘guardrails’ and ‘circuit breakers’.

**Guardrails** are an alert threshold that indicates to the Community Notes team there may be low or declining quality in notes that earn the status of Helpful. A guardrail alert indicates that the Community Notes team should conduct an investigation to determine causes of the metric change, and identify whether there are systemic issues to be addressed. If an investigation uncovers a systemic or urgent problem, a “circuit breaker” trip can occur.

**Circuit breakers** are an alert threshold that indicates to the Community Notes team that there may be significant emerging issues with notes that earn the status of Helpful. For example: a number of notes with the status of Helpful are evaluated as low Accuracy or identified as having low helpfulness ratings from diverse audiences in a short period. If a circuit breaker is tripped, our policy is to temporarily turn off display of notes for viewers beyond enrolled contributors, then to investigate and to pursue a remediation. A temporary Remediation (see more on this below) may be enacted while a more permanent solution is built.

### Remediations

Community Notes context is intended to be community driven. Notes are written and selected by people on X, for people on X. Except in the case of a Rule violation, X employees do not make decisions about which individual notes do or do not display on X, even in cases when a guardrail or circuit breaker condition is triggered.

Instead, the goal is to build a _system_ that consistently elevates helpful, informative, and accurate information. In case of a serious note quality problem, the policy and operating procedure is to take system-wide actions, fix the problem, then resume normal service.

Examples of system-wide actions Community Notes may take in the case of a guardrail or circuit breaker situation may include:

- Temporarily raising the [threshold](./note-ranking-code.md) required for notes to be publicly visible on a post
- Temporarily pausing the scoring of newly created notes
- Temporarily pausing the display of all notes on posts for viewers outside enrolled Community Notes contributors

These actions allow Community Notes to mitigate potential risk of inaccurate or low quality notes, while the internal team conducts an investigation and builds an appropriate fix, if needed. When selecting which remediation to use, we aim to take the smallest corrective action possible to balance mitigating the risk of showing low quality notes with the risk of failing to show helpful, informative notes on potentially misleading posts.

_Note: Community Notes contributors and notes are subject to X's [Rules](https://help.x.com/rules-and-policies/twitter-rules). Failure to abide by rules can result in removal from the Community Notes program._

## Understanding helpfulness across political viewpoints

The goal of community notes is to provide context that people from different points of view find helpful. In order to identify notes that are broadly found helpful, our algorithm can identify people who frequently agree or disagree with each other, based on their history of past ratings of Community Notes. Notably, however, the algorithm does not make assumptions about the political connotations of these agreements or disagreements, as it lacks data on political viewpoints or the subject matter of each note.

To better understand if notes are found widely helpful or only helpful to people from a given political perspective, we conduct additional analyses. These are solely for comparative purposes and do not influence the algorithm's decisions about each note, but they do provide valuable information about note quality that aids in system development.

Prior to the roll-out of Community Notes in the United States, we employed representative surveys involving users to understand the helpfulness of notes to diverse political viewpoints. In these surveys, participants optionally declared their political leanings before viewing and providing evaluations of posts with and without Community Notes. Details of our methodology and past findings can be found in [our paper](https://github.com/twitter/communitynotes/blob/main/birdwatch_paper_2022_10_27.pdf).

Additionally, we utilize a widely-used technique for viewpoint estimation. This approach makes use of our follow, like, and repost graphs to estimate political leanings based on an account’s  proximity to and interaction with political accounts and content within the network. Our calculations can be approximated by the methods described in [this paper](http://pablobarbera.com/static/barbera_twitter_ideal_points.pdf). Note that calculation of viewpoint estimations are anonymized and used on an aggregate basis only to help evaluate perceived note quality.

## Feedback? Ideas?

We will continue to evolve our approach as we grow the program, conduct additional analyses, and learn. We welcome feedback and ideas to improve. Please DM us at [@CommunityNotes](https://x.com/communitynotes).

## Current Status

Community Notes is operating normally. This page will be updated if Community Notes is operating with a system-wide remediation in place as a result of triggering a guardrail or circuit breaker.
