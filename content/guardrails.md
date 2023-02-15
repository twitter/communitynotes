---
title: Guardrails and Circuit Breakers
geekdocBreadcrumb: false
aliases: ["/guardrails"]
description: Community Notes tracks a number of quality measures, with a corresponding set of operational guardrails.
---

Community Notes aims to create a better-informed world, by empowering people on Twitter to collaboratively add helpful context to Tweets that might be misleading or missing important context.

It’s important that the context that is [elevated to viewers from Community Notes](../notes-on-twitter) be helpful, informative, and accurate.

The [note ranking algorithm](../note-ranking) is designed to help meet these quality standards, along with a number of [additional safeguards](../challenges) built over the course of the program’s development. As Community Notes and its approach are novel, we expect there will be challenges and bumps in the road that could impact note quality.

To identify potential problems, Community Notes tracks a number of quality measures, with a corresponding set of operational “guardrails” and “circuit breaker” thresholds and procedures to respond to issues. We anticipate that we may need to occasionally trigger these mechanisms as we learn and grow. Here’s how they work:

## Quality Measures

Community Notes measures and monitors three top-line metrics to understand the quality of notes and identify potential issues to mitigate along the way. These measures are used for monitoring purposes, and do not impact the note status outcomes or visibility for individual notes. At present, these measures focus on notes that earn the status of Helpful (i.e., those that are [shown to viewers on Twitter](../notes-on-twitter), beyond enrolled contributors).

Evidence of consistent or systematic problems in top rated notes that earn the status of Helpful on any of the following measures can trigger a guardrail or circuit breaker procedure:

1. <div> <strong>Accuracy: Whether notes that earn the status of Helpful contain accurate, high-quality information</strong><label>
   <br/>This is measured via partnerships with <a href="https://blog.twitter.com/en_us/topics/company/2021/bringing-more-reliable-context-to-conversations-on-twitter" target="_blank">professional reviewers</a> who provide evaluations of note accuracy. These evaluations provide a tracking measure of how often Helpful notes that are rated as accurate vs. inaccurate by reviewers over time; they don't impact notes’ ratings/status. <br/> <br/>We don't expect perfect consensus among expert reviewers that all notes are accurate, nor do professionals always <a href="https://www.science.org/doi/10.1126/sciadv.abf4393" target="_blank">agree with one another on accuracy ratings</a>. However, this gives us an indicator of whether there might be broad or consistent issues emerging with note accuracy.</label></div>

2. <div> <strong>Informativeness: Whether notes that earn the status of Helpful help inform people’s understanding of the subject matter in Tweets</strong><label>
   <br/> This is measured via survey experiments of random samples of Twitter users in the US, comparing whether people who see potentially misleading Tweets with notes tend to have a different view of the Tweet’s main claim vs those who see Tweets without notes. </br><br/> While notes may vary in their ability to inform, we want to ensure that on average notes that earn the status of Helpful are effective at helping people understand the topics in Tweets (vs having no impact or an adverse effect).</label></div>

3. <div> <strong>Helpfulness: Whether Twitter users beyond the contributor pool tend to find notes that earn the status of Helpful to be helpful</strong><label>
   <br/> This is measured via surveys of random samples of Twitter users in the US. The surveys allow these Twitter users to give feedback on the helpfulness of Community Notes.</br><br/> We don’t expect all notes to be perceived as helpful by all people all the time. Instead, the goal is to ensure that on average notes that earn the status of Helpful are likely to be seen as helpful by a wide range of people from different points of view, and not only be seen as helpful by people from one viewpoint.</label></div>

In addition to the three top-line metrics listed above, Community Notes monitors additional operational quality metrics, like whether any notes violate Twitter Rules.

## Thresholds

For each of these measures and metrics, there are thresholds to trigger ‘guardrails’ and ‘circuit breakers’.

**Guardrails** are an alert threshold that indicates to the Community Notes team there may be low or declining quality in notes that earn the status of Helpful. A guardrail alert indicates that the Community Notes team should conduct an investigation to determine causes of the metric change, and identify whether there are systemic issues to be addressed. If an investigation uncovers a systemic or urgent problem, a “circuit breaker” trip can occur.

**Circuit breakers** are an alert threshold that indicates to the Community Notes team that there may be significant emerging issues with notes that earn the status of Helpful. For example: a number of notes with the status of Helpful are evaluated as low Accuracy or identified as having low helpfulness ratings from diverse audiences in a short period. If a circuit breaker is tripped, our policy is to temporarily turn off display of notes for viewers beyond enrolled contributors, then to investigate and to pursue a remediation. A temporary Remediation (see more on this below) may be enacted while a more permanent solution is built.

## Remediations

Community Notes context is intended to be community driven. Notes are written and selected by people on Twitter, for people on Twitter. Except in the case of a Twitter Rule violation, Twitter employees do not make decisions about which individual notes do or do not display on Twitter, even in cases when a guardrail or circuit breaker condition is triggered.

Instead, the goal is to build a _system_ that consistently elevates helpful, informative, and accurate information. In case of a serious note quality problem, the policy and operating procedure is to take system-wide actions, fix the problem, then resume normal service.

Examples of system-wide actions Community Notes may take in the case of a guardrail or circuit breaker situation may include:

- Temporarily raising the [threshold](../note-ranking) required for notes to be publicly visible on a Tweet
- Temporarily pausing the scoring of newly created notes
- Temporarily pausing the display of all notes on Tweets for viewers outside enrolled Community Notes contributors

These actions allow Community Notes to mitigate potential risk of inaccurate or low quality notes, while the internal team conducts an investigation and builds an appropriate fix, if needed. When selecting which remediation to use, we aim to take the smallest corrective action possible to balance mitigating the risk of showing low quality notes with the risk of failing to show helpful, informative notes on potentially misleading Tweets.

_Note: Community Notes contributors and notes are subject to the Twitter Rules. Failure to abide by rules can result in removal from the Community Notes program._

## Feedback? Ideas?

We will continue to evolve our approach as we grow the program, conduct additional analyses, and learn. We welcome feedback and ideas to improve. Please DM us at [@CommunityNotes](https://twitter.com/communitynotes).

## Current Status

Community Notes is operating normally. This page will be updated if Community Notes is operating with a system-wide remediation in place as a result of triggering a guardrail or circuit breaker.
