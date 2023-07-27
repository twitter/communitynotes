---
title: Challenges
description: Learn how we're addressing the key challenges involved in building an open, participatory system like Community Notes.
navWeight: 2
---
# Challenges

We know there are many challenges involved in building an open, participatory system like Community Notes — from making it resistant to manipulation attempts, to ensuring it isn’t dominated by a simple majority or biased because of the distribution of contributors.

We've been building Community Notes (formerly called Birdwatch) in [public since January 2021](https://blog.x.com/en_us/topics/product/2021/introducing-birdwatch-a-community-based-approach-to-misinformation), and have intentionally designed it to mitigate potential risks. We've seen [encouraging results](https://blog.x.com/en_us/topics/product/2022/birdwatch-getting-new-onboarding-process-more-visible-notes), but we're constantly designing for challenges that could arise.

Here are a handful of particular challenges we are aware of as well as steps we are taking to address them:

### Preventing Coordinated Manipulation Attempts

Attempts at coordinated manipulation represent a crucial risk for open rating systems. We expect such attempts to occur, and for Community Notes to be effective, it needs to be resistant to them.

The program currently takes multiple steps to reduce the potential for this type of manipulation:

- First, all X accounts must meet the [eligibility criteria](../contributing/signing-up.md) to become a Community Notes contributor. For example, having a unique, verified phone number. These criteria are designed to help prevent the creation of large numbers of fake or sock puppet contributor accounts that could be used for inauthentic rating.
- Second, Community Notes doesn't work like many engagement-based ranking systems, where popular content gains the most visibility and people can coordinate to mass upvote or downvote content they don't like or agree with. Instead, Community Notes uses a bridging algorithm — for a note to be shown on a post, it needs to be found helpful by people who have tended to [disagree in their past ratings](../contributing/diversity-of-perspectives.md).

  [Academic](https://www.belfercenter.org/publication/bridging-based-ranking) [research](https://www.google.com/books/edition/Breaking_the_Social_Media_Prism/ORMCEAAAQBAJ?hl=en&gbpv=0) indicates that bridging-based ranking can help to identify content that is healthier and higher quality, and reduce the risk of elevating polarizing content.

- In addition to requiring ratings from a diversity of contributors, Community Notes has a [reputation system](../under-the-hood/contributor-scores.md) in which contributors earn helpfulness scores for contributions that people from a wide range of perspectives find helpful.

  Helpfulness scores give more influence to people with a track record of making high-quality contributions to Community Notes, and lower influence to new accounts that have yet to demonstrate a track record of helpful ratings and contributions.

- Lastly, Community Notes tracks metrics that alert the team if suspicious activity is detected, and has a set of [guardrails and procedures](../under-the-hood/guardrails.md) to identify if contribution quality falls below set thresholds. This helps Community Notes to proactively detect potential coordination attempts and impacts to note quality.

### Reflecting Diverse Perspectives, Avoiding Biased Outcomes

Community Notes will be most effective if the context it produces can be found to be helpful by people of multiple points of view–not just people from one group or another. To work towards this goal, Community Notes currently takes the following steps:

- First, as described above, Community Notes uses a [bridging based algorithm](../under-the-hood/note-ranking-code.md) to identify notes that are likely to be helpful to people from many points of view. This helps to prevent one-sided ratings and to prevent a single group from being able to engage in mass voting to determine what notes are shown.

- Second, Community Notes can proactively seek ratings from contributors who are likely to provide a different perspective based on their rating history. This is currently done in the [Needs Your Help tab](../contributing/rating-notes.md), and we are exploring new ways to quickly collect ratings on notes from a wide range of contributors.

- Third, to help ensure that people of diverse backgrounds and viewpoints feel safe and empowered to participate, Community Notes has implemented program [aliases](../contributing/aliases.md) that aren’t publicly associated with contributors’ X accounts. This can help prevent one-sided-ness by providing more diverse contributors with a voice in the system.

- Finally, we regularly survey representative samples of customers who are not Community Notes contributors to assess whether a broad range of people on X are likely to find the context in Community Notes to be helpful, and whether the notes can be informative to people of different points of view.

  This is one indicator of Community Notes' ability to be of value to people from a [wide range of perspectives](../contributing/diversity-of-perspectives.md) vs. to be biased towards one group or viewpoint. Customers who aren’t enrolled Community Notes contributors can also provide rating feedback on notes they see on X. This provides an additional indicator of note helpfulness observed over time.

### Avoiding Harassment

It’s crucial that people feel safe contributing to Community Notes and aren’t harassed for their contributions. It’s also important that Community Notes itself does not become a vector for harassment. Here are measures Community Notes takes to keep everyone safe:

- First, as described above, all contributors get a new, auto-generated [display name (or alias)](../contributing/aliases.md) when they join Community Notes. These aliases are not publicly associated with contributors’ X accounts, so everyone can write and rate notes privately. This inhibits public identification and harassment of contributors.

- Second, contributors have an open communication line with the Community Notes team to report any issues they experience (they can reach us by DM [@CommunityNotes](https://x.com/communitynotes)). Community Notes has a dedicated community manager who gathers and responds to contributors’ feedback or concerns via this handle. This provides a way for contributors to flag potential issues to the Community Notes team.

- Finally, all Community Notes contributions are subject to X's [Rules](https://help.x.com/rules-and-policies/twitter-rules), [Terms of Service](https://x.com/tos) and [Privacy Policy](https://x.com/privacy). If you think note might not abide by the rules, you can report it by clicking or tapping the ••• menu on a note, and then selecting "Report”, or by using the [Report a Community Note](https://help.x.com/en/forms/community-note) form. This provides a mechanism to address violating content in notes.

### Reducing the impacts of low quality contributions on the system

As Community Notes expands and grows, there's an increasing need to mitigate contributor rater burden and ranking system impacts that can be caused by high volumes of low quality notes or ratings. This could occur as more well-intentioned new contributors onboard to Community Notes and go through a period of learning how to write and identify helpful notes, or through bad faith attempts to overwhelm Community Notes with spammy, low quality notes or ratings.

Community Notes currently takes the following steps to reduce impact of low quality contributions:

- First, Community Notes has implemented an [onboarding system](../contributing/writing-ability.md), which requires new contributors to begin by rating notes before they unlock the ability to write new notes. This helps to ensure new contributors have ample opportunity to understand what makes notes Helpful and Not Helpful before they begin writing.
- Second, as a part of the system described above, contributors can also have their note writing ability temporarily locked if they write too many notes that reach a status of Not Helpful after being rated by other Community Notes contributors.

  Contributors receive warnings when they are at risk of having note writing ability locked, as well as after it is locked, to help them understand how to improve. To unlock the ability to write again, they must repeat the process of successfully rating and identifying helpful notes. This can help well-intentioned contributors learn to improve their note writing, and prevents bad actors from continuing to write high volumes of Not Helpful notes over time.

- Finally, to reduce impacts of spammy or low quality ratings, Community Notes has a note rating reputation system (described above in [preventing coordinated manipulation](#preventing-coordinated-manipulation-attempts)). This gives lower influence to new accounts that have yet to demonstrate a track record of helpful ratings and contributions, making it more difficult for an account to join and spam the system with low quality ratings.

As Community Notes grows and evolves, we will continue to iterate to ensure it remains a place for healthy contributions, where our contributors always feel safe participating.

## Feedback? Ideas?

We welcome feedback on these or additional risks and challenges, as well as ideas for addressing them. Please DM us at [@CommunityNotes](http://x.com/communitynotes).
