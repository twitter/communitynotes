---
title: Challenges
geekdocBreadcrumb: false
aliases: ["/challenges", "/risks", "/about/challenges"]
---

We know there are many challenges involved in building an open, participatory system like Birdwatch — from making it resistant to manipulation attempts, to ensuring it isn’t dominated by a simple majority or biased because of the distribution of contributors.

We've been building Birdwatch in [public since January 2021](https://blog.twitter.com/en_us/topics/product/2021/introducing-birdwatch-a-community-based-approach-to-misinformation), and have intentionally designed it to mitigate potential risks. We've seen [encouraging results](https://blog.twitter.com/en_us/topics/product/2022/birdwatch-getting-new-onboarding-process-more-visible-notes), but we're constantly designing for challenges that could arise.

Here are a handful of particular challenges we are aware of as well as steps we are taking to address them:

### Preventing Coordinated Manipulation Attempts

Attempts at coordinated manipulation represent a crucial risk for open rating systems. We expect such attempts to occur, and for Birdwatch to be effective, it needs to be resistant to them.

Birdwatch currently takes multiple steps to reduce the potential for this type of manipulation:

- First, all Twitter accounts must meet the [eligibility criteria](../signup) to become a Birdwatch contributor. For example, having a unique, verified phone number. These criteria are designed to help prevent the creation of large numbers of fake or sock puppet contributor accounts that could be used for inauthentic rating.
- Second, Birdwatch doesn't work like many engagement-based ranking systems, where popular content gains the most visibility and people can simply coordinate to mass upvote or downvote content they don't like or agree with. Instead, Birdwatch uses a bridging algorithm — for a note to be shown on a Tweet, it needs to be found helpful by people who have tended to [disagree in their past ratings](../diversity).

  [Academic](https://www.belfercenter.org/publication/bridging-based-ranking) [research](https://www.google.com/books/edition/Breaking_the_Social_Media_Prism/ORMCEAAAQBAJ?hl=en&gbpv=0) indicates this sort of bridging based ranking can help to identify content that is healthier and higher quality, and reduce the risk of elevating polarizing content.

- In addition to requiring ratings from a diversity of contributors, Birdwatch has a [reputation system](../contributor-scores/) in which contributors earn helpfulness scores for contributions that people from a wide range of perspectives find helpful.

  Helpfulness scores give more influence to people with a track record of making high-quality contributions to Birdwatch, and lower influence to new accounts that have yet to demonstrate a track record of helpful ratings and contributions.

- Lastly, Birdwatch tracks metrics that alert the team if suspicious activity is detected, and has a set of [guardrails and procedures](../guardrails) to identify if contribution quality falls below set thresholds. This helps Birdwatch to proactively detect potential coordination attempts and impacts to note quality.

### Reflecting Diverse Perspectives, Avoiding Biased Outcomes

Birdwatch will most be effective if the context it produces can be found to be helpful by people of multiple points of view–not just people from one group or another. To work towards this goal, Birdwatch currently takes the following steps:

- First, as described above, Birdwatch uses a [bridging based algorithm](../note-ranking) to identify notes that are likely to be helpful to people from many points of view. This helps to prevent one-sided ratings and to prevent a single group from being able to engage in mass voting to determine what notes are shown.

- Second, Birdwatch can proactively seek ratings from contributors who are likely to provide a different perspective based on their rating history. This is currently done in the [Needs Your Help tab](../rating-notes), and we are exploring new ways to quickly collect ratings on notes from a wide range of contributors.

- Third, to help ensure that people of diverse backgrounds and viewpoints feel safe and empowered to participate, Birdwatch has implemented program [aliases](../aliases) that aren’t publicly associated with contributors’ Twitter accounts. This can help prevent one-sided-ness by providing more diverse contributors with a voice in the system.

- Finally, we regularly survey representative samples of Twitter customers who are not Birdwatch contributors to assess whether a broad range of people on Twitter are likely to find the context in Birdwatch notes to be helpful, and whether the notes can be informative to people of different points of view.

  This is one indicator of Birdwatch’s ability to be of value to people from a [wide range of perspectives](../diversity/) vs. to be biased towards one group or viewpoint. Twitter customers who aren’t enrolled Birdwatch contributors can also provide rating feedback on notes they see on Twitter. This provides an additional indicator of note helpfulness observed over time.

### Avoiding Harassment

It’s crucial that people feel safe contributing to Birdwatch and aren’t harassed for their contributions. It’s also important that Birdwatch itself does not become a vector for harassment. Here are some measures Birdwatch takes to keep everyone safe:

- First, as described above, all contributors get a new, auto-generated [display name (or alias)](../aliases) when they join Birdwatch. These aliases are not publicly associated with contributors’ Twitter accounts, so everyone can write and rate notes privately. This inhibits public identification and harassment of contributors.

- Second, contributors have an open communication line with the Birdwatch team to report any issues they experience (they can reach us by DM [@Birdwatch](https://twitter.com/birdwatch)). Birdwatch has a dedicated community manager who gathers and responds to contributors’ feedback or concerns via this handle. This provides a way for contributors to flag potential issues to the Birdwatch team.

- Finally, all Birdwatch contributions are subject to Twitter [Rules](https://help.twitter.com/rules-and-policies/twitter-rules), [Terms of Service](https://twitter.com/tos) and [Privacy Policy](https://twitter.com/privacy). If you think a Birdwatch note might not abide by the rules, you can report it by clicking or tapping the ••• menu on a note, and then selecting "Report”, or by using the [Report a Birdwatch Note](https://help.twitter.com/en/forms/birdwatch) form. This provides a mechanism to address violating content in Birdwatch notes.

### Reducing the impacts of low quality contributions on the system

As Birdwatch expands and grows, there's an increasing need to mitigate contributor rater burden and ranking system impacts that can be caused by high volumes of low quality notes or ratings. This could occur as more well-intentioned new contributors onboard to Birdwatch and go through a period of learning how to write and identify helpful notes, or through bad faith attempts to overwhelm Birdwatch with spammy, low quality notes or ratings.

Birdwatch currently takes the following steps to reduce impact of low quality contributions:

- First, Birdwatch has implemented an [onboarding system](../writing-ability), which requires new contributors to begin by rating notes before they unlock the ability to write new notes. This helps to ensure new contributors have ample opportunity to understand what makes notes Helpful and Not Helpful before they begin writing.
- Second, as a part of the system described above, contributors can also have their note writing ability temporarily locked if they write too many notes that reach a status of Not Helpful after being rated by other Birdwatch contributors.

  Contributors receive warnings when they are at risk of having note writing ability locked, as well as after it is locked, to help them understand how to improve. To unlock the ability to write again, they must repeat the process of successfully rating and identifying helpful notes. This can help well-intentioned contributors learn to improve their note writing, and prevents bad actors from continuing to write high volumes of Not Helpful notes over time.

- Finally, to reduce impacts of spammy or low quality ratings, Birdwatch has a note rating reputation system (described above in [preventing coordinated manipulation](#preventing-coordinated-manipulation-attempts)). This gives lower influence to new accounts that have yet to demonstrate a track record of helpful ratings and contributions, making it more difficult for an account to join and spam the system with low quality ratings.

As Birdwatch grows and evolves, we will continue to iterate to ensure it remains a place for healthy contributions, where our contributors always feel safe participating.

## Feedback? Ideas?

We welcome feedback on these or additional risks and challenges, as well as ideas for addressing them. Please DM us at [@Birdwatch](http://twitter.com/birdwatch).
