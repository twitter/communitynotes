---
title: Overview
geekdocBreadcrumb: false
aliases: ["/overview", "/about/overview"]
---

{{< figure src="../images/1-writing-notes.png">}}

<br>

Birdwatch aims to create a better informed world, by empowering people on Twitter to collaboratively add helpful notes to Tweets that might be misleading.
Contributors can identify Tweets they believe are misleading, write notes that provide context to the Tweet, and rate the quality of other contributors’ notes. Through consensus from a broad and diverse set of people, our eventual goal is that the most helpful notes will be visible directly on Tweets, available to everyone on Twitter.

# How it works

In this phase, Birdwatch has three core elements: notes, ratings, and the Birdwatch site.

1. ### Notes

People who have signed up to be Contributors can add notes to any Tweet they think might be misleading. Notes are composed of multiple-choice questions and an open text field where contributors can explain why they believe a Tweet is misleading, as well as link to relevant sources. Birdwatch notes are public, and anyone in the US can browse the Birdwatch site to see them. For those not in the pilot, notes currently appear only on the Birdwatch site. They are intentionally kept separate from Twitter for now, while we build Birdwatch and gain confidence that it produces context people find helpful and appropriate. People in the pilot may see notes directly on Tweets when browsing Twitter, and are able to rate those notes for helpfulness.

[Learn more about writing notes →](../writing-notes/)

2. ### Ratings

Contributors can rate the helpfulness of other's notes. Ratings help identify which notes are most helpful, and allow Birdwatch to raise the visibility of those found most helpful by a wide range of people. Ratings also inform our reputation models that recognize those whose contributions are consistently found helpful by a diverse set of people.

[Learn more about rating notes →](../rating-notes/)

3. ### The Birdwatch site

The Birdwatch site is the home for all Birdwatch notes and ratings, separate from the main Twitter apps, and is available at [birdwatch.twitter.com](https://birdwatch.twitter.com).

During this phase of the pilot, Birdwatch contributions will not affect the way people outside of the pilot see Tweets or our system recommendations. Our priority is to understand how to build and adopt an approach that takes input from a diverse set of contributors and identifies the context that people will find most helpful.

[Visit the Birdwatch Site →](https://birdwatch.twitter.com)

# Transparency and visibility

We believe it’s important for people to understand how Birdwatch works, and to be able to help shape it. To that end, we’re taking significant steps to make Birdwatch transparent.

Even though only Birdwatch contributors and people included in test groups can write and rate notes, anyone in the US can access the Birdwatch site to browse contributions. Additionally, all data contributed to Birdwatch will be publicly available and for [download in `tsv` files](https://twitter.com/i/birdwatch/download-data).

As we develop algorithms that power Birdwatch — such as reputation, consensus, and ranking systems — we also aim to make them publicly available. Our current implementation is already [available here](../note-ranking).

We hope that steps like these will enable experts, researchers, and the public to analyze or audit Birdwatch, identifying opportunities or flaws that can help us more quickly build an effective community-driven solution.
