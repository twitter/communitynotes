# Twitter Birdwatch

![](/static/images/hero.png)

## Welcome to Birdwatch's public repository

This repository holds the source code and content for our [documentation website](https://twitter.github.io/birdwatch/), the [source code](https://github.com/twitter/birdwatch/tree/main/static/sourcecode) powering Birdwatch under the hood, our [research paper](https://github.com/twitter/birdwatch/blob/main/birdwatch_paper_2022_10_27.pdf), and is a place for us to transparently share updates about the program.

## About Birdwatch

Birdwatch aims to create a better informed world, by empowering people on Twitter to add helpful notes to Tweets that might be misleading.

Currently, Birdwatch is in pilot mode for people in the US. We're building it in the open, with the public’s input, and we’re taking significant steps to make Birdwatch transparent.

## Sign up and become a Birdwatch contributor

Our goal is to expand Birdwatch to the global Twitter community. We want anyone to be able to participate and know that having contributors with different points of view is essential to Birdwatch helpfully addressing misinformation. We’ll draw on learnings from this initial test and, over time, scale safely.

[Sign up here](https://twitter.com/i/flow/join-birdwatch)

## How to contribute to this repo

For this initial phase of the Birdwatch pilot, we’ve turned off public Github contribution tools on GitHub (Issues, Pull Requests, Discussions) while we explore the best ways to engage with the growing community who is interested in studying or contributing to Birdwatch.

For now, this site is intended as a transparent source of information about Birdwatch, including version history of that information. As we develop algorithms that power Birdwatch — such as reputation, consensus, and ranking systems — we also aim to make them publicly available. Our current implementation is already available [here](https://twitter.github.io/birdwatch/ranking-notes).

You can talk directly with the team building Birdwatch on Twitter, at [@Birdwatch](https://twitter.com/birdwatch).

---

### Guide source code

Our static documentation site (called "Birdwatch Guide") is built with [Hugo](https://gohugo.io/), using the [Hugo Geekdoc theme](https://github.com/thegeeklab/hugo-geekdoc). Follow the instructions on the Hugo website for downloading and running Hugo.

### Birdwatch source code

The algorithm that powers Birdwatch can be found on the [sourcecode folder](https://github.com/twitter/birdwatch/tree/main/static/sourcecode), and instructions on how to use it can be found in the [Guide](https://twitter.github.io/birdwatch/note-ranking-code/)

### Birdwatch data

All Birdwatch notes and ratings are [publicly available](https://twitter.com/i/birdwatch/download-data). Instructions on how to use them can be found in the [Birdwatch Guide](https://twitter.github.io/birdwatch/download-data/).

### Birdwatch paper

We've published a paper that details the research, analysis and outcomes that informed Birdwatch's development and helped us understand its performance. You can find it [here](https://github.com/twitter/birdwatch/blob/main/birdwatch_paper_2022_10_27.pdf).
