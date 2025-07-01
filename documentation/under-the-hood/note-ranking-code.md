---
title: Open-source code
description: Download the open-source files that power Community Notes on X.
navWeight: 2
---
# Open-source code

Here you can find links to code that reproduces the note scoring/ranking code that X runs in production: [code on Github](https://github.com/twitter/communitynotes/tree/main/scoring/src).

If you download the data files made available on the [Data Download](https://x.com/i/communitynotes/download-data) page and put them in the same directory as the following code files, you can then run `python main.py` to produce a `scoredNotes.tsv` file that contains note scores, statuses, and explanation tags that will match what’s running in production (as of the time the data was from).

Note that the algorithm is split into two main binaries: prescoring and final scoring. In production at X, each binary is run separately as often as possible, each always reading the most recent input data available. One subtle implication of this is that in order to exactly reproduce the scoring results as they are run in production at X, the prescorer should be run on input data that's 1 hour older than the final scorer (although this makes very little difference in practice).
