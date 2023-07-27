---
title: Open-source code
description: Download the open-source files that power Community Notes on X.
navWeight: 2
---
# Open-source code

Here you can find links to code that reproduces the note scoring/ranking code that X runs in production: [code on Github](https://github.com/twitter/communitynotes/tree/main/sourcecode).

If you download the data files made available on the [Data Download](https://x.com/i/communitynotes/download-data) page and put them in the same directory as the following code files, you can then run `python main.py` to produce a `scoredNotes.tsv` file that contains note scores, statuses, and explanation tags that will match what’s running in production (as of the time the data was from).
