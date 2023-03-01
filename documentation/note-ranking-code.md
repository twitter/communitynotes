---
title: Open-source code
aliases: ["/note-ranking-code", "/ranking-code", "/code"]
geekdocBreadcrumb: false
geekdocToc: 1
description: Download the open-source files that power Community Notes on Twitter.
---

Here you can find links to code that reproduces the note scoring/ranking code that Twitter runs in production.

If you download the data files made available on the [Data Download](https://twitter.com/i/communitynotes/download-data) page and put them in the same directory as the following code files, you can then run `python main.py` to produce a `scoredNotes.tsv` file that contains note scores, statuses, and explanation tags that will match what’s running in production (as of the time the data was from).

- [main.py](../sourcecode/main.py)
- [algorithm.py](../sourcecode/algorithm.py)
- [constants.py](../sourcecode/constants.py)
- [explanation_tags.py](../sourcecode/explanation_tags.py)
- [helpfulness_scores.py](../sourcecode/helpfulness_scores.py)
- [matrix_factorization.py](../sourcecode/matrix_factorization.py)
- [process_data](../sourcecode/process_data.py)

You can also see this code on [Github](https://github.com/twitter/communitynotes/tree/main/static/sourcecode).
