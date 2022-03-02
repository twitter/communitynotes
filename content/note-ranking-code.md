---
title: Note ranking code
aliases: ["/note-ranking-code", "/ranking-code", "/code"]
geekdocBreadcrumb: false
geekdocToc: 1
---

Here you can find links to code that reproduces the note scoring/ranking code that Birdwatch runs in production.

If you download the notes and ratings files made available on the [Birdwatch Data Download](https://twitter.com/i/birdwatch/download-data) page and put them in the same directory as the following files, you can then run `python score_notes.py` to produce a `scoredNotes.tsv` file that contains note scores, statuses, and explanation tags that will match what’s running in production.

- [constants.py](../sourcecode/constants.py)
- [explanation_tags.py](../sourcecode/explanation_tags.py)
- [helpfulness_scores.py](../sourcecode/helpfulness_scores.py)
- [matrix_factorization.py](../sourcecode/matrix_factorization.py)
- [process_data](../sourcecode/process_data.py)
- [score_notes.py](../sourcecode/score_notes.py)

You can also see this code on [Github](https://github.com/twitter/birdwatch/tree/main/static/sourcecode).
