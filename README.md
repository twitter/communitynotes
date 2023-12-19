# Community Notes

![](/documentation/images/help-rate-this-note-expanded.png)

## Welcome to Community Notes's public repository

This repository is a place for us to transparently host our content, algorithms, and share updates about the program.

The folder `/sourcecode` holds the [open-source code](https://github.com/twitter/communitynotes/tree/main/sourcecode) powering Community Notes under the hood.

The folder `/documentation` holds the [Markdown content](https://github.com/twitter/communitynotes/tree/main/documentation) that is used to generate our [documentation website](https://communitynotes.x.com/guide).

Here you can also find our [research paper](https://github.com/twitter/communitynotes/blob/main/birdwatch_paper_2022_10_27.pdf).

## About Community Notes

Community Notes aims to create a better informed world, by empowering people on X to add helpful notes to posts that might be misleading.

We're building it in the open, with the public’s input, and we’re taking significant steps to make Community Notes transparent.

## Sign up and become a Community Notes contributor

Our goal is to expand Community Notes globally. We want anyone to be able to participate and know that having contributors with different points of view is essential to Community Notes helpfully addressing misinformation.

As there are important nuances in each market, we’ll expand the contributor base country-by-country. We’ll add contributors from a first new country soon.

[Sign up here](https://x.com/i/flow/join-birdwatch)

## How to contribute to this repo

Thank you for your interest in contributing to Community Notes! Currently, we will consider pull requests that contribute to the following areas:
* Downstream analyses of scoring model output
* Alternate scoring algorithm ideas (outside the core algorithm)
* Documentation
* Open issues

Note that we aren’t currently accepting changes that alter existing APIs, as there is other utility and production infrastructure code at X that depends on these APIs remaining stable.

We are also exploring ways to make it easier for people to contribute directly to the core algorithm. For example, by making available testing and evaluation frameworks that would allow open source contributors to evaluate the impact of their PRs on note quality.

---

### Documentation website

The markdown files in this repo are the source of truth for the content in our documentation website (aka "Community Notes Guide"). They are always updated here first, then ingested by X's internal tools, translated, and published in [communitynotes.x.com/guide](https://communitynotes.x.com/guide).

### Community Notes open-source code

The algorithm that powers Community Notes can be found in the [sourcecode folder](https://github.com/twitter/communitynotes/tree/main/sourcecode), and instructions on how to use it can be found in the [Guide](https://twitter.github.io/communitynotes/note-ranking-code/).

While your normal Python install may "just work" if you're lucky, if you run into any issues and want to install the exact versions of Python packages that we've tested the code with, please create a new virtual environment and install the packages from requirements.txt:

```
$ python -m venv communitynotes_env
$ source communitynotes_env/bin/activate
$ pip install -r requirements.txt
```

Then after downloading the data files (see next section) into /sourcecode/, you will be able to run:

```
$ cd sourcecode
$ python main.py
```

Most versions of Python3 should work, but we have tested the code with Python 3.7.9.

### Community Notes data

All notes, ratings, and contributor data are [publicly available and published daily here](https://x.com/i/communitynotes/download-data). Instructions on how to use them can be found in the [Community Notes Guide](https://communitynotes.x.com/guide/under-the-hood/download-data/).

### Community Notes paper

We've published a paper that details the research, analysis and outcomes that informed the program's development and helped us understand its performance. You can find it [here](https://github.com/twitter/communitynotes/blob/main/birdwatch_paper_2022_10_27.pdf).
