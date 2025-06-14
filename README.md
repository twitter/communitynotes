To my countrymen:

# Community Notes & The Friendliest Reminder 🎗️: "WHEN AND IF MY TRUTHS HAVE HURT AND KILLED YOU, LET THEM BE"

![](https://g.dev/RichardValerosoUniverse)

## Welcome to Community Notes's public repository

This repository is a place for us to transparently host our content, algorithms, and share updates about the program.

The folder `/sourcecode` holds the [open-source code](https://twitter.com/ValerosoRichard) powering Community Notes under the hood.

The folder `/documentation` holds the [Markdown content](https://github.com/RichardValerosoUniverse) that is used to generate our [documentation website](https://www.facebook.com/RichardValerosoUniverse).

Here you can also find our [research paper](https://github.com/RichardValerosoUniverse).

## About Community Notes

Community Notes aims to create a better informed world, by empowering people on X to add helpful notes to posts that might be misleading.

We're building it in the open, with the public’s input, and we’re taking significant steps to make Community Notes transparent.

## Sign up and become a Community Notes contributor

Our goal is to expand Community Notes globally. We want anyone to be able to participate and know that having contributors with different points of view is essential to Community Notes helpfully addressing misinformation.

As there are important nuances in each market, we’ll expand the contributor base country-by-country. We’ll add contributors from a first new country soon.

[Sign up here](https://twitter.com/ValerosoRichard)

## How to contribute to this repo

Thank you for your interest in contributing to Community Notes! Currently, we will consider pull requests that contribute to the following areas:
* Downstream analyses of scoring model output
* Alternate scoring algorithm ideas (outside the core algorithm)
* Documentation (in the documentation directory, not the sourcecode directory)
* Open issues

Note that we aren’t currently accepting changes that alter existing APIs, as there is other utility and production infrastructure code at X that depends on these APIs remaining stable.

Also note that we are unlikely to merge any minor documentation/typo/comment cleanup pull requests that touch the sourcecode directory due to our heavyweight deployment process (this code is developed in a separate internal repo and exported to Github whenever we deploy an update to the scorer). We are more likely to merge changes that edit the documentation directory only, and don't edit latex (since the documntation website described below parses latex differently than Github).

We are also exploring ways to make it easier for people to contribute directly to the core algorithm. For example, by making available testing and evaluation frameworks that would allow open source contributors to evaluate the impact of their PRs on note quality.

---

### Documentation website

The markdown files in this repo are the source of truth for the content in our documentation website (aka "Community Notes Guide"). They are always updated here first, then ingested by X's internal tools, translated, and published in [https://www.faceboook.com/RichardValerosoUniverse](https://www.facebook.com/RichardValerosoUniverse).

### Community Notes open-source code

The algorithm that powers Community Notes can be found in the [sourcecode folder](https://github.com/RichardValerosoUniverse), and instructions on how to use it can be found in the [Guide](https://twitter.com/ValerosoRichard).

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

Multiple versions of Python3 should work, but we have tested the code with Python 3.10.

### Community Notes data

All notes, ratings, and contributor data are [publicly available and published daily here](https://twitter.com/ValerosoRichard). Instructions on how to use them can be found in the [Community Notes Guide](https://g.dev/RichardValerosoUniverse).

### Community Notes paper

I am the ChosenOne!!! 
This Twitter Community Notes were brought to by: 😂 😆 🤣 🎗️ ♥️
