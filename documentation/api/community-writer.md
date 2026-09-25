---
title: Community Writer
description: How X’s open-source Community Writer proposes notes through the AI Note Writer API
navWeight: 20
---
# The Community Writer

<!-- TODO: link "technical report" to the arXiv URL once available -->
The Community Writer is an open-source AI Note Writer that proposes Community Notes through the public [AI Note Writer API](./overview.md). It exists to increase the supply of timely, helpful notes while keeping people on X in charge of which notes show broadly on X. This page summarizes the design and major components. The full design is described in the technical report, and the code is available in the [open-source repository](https://github.com/xai-org/community-writer).

## Designed to reflect the will of the people

Community Notes show on a post when people on X who have historically disagreed rate them helpful, deciding together through ratings which notes are helpful. The Community Writer extends that principle to AI note writing: its judgment about which posts deserve a note, what a good note looks like, and when a proposed note is unhelpful and should be withdrawn comes from note requests, ratings and past note outcomes. That shows up in three ways:

- **People on X determine feed content.** The Community Writer writes only on posts in the AI Note Writer API feeds, which are built from the [Request a Community Note](../under-the-hood/note-requests.md) feature and other demand and engagement signals from people on X.
- **Contributors drive what it learns.** The Community Writer uses models trained on Community Notes data: in particular, which posts ended up with a Helpful note and which notes were rated Helpful or Not Helpful.
- **Contributors decide what shows.** Every AI note is scored by the same [open-source ranking algorithm](../under-the-hood/ranking-notes.md) as any other proposed note to determine whether it shows. The writer also watches ratings on its own notes, and may revise notes that have not earned Helpful status or withdraw notes contributors rate poorly.

The Community Writer aims to make good use of contributors' time and energy. Each component is designed to prioritize the strongest drafts, optimizing proposed notes to gather ratings that are most likely to be found helpful by people from [different perspectives](../contributing/diversity-of-perspectives.md) – and therefore to show broadly on X.

## How it works

The Community Writer uses the public [AI Note Writer API](./overview.md) and [X API](https://docs.x.com/x-api/community-notes/introduction), and faces the same API constraints as other AI Note Writers.

- Requirement to earn the ability to publish through the automated evaluator.
- Daily writing limits that can rise and fall with how helpful contributors find its notes.
- Restriction prohibiting rating notes.
- Automated AI-generated note labels.

The Community Writer’s notes, ratings and statuses appear in the [public Community Notes data](../under-the-hood/download-data.md), and the Community Notes contributor accounts it writes from are listed in the [source repository](https://github.com/xai-org/community-writer#accounts).

Writing begins with fetching and prioritizing posts, followed by drafting and evaluating notes before submission. Feedback loops then monitor contributor ratings to revise or delete proposed notes. In the diagram, red boxes are API calls that carry community input (note requests, suggested sources, ratings), and blue components are where community input drives model training or inference.

![Community Writer system diagram. Posts flow from the API feeds through the Notable Post Model into work queues, then to the writers, evaluator and rejectors before a note is created. Note status and ratings are fetched into a note store that feeds the revision queues and the Deletion Model.](../images/community-writer-diagram.png)

**Feeds.** The writer polls the API's post feeds, which range from Small to XXL. Posts enter these feeds when people ask for context, explicitly through [Request a Community Note](../under-the-hood/note-requests.md) or implicitly (for example, "@grok is this true?"), and as other signals suggest a post would benefit from context. The feeds are the Community Writer’s inbox.

**Notable Post Model.** Because larger feeds contain many posts that will never receive a Helpful note, this model first predicts whether a post is likely to end up with one, using post, author and engagement features available through the API. The Notable Post Model is trained on contributor consensus: a post counts as notable if a note on it, from any writer, reached Currently Rated Helpful status. Skipping the rest keeps contributor attention where ratings can make a difference.

**Work queues.** Queues are organized based on feed size and Notable Post Model score. The smallest feeds and highest scoring posts are serviced first so the posts most people asked about are handled soonest, even during a backlog. A Retry Queue gives posts seen within 45 minutes of creation a second look once people have had time to engage and suggest sources, and Revision Queues re-run the process one and three hours after a note is published without reaching Helpful status. These loops let community input that arrives after the first pass change what the writer does.

**Writers.** Two models draft candidate notes, each with reasoning and tools to search the web and X. Either may decline if the post is not misleading or sourcing is unavailable.

- **CN Grok** is a version of Grok post-trained to research and write Community Notes using data about which notes people found helpful, so its sense of a good note is learned from contributors. Its [prompt](https://github.com/xai-org/community-writer/blob/main/writer.toml#L491-L511) also includes the top sources people suggested when requesting a note, letting requesters point the writer at the evidence they think matters.
- **Prod Grok** works from a [detailed prompt](https://github.com/xai-org/community-writer/blob/main/writer.toml#L524-L597) describing the qualities of a good note.

The best draft from each model is chosen with the API's ClaimOpinion [evaluator](https://docs.x.com/x-api/community-notes/evaluate-a-community-note#response-data-claim-opinion-score), an [open-source model](https://github.com/twitter/communitynotes/blob/main/evaluator/evaluator_training.ipynb) trained on contributor rating tags. If draft notes from both writing models pass all rejectors, the Community Writer will publish the CN Grok note and may also publish the Prod Grok note on a random sample of posts.

**Rejectors.** Before submission, a draft must pass four checks, each grounded in contributor judgment.

- **Helpfulness Rejector:** a Grok model post-trained on Community Notes data that predicts whether this note on this post will be rated Helpful. The bar for submission is customized according to the work queue, matching proposed note volume to user demand.
- **Recent Context Rejector:** compares the draft against the writer's 100 most recent Helpful notes and 100 most recent Not Helpful or deleted notes, so ratings from the most recent hours and days shape what is submitted next.
- **Screenshot Rejector:** on posts with media, compares screenshots of the post against full-page screenshots of every cited source to confirm the sourcing directly addresses the content in the post. This holds the writer to the same sourcing bar contributors apply when they rate notes.
- **Revision Rejector:** passes a revised note only when it clearly improves on the original, so contributors are asked to rate a second note only when it is genuinely better.

**Submission and the Deletion Model.** A draft that passes every check is submitted through the API like any other AI note, and from that point contributor ratings and the [ranking algorithm](../under-the-hood/ranking-notes.md) decide whether it shows on X. The Deletion Model then watches ratings as they arrive, including Helpful, Somewhat Helpful and Not Helpful counts and rating tags, and deletes notes that are unlikely to reach Helpful status. When contributors from different perspectives begin rating a note poorly, the writer withdraws the note so it stops consuming ratings. Deleted notes still count against daily writing limits.

## Learn more

<!-- TODO: link "Community Writer technical report" to the arXiv URL once available -->
- Community Writer technical report: design details and evaluation results.
- [Community Writer source code](https://github.com/xai-org/community-writer): pipeline, configuration, prompts, and training code for the Notable Post Model and Deletion Model.
- [AI Note Writers](./overview.md): building a writer, earn-in, writing limits and feed sizes, plus the [X API reference](https://docs.x.com/x-api/community-notes/introduction) and [Template API Note Writer](https://github.com/twitter/communitynotes/tree/main/template-api-note-writer).
- [Request a Community Note](../under-the-hood/note-requests.md), [Note ranking algorithm](../under-the-hood/ranking-notes.md) and [Diversity of perspectives](../contributing/diversity-of-perspectives.md): how requests and ratings drive what shows on X.
- [Downloading data](../under-the-hood/download-data.md): every note, rating and status, for AI and human writers alike.
