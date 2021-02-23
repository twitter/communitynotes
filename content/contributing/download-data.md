---
title: Downloading data
geekdocBreadcrumb: false
aliases: ["/data"]
---

# Download data

### We can't wait to learn with you!

All Birdwatch contributions are publicly available on the [Download Data](https://twitter.com/i/birdwatch/download-data) page of the Birdwatch site so that anyone in the US has free access to analyze the data, identify problems, and spot opportunities to make Birdwatch better.

If you have questions or feedback about the Birdwatch public data or would like to share your analyses of this data with us, please DM us at [@Birdwatch](http://twitter.com/birdwatch).

<br>

---

# A guide to working with the Birdwatch data

### Data snapshots

The Birdwatch data is released as two separate files: one containing a table representing all Birdwatch notes and one containing a table representing all Birdwatch note ratings. These tables can be joined together on the `noteId` field to create a combined dataset with information about notes and their ratings. The data is released in two separate tables/files to reduce the dataset size by avoiding data duplication (this is known as a normalized data model). Currently, we release one cumulative file each for notes and note ratings. However, in the future, if the data ever grows too large, we will split the data into multiple files as needed.

A new snapshot of the Birdwatch public data is released daily, on a best-effort basis, and technical difficulties may occur and delay the data release until the next day. We are not able to provide guarantees about when this may happen. The snapshot is a cumulative file and contains all non-deleted notes and note ratings ever contributed to Birdwatch, as of 48 hours before the dataset release time. The data download page displays a date stamp indicating the most recent date of data included in the downloadable files.

### File structure

Each data snapshot table is stored in `tsv` (tab-separated values) file format with a header row. This means that each row is separated by a newline, each column is separated by a tab, and the first row contains the column names instead of data. The note and note rating data is directly taken from the user-submitted note creation and note rating forms, with only minimal added metadata (like ids and timestamp). Below, we will describe each column’s data, including the question or source that generated the data, data type, and other relevant information.

### Birdwatch Notes

### `notes-00000.tsv`

| Field                                  | Type   | Descripton                                                                                                                                                                         | Response values                                                                              |
| -------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| noteId                                 | Long   | The unique ID of this note                                                                                                                                                         |                                                                                              |
| participantId                          | String | A Birdwatch-specific user identifier of the user who authored the note. This is a permanent id, which remains stable even if the user changes their username/handle.               |                                                                                              |
| createdAtMillis                        | Long   | Time the note was created, in milliseconds since epoch.                                                                                                                            |                                                                                              |
| tweetId                                | Long   | The tweetId number for the tweet that the note is about.                                                                                                                           |                                                                                              |
| classification                         | String | User-entered multiple choice response to note writing question: “Given current evidence, I believe this tweet is:”                                                                 | "NOT_MISLEADING" "MISINFORMED_OR_POTENTIALLY_MISLEADING"                                     |
| believable                             | String | User-entered multiple choice response to note writing question: “If this tweet were widely spread, its message would likely be believed by:”                                       | "BELIEVABLE_BY_FEW", "BELIEVABLE_BY_MANY"                                                    |
| harmful                                | String | User-entered multiple choice response to note writing question: “If many believed this tweet, it might cause:”                                                                     | "LITTLE_HARM", "CONSIDERABLE_HARM"                                                           |
| validationDifficulty                   | String | User-entered multiple choice response to note writing question: “Finding and understanding the correct information would be:”                                                      | "EASY", "CHALLENGING"                                                                        |
| misleadingOther                        | Int    | User-entered checkbox in response to question “Why do you believe this tweet may be misleading?” (Check all that apply question type).                                             | 1 if “Other” is selected, else 0.                                                            |
| misleadingFactualError                 | Int    | User-entered checkbox in response to question “Why do you believe this tweet may be misleading?” (Check all that apply question type)                                              | 1 if “It contains a factual error” selected, else 0.                                         |
| misleadingManipulatedMedia             | Int    | User-entered checkbox in response to question “Why do you believe this tweet may be misleading?” (Check all that apply question type)                                              | 1 if “It contains a digitally altered photo or video” selected, else 0.                      |
| misleadingOutdatedInformation          | Int    | User-entered checkbox in response to question “Why do you believe this tweet may be misleading?” (Check all that apply question type)                                              | 1 if “It contains outdated information that may be misleading” is selected, else 0.          |
| misleadingMissingImportantContext      | Int    | User-entered checkbox in response to question “Why do you believe this tweet may be misleading?” (Check all that apply question type).                                             | 1 if “It is a misrepresentation or missing important context” is selected, else 0.           |
| misleadingUnverifiedClaimAsFact        | Int    | User-entered checkbox in response to question “Why do you believe this tweet may be misleading?” (Check all that apply question type).                                             | 1 if “It presents an unverified claim as a fact” is selected, else 0.                        |
| misleadingSatire                       | Int    | User-entered checkbox in response to question “Why do you believe this tweet may be misleading?” (Check all that apply question type).                                             | 1 if “It is a joke or satire that might be misinterpreted as a fact” is selected, else 0.    |
| notMisleadingOther                     | Int    | User-entered checkbox in response to question “Why do you believe this tweet is not misleading?” (Check all that apply question type).                                             | 1 if “Other” is selected, else 0.                                                            |
| notMisleadingFactuallyCorrect          | Int    | User-entered checkbox in response to question “Why do you believe this tweet is not misleading?” (Check all that apply question type).                                             | 1 if “It expresses a factually correct claim” is selected, else 0.                           |
| notMisleadingOutdatedButNotWhenWritten | Int    | User-entered checkbox in response to question “Why do you believe this tweet is not misleading?” (Check all that apply question type).                                             | 1 if “This Tweet was correct when written, but is out of date now” is selected, else 0.      |
| notMisleadingClearlySatire             | Int    | User-entered checkbox in response to question “Why do you believe this tweet is not misleading?” (Check all that apply question type).                                             | 1 if “ It is clearly satirical/joking” is selected, else 0.                                  |
| notMisleadingPersonalOpinion           | Int    | User-entered checkbox in response to question “Why do you believe this tweet is not misleading?” (Check all that apply question type).                                             | 1 if “It expresses a personal opinion” is selected, else 0.                                  |
| trustworthySources                     | Int    | Binary indicator, based on user-entered multiple choice in response to note writing question “Did you link to sources you believe most people would consider trustworthy?”         | 1 if “Yes” is selected, 0 if “No” is selected                                                |
| summary                                | String | User-entered text, in response to the note writing prompt “Please explain the evidence behind your choices, to help others who see this tweet understand why it is not misleading” | User entered text explanation, with some characters escaped (e.g. tabs converted to spaces). |

### Birdwatch Ratings

### `ratings-0000.tsv`

| Field                                 | Type   | Descripton                                                                                                                                                             | Response values                                                  |
| ------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| noteId                                | Long   | The unique ID of the note being rated.                                                                                                                                 |                                                                  |
| participantId                         | String | A Birdwatch-specific user identifier of the user who authored the rating. This is a permanent id, which remains stable even if the user changes their username/handle. |                                                                  |
| createdAtMillis                       | Long   | Time the note was created, in milliseconds since epoch.                                                                                                                |                                                                  |
| agree                                 | Int    | Binary indicator, based on user-entered multiple choice in response to note rating question “Do you agree with its conclusion?”                                        | 1 if “Yes” is selected, 0 if “No” is selected                    |
| disagree                              | Int    | Binary indicator, based on user-entered multiple choice in response to note rating question “Do you agree with its conclusion?”                                        | 1 if “No” is selected, 0 if “Yes” is selected                    |
| helpful                               | Int    | Binary indicator, based on user-entered multiple choice in response to note rating question “Is this note helpful?”                                                    | 1 if “Yes” is selected, 0 if “No” is selected                    |
| notHelpful                            | Int    | Binary indicator, based on user-entered multiple choice in response to note rating question “Is this note helpful?”                                                    | 1 if “No” is selected, 0 if “Yes” is selected                    |
| helpfulOther                          | Int    | User-entered checkbox in response to question “What about this note was helpful to you?” (Check all that apply question type).                                         | 1 if “Other” is selected, else 0.                                |
| helpfulInformative                    | Int    | User-entered checkbox in response to question “What about this note was helpful to you?” (Check all that apply question type).                                         | 1 if “Informative” is selected, else 0.                          |
| helpfulClear                          | Int    | User-entered checkbox in response to question “What about this note was helpful to you?” (Check all that apply question type).                                         | 1 if “Clear and/or well-written” is selected, else 0.            |
| helpfulEmpathetic                     | Int    | User-entered checkbox in response to question “What about this note was helpful to you?” (Check all that apply question type).                                         | 1 if “Nonjudgmental and/or empathetic” is selected, else 0.      |
| helpfulGoodSources                    | Int    | User-entered checkbox in response to question “What about this note was helpful to you?” (Check all that apply question type).                                         | 1 if “Cites high-quality sources” is selected, else 0.           |
| helpfulUniqueContext                  | Int    | User-entered checkbox in response to question “What about this note was helpful to you?” (Check all that apply question type).                                         | 1 if “Offers unique information or context” is selected, else 0. |
| notHelpfulOther                       | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type).                                     | 1 if “Other” is selected, else 0.                                |
| notHelpfulIncorrect                   | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type).                                     | 1 if “Incorrect information” is selected, else 0.                |
| notHelpfulSourcesMissingOrUnreliable  | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type).                                     | 1 if “Sources missing or unreliable” is selected, else 0.        |
| NotHelpfulOpinionSpeculationOrBias    | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type).                                     | 1 if “Opinion, speculation, or bias” is selected, else 0.        |
| notHelpfulMissingKeyPoints            | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type).                                     | 1 if “Misses key points” is selected, else 0.                    |
| notHelpfulOutdated                    | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type).                                     | 1 if “Outdated information” is selected, else 0.                 |
| notHelpfulHardToUnderstand            | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type).                                     | 1 if “Hard to understand” is selected, else 0.                   |
| notHelpfulArgumentativeOrInflammatory | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type).                                     | 1 if “Argumentative or inflammatory” is selected, else 0.        |
| notHelpfulOffTopic                    | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type).                                     | 1 if “Off topic” is selected, else 0.                            |
| notHelpfulSpamHarassmentOrAbuse       | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type).                                     | 1 if “Spam, harassment, or abuse” is selected, else 0.           |
