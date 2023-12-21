---
title: Downloading data
description: All Community Notes contributions are publicly available so that anyone has free access to analyze the data.
navWeight: 1
---
# Downloading data

We can't wait to learn with you!

All Community Notes contributions are publicly available on the [Download Data](https://x.com/i/communitynotes/download-data) page of the Community Notes site so that anyone has free access to analyze the data, identify problems, and spot opportunities to make Community Notes better.

If you have questions or feedback about the Community Notes public data or would like to share your analyses of this data with us, please DM us at [@CommunityNotes](http://x.com/communitynotes).

---

## Working with the Community Notes data

### Data snapshots

The [Community Notes data](https://x.com/i/communitynotes/download-data) is released as four separate files:

- **Notes:** Contains a table representing all notes
- **Ratings:** Contains a table representing all ratings
- **Note Status History:** Contains a table with metadata about notes including what statuses they received and when.
- **User Enrollment:** Contains a table with metadata about each user's enrollment state.

These tables can be joined on the noteId field to create a combined dataset with information about users, notes, and their ratings. The data is released in separate tables/files to reduce the dataset size by avoiding data duplication (this is known as a normalized data model).

Currently, we release one cumulative file each for notes, notes status history, note ratings, and user enrollment. However, in the future, if the data ever grows too large, we will split the data into multiple files as needed.

A new snapshot of the Community Notes public data is released daily, on a best-effort basis, and technical difficulties may occur and delay the data release until the next day. We are not able to provide guarantees about when this may happen. The snapshots are cumulative files, but only contain notes and ratings that were created as of 48 hours before the dataset release time. When notes and ratings are deleted, they will no longer be released in any future versions of the data downloads, although the note status history dataset will continue to contain metadata about all scored notes even after they’ve been deleted, which includes noteId, creation time, the hashed participant ID of the note’s author, and a history of which statuses each notes received and when; however, all the content of the note itself e.g. the note’s text will no longer be available.

The [data download page in Community Notes](https://x.com/i/communitynotes/download-data) displays a date stamp indicating the most recent date of data included in the downloadable files.

### File structure

Each data snapshot table is stored in tsv (tab-separated values) file format with a header row. This means that each row is separated by a newline, each column is separated by a tab, and the first row contains the column names instead of data. The note and note rating data is directly taken from the user-submitted note creation and note rating forms, with minimal added metadata (like ids and timestamp). The note status history file contains metadata derived from the raw notes and ratings, and contains the outputs of the [note scoring algorithm](./ranking-notes.md). Below, we will describe each column’s data, including the question or source that generated the data, data type, and other relevant information.

### Updates to the Data

As we iterate and improve Community Notes, we will occasionally make changes to the questions we ask contributors in the note writing and note rating forms, or additional metadata shared about notes and rating. When we do this, some question fields / columns in our public data will be deprecated (no longer populated), and others will be added. Below we will keep a change log of changes we have made to the contribution form questions and other updates we have made to the data, as well as when those changes were made.

{% accordionSection %}
{% accordionItem title="2022-11-25 - New User Enrollment dataset "  %}

- New dataset with user enrollment states. These states define what actions users can take in the system (eg. rating, writing).

{% /accordionItem %}

{% accordionItem title="2022-10-27 - Deprecated fields in note writing form "  %}

- Deprecated Columns
  - `believable`
  - `harmful`
  - `validationDifficulty`

{% /accordionItem %}

{% accordionItem title="2022-07-19 - New Note Status History dataset "  %}

- Added entirely new note status history dataset

{% /accordionItem %}

{% accordionItem title="2021-12-15 - Updated Note Rating Questions"  %}

**Updated Columns**

- `notHelpfulArgumentativeOrInflammatory` - Changed name to `notHelpfulArgumentativeOrBiased`

**Added Columns**

- `helpfulUnbiasedLanguage`
- `notHelpfulOpinionSpeculation`
- `notHelpfulNoteNotNeeded`

{% /accordionItem %}
{% accordionItem title="2021-06-30 - Updated Note Rating Questions"  %}

- Note Helpfulness question now has 3 response categories (Yes, Somewhat, No), rather than 2 (originally: Yes, No)
- We have removed the ‘Agree’ note rating question
- We have updated the set of categories contributors can use to describe why a note is helpful or unhelpful. (Note: both helpful and unhelpful descriptors can be selected for notes that are rated as ‘Somewhat’ Helpful)

**Deprecated Columns**

- `helpful` - Replaced with helpfulnessLevel
- `notHelpful` - Replaced with helpfulnessLevel
- `helpfulInformative`
- `helpfulEmpathetic`
- `helpfulUniqueContext`
- `notHelpfulOpinionSpeculationOrBias`
- `notHelpfulOutdated`
- `notHelpfulOffTopic`

**Added Columns**

- `helpfulnessLevel`
- `helpfulAddressesClaim`
- `helpfulImportantContext`
- `notHelpfulIrrelevantSources`

{% /accordionItem %}

{% /accordionSection %}

### Notes

| Field | Type | Description | Response values |
| ---------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `noteId` | Long | The unique ID of this note | |
| `participantId` | String | A Community Notes-specific user identifier of the user who authored the note. This is a permanent id, which remains stable even if the user changes their username/handle. | |
| `createdAtMillis` | Long | Time the note was created, in milliseconds since epoch (UTC). | |
| `tweetId` | Long | The tweetId number for the tweet that the note is about. | |
| `classification` | String | User-entered multiple choice response to note writing question: “Given current evidence, I believe this tweet is:” | "NOT\*MISLEADING" "MISINFORMED*OR*POTENTIALLY_MISLEADING" |
| ~~`believable`~~ | String | User-entered multiple choice response to note writing question: “If this tweet were widely spread, its message would likely be believed by:” - **Deprecated as of 2022-10-27**. | "BELIEVABLE*BY*FEW", "BELIEVABLE_BY_MANY" |
| ~~`harmful`~~ | String | User-entered multiple choice response to note writing question: “If many believed this tweet, it might cause:” **Deprecated as of 2022-10-27**. | "LITTLE\*HARM", "CONSIDERABLE\*HARM" |
| ~~`validationDifficulty`~~ | String | User-entered multiple choice response to note writing question: “Finding and understanding the correct information would be:” **Deprecated as of 2022-10-27**. | "EASY", "CHALLENGING" |
| `misleadingOther` | Int | User-entered checkbox in response to question “Why do you believe this tweet may be misleading?” (Check all that apply question type). | 1 if “Other” is selected, else 0. |
| `misleadingFactualError` | Int | User-entered checkbox in response to question “Why do you believe this tweet may be misleading?” (Check all that apply question type) | 1 if “It contains a factual error” selected, else 0. |
| `misleadingManipulatedMedia` | Int | User-entered checkbox in response to question “Why do you believe this tweet may be misleading?” (Check all that apply question type) | 1 if “It contains a digitally altered photo or video” selected, else 0. |
| `misleadingOutdatedInformation` | Int | User-entered checkbox in response to question “Why do you believe this tweet may be misleading?” (Check all that apply question type) | 1 if “It contains outdated information that may be misleading” is selected, else 0. |
| `misleadingMissingImportantContext` | Int | User-entered checkbox in response to question “Why do you believe this tweet may be misleading?” (Check all that apply question type). | 1 if “It is a misrepresentation or missing important context” is selected, else 0. |
| `misleadingUnverifiedClaimAsFact` | Int | User-entered checkbox in response to question “Why do you believe this tweet may be misleading?” (Check all that apply question type). | 1 if “It presents an unverified claim as a fact” is selected, else 0. |
| `misleadingSatire` | Int | User-entered checkbox in response to question “Why do you believe this tweet may be misleading?” (Check all that apply question type). | 1 if “It is a joke or satire that might be misinterpreted as a fact” is selected, else 0. |
| `notMisleadingOther` | Int | User-entered checkbox in response to question “Why do you believe this tweet is not misleading?” (Check all that apply question type). | 1 if “Other” is selected, else 0. |
| `notMisleadingFactuallyCorrect` | Int | User-entered checkbox in response to question “Why do you believe this tweet is not misleading?” (Check all that apply question type). | 1 if “It expresses a factually correct claim” is selected, else 0. |
| `notMisleadingOutdatedButNotWhenWritten` | Int | User-entered checkbox in response to question “Why do you believe this tweet is not misleading?” (Check all that apply question type). | 1 if “This Tweet was correct when written, but is out of date now” is selected, else 0. |
| `notMisleadingClearlySatire` | Int | User-entered checkbox in response to question “Why do you believe this tweet is not misleading?” (Check all that apply question type). | 1 if “ It is clearly satirical/joking” is selected, else 0. |
| `notMisleadingPersonalOpinion` | Int | User-entered checkbox in response to question “Why do you believe this tweet is not misleading?” (Check all that apply question type). | 1 if “It expresses a personal opinion” is selected, else 0. |
| `trustworthySources` | Int | Binary indicator, based on user-entered multiple choice in response to note writing question “Did you link to sources you believe most people would consider trustworthy?” | 1 if “Yes” is selected, 0 if “No” is selected |
| `summary` | String | User-entered text, in response to the note writing prompt “Please explain the evidence behind your choices, to help others who see this tweet understand why it is not misleading” | User entered text explanation, with some characters escaped (e.g. tabs converted to spaces). |
| `isMediaNote` | Int | User-entered checkbox in response to question “Is your note about the Tweet or the image?”. _New as of 2023-05-24_. | 1 if “About the image in this Tweet, and should appear on all Tweets that include this image” is selected, and 0 otherwise (including both if "About this specific Tweet" is selected instead, or by default, e.g. if the note was written on a Tweet without media). |

### Note Status History

| Field                                 | Type   | Description                                                                                                                                                                                                                                                                                                                                                                                           | Response values                                                                |
| ------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `noteId`                              | Long   | The unique ID of this note.                                                                                                                                                                                                                                                                                                                                                                          |                                                                                |
| `participantId`                       | String | A Community Notes-specific user identifier of the user who authored the rating. This is a permanent id, which remains stable even if the user changes their username/handle.                                                                                                                                                                                                                         |                                                                                |
| `createdAtMillis`                     | Long   | Time the note was created, in milliseconds since epoch (UTC).                                                                                                                                                                                                                                                                                                                                              |                                                                                |
| `timestampMillisOfFirstNonNMRStatus`  | Long | The timestamp, in milliseconds since epoch (UTC), of when the note got its first status besides “Needs More Ratings”. Empty if the note never left “Needs More Ratings” status.                                                                                                                                                                                                                      | 1 if “Yes” is selected, 0 if “No” is selected                                  |
| `firstNonNMRStatus`                   | String | The first status the note received when it got a status besides “Needs More Ratings”. Empty if the note never left “Needs More Ratings” status.                                                                                                                                                                                                                                                | "", "CURRENTLY_RATED_HELPFUL", "CURRENTLY_RATED_NOT_HELPFUL"                   |
| `timestampMillisOfCurrentStatus`      | Long | The timestamp, in milliseconds since epoch (UTC), of when the note got its current status, including “Needs More Ratings”.                                                                                                                                                                                                                                                                                 | 1 if “Yes” is selected, 0 if “No” is selected                                  |
| `currentStatus`                       | String | The current status of the note.                                                                                                                                                                                                                                                                                                                                                                      | "NEEDS_MORE_RATINGS", "CURRENTLY_RATED_HELPFUL", "CURRENTLY_RATED_NOT_HELPFUL" |
| `timestampMillisOfLatestNonNMRStatus` | Long | The timestamp, in milliseconds since epoch (UTC), of when the note most recently received a status of either “Currently Rated Helpful” or “Currently Rated Not Helpful”. This value will be the same as timestampMillisOfFirstNonNMRStatus if the note has never switched status after receiving its first non-”Needs More Rating” status. Value is empty if the note never left “Needs More Ratings” status. | "NEEDS_MORE_RATINGS", "CURRENTLY_RATED_HELPFUL", "CURRENTLY_RATED_NOT_HELPFUL" |
| `latestNonNMRStatus`                  | String | The latest status the note received, when it got a status besides “Needs More Ratings”. Value is empty if the note never left “Needs More Ratings” status.                                                                                                                                                                                                                                              | "", "CURRENTLY_RATED_HELPFUL", "CURRENTLY_RATED_NOT_HELPFUL"                   |
| `timestampMillisOfStatusLock` | Long | The timestamp, in milliseconds since epoch (UTC), of when the note's status was locked. Value is empty if the note's status is unlocked. | 
| `lockedStatus` | String | The status that the note is locked at. | "", "NEEDS_MORE_RATINGS", "CURRENTLY_RATED_HELPFUL", "CURRENTLY_RATED_NOT_HELPFUL" |  
| `timestampMillisOfRetroLock` | Long | The timestamp, in milliseconds since epoch (UTC), of when the note's status was retroactively locked. Value is empty if the note's status was not retroactively locked. Retroactive locking was a one-time event on January 20, 2023, which applied status locking rules to notes that were created before status locking was first launched. | 
| `currentCoreStatus` | String | The current status, if any, assigned by the core submodel. | "", "NEEDS_MORE_RATINGS", "CURRENTLY_RATED_HELPFUL", "CURRENTLY_RATED_NOT_HELPFUL" |  
| `currentCoreStatus` | String | The current status, if any, assigned by the core submodel. | "", "NEEDS_MORE_RATINGS", "CURRENTLY_RATED_HELPFUL", "CURRENTLY_RATED_NOT_HELPFUL" |
| `currentExpansionStatus` | String | The current status, if any, assigned by the expansion submodel. | "", "NEEDS_MORE_RATINGS", "CURRENTLY_RATED_HELPFUL", "CURRENTLY_RATED_NOT_HELPFUL" |
| `currentGroupStatus` | String | The current status, if any, assigned by the group submodel. | "", "NEEDS_MORE_RATINGS", "CURRENTLY_RATED_HELPFUL", "CURRENTLY_RATED_NOT_HELPFUL" |
| `currentDecidedByKey` | String | The submodel whose status was used to determine the note's overall current status. | "CoreModel (v1.1)", "ExpansionModel (v1.1)", "GroupModel01 (v1.1)", "GroupModel02 (v1.1)", ..., "InsufficientExplanation (v1.0)", "ScoringDriftGuard (v1.0)" |
| `currentModelingGroup` | Int | The ID of the modeling group that this note would be scored by, if eligible to be scored by a group model (determined by the modeling groups of its raters, from the user enrollment file). 0 is a placeholder for no modeling group. | 0-13 |


### Ratings

| Field                                    | Type   | Description                                                                                                                                                                                                          | Response values                                                  |
| ---------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `noteId`                                 | Long   | The unique ID of the note being rated.                                                                                                                                                                              |                                                                  |
| `participantId`                          | String | A Community Notes-specific user identifier of the user who authored the rating. This is a permanent id, which remains stable even if the user changes their username/handle.                                        |                                                                  |
| `createdAtMillis`                        | Long   | Time the note was created, in milliseconds since epoch (UTC).                                                                                                                                                             |                                                                  |
| `agree`                                  | Int    | Binary indicator, based on user-entered multiple choice in response to note rating question “Do you agree with its conclusion?”                                                                                     | 1 if “Yes” is selected, 0 if “No” is selected                    |
| `disagree`                               | Int    | Binary indicator, based on user-entered multiple choice in response to note rating question “Do you agree with its conclusion?”                                                                                     | 1 if “No” is selected, 0 if “Yes” is selected                    |
| ~~`helpful`~~                            | Int    | Binary indicator, based on user-entered multiple choice in response to note rating question “Is this note helpful? ” - _Deprecated as of 2021-06-30_.                                                               | 1 if “Yes” is selected, 0 if “No” is selected                    |
| ~~`notHelpful`~~                         | Int    | Binary indicator, based on user-entered multiple choice in response to note rating question “Is this note helpful?” _Deprecated as of 2021-06-30_.                                                                  | 1 if “No” is selected, 0 if “Yes” is selected                    |
| `helpfulnessLevel`                       | String | User-entered multiple choice response to note rating question: “Is this note helpful” _Added as of 2021-06-30_.                                                                                                     | "NOT_HELPFUL" "SOMEWHAT_HELPFUL" "HELPFUL"                       |
| `helpfulOther`                           | Int    | User-entered checkbox in response to question “What about this note was helpful to you?” (Check all that apply question type).                                                                                      | 1 if “Other” is selected, else 0.                                |
| ~~`helpfulInformative`~~                 | Int    | User-entered checkbox in response to question “What about this note was helpful to you?” (Check all that apply question type). _Deprecated as of 2021-06-30_.                                                       | 1 if “Informative” is selected, else 0.                          |
| `helpfulClear`                           | Int    | User-entered checkbox in response to question “What about this note was helpful to you?” (Check all that apply question type).                                                                                      | 1 if “Clear and/or well-written” is selected, else 0.            |
| ~~`helpfulEmpathetic`~~                  | Int    | User-entered checkbox in response to question “What about this note was helpful to you?” (Check all that apply question type). _Deprecated as of 2021-06-30_.                                                       | 1 if “Nonjudgmental and/or empathetic” is selected, else 0.      |
| `helpfulGoodSources`                     | Int    | User-entered checkbox in response to question “What about this note was helpful to you?” (Check all that apply question type).                                                                                      | 1 if “Cites high-quality sources” is selected, else 0.           |
| ~~`helpfulUniqueContext`~~               | Int    | User-entered checkbox in response to question “What about this note was helpful to you?” (Check all that apply question type). _Deprecated as of 2021-06-30_.                                                       | 1 if “Offers unique information or context” is selected, else 0. |
| `helpfulAddressesClaim`                  | Int    | User-entered checkbox in response to question “What was helpful about it?” (Check all that apply question type). _New as of 2021-06-30_                                                                             | 1 if “Directly addresses the Tweet's claim” is selected, else 0. |
| `helpfulImportantContext`                | Int    | User-entered checkbox in response to question “What was helpful about it?” (Check all that apply question type). _New as of 2021-06-30_                                                                             | 1 if “Provides important context” is selected, else 0.           |
| `helpfulUnbiasedLanguage`                | Int    | User-entered checkbox in response to question “What was helpful about it?” (Check all that apply question type). _New as of 2021-12-15_                                                                             | 1 if “Neutral or unbiased language” is selected, else 0.         |
| `notHelpfulOther`                        | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type).                                                                                  | 1 if “Other” is selected, else 0.                                |
| `notHelpfulIncorrect`                    | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type).                                                                                  | 1 if “Incorrect information” is selected, else 0.                |
| `notHelpfulSourcesMissingOrUnreliable`   | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type).                                                                                  | 1 if “Sources missing or unreliable” is selected, else 0.        |
| ~~`NotHelpfulOpinionSpeculationOrBias`~~ | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type). _Deprecated as of 2021-06-30_.                                                   | 1 if “Opinion, speculation, or bias” is selected, else 0.        |
| `notHelpfulMissingKeyPoints`             | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type).                                                                                  | 1 if “Misses key points or irrelevant” is selected, else 0.      |
| ~~`notHelpfulOutdated`~~                 | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type). _Deprecated as of 2021-06-30_.                                                   | 1 if “Outdated information” is selected, else 0.                 |
| `notHelpfulHardToUnderstand`             | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type).                                                                                  | 1 if “Hard to understand” is selected, else 0.                   |
| `notHelpfulArgumentativeOrBiased`        | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type). _Variable name changed from notHelpfulArgumentativeOrInflammatory in 2021-12-15_ | 1 if “Argumentative or biased language is selected, else 0.      |
| ~~`notHelpfulOffTopic`~~                 | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type). _Deprecated as of 2021-06-30_.                                                   | 1 if “Off topic” is selected, else 0.                            |
| `notHelpfulSpamHarassmentOrAbuse`        | Int    | User-entered checkbox in response to prompt “Help us understand why this note was unhelpful” (Check all that apply question type).                                                                                  | 1 if “Spam, harassment, or abuse” is selected, else 0.           |
| `notHelpfulIrrelevantSources`            | Int    | User-entered checkbox in response to prompt “What was unhelpful about it?” (Check all that apply question type). New as of 2021-06-30                                                                               | 1 if “Sources do not support note” is selected, else 0.          |
| `notHelpfulOpinionSpeculation`           | Int    | User-entered checkbox in response to prompt “What was unhelpful about it?” (Check all that apply question type). New as of 2021-12-15                                                                               | 1 if “Opinion or speculation” is selected, else 0.               |
| `notHelpfulNoteNotNeeded`                | Int    | User-entered checkbox in response to prompt “What was unhelpful about it?” (Check all that apply question type). New as of 2021-12-15                                                                               | 1 if “Note not needed on this Tweet” is selected, else 0.        |
| `ratedOnTweetId`                         | Long   | The unique ID of the Tweet that the note was rated on (which, in the case of media notes, may not be the same Tweet as the note was originally written on). _New as of 2023-05-24_.                                 |

### User Enrollment

| Field                            | Type   | Description                                                                                                                                                                   | Response values                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| -------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `participantId`                  | String | A Community Notes-specific user identifier of the user who authored the rating. This is a permanent id, which remains stable even if the user changes their username/handle. |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `enrollmentState`                | String | Defines the user's enrollment state and the actions they can take on the system                                                                                              | `newUser`: newly admitted users, who only have rating ability. <br/> `earnedIn`: users who've earned writing ability. <br/> `atRisk`: users who are one Not Helpful note away from having writing ability locked. <br/> `earnedOutNoAcknowledge`: users with writing ability locked that have not yet clicked the acknowledgement button it in the product. <br/> `earnedOutAcknowledge`: users who've lost the ability to write and acknowledged it in the product, at which point their ratings start counting towards going back to `earnedIn`. |
| `successfulRatingNeededToEarnIn` | Int    | The target Rating Impact a user has to reach to earn the ability to write notes. Starts                                                                                      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `timestampOfLastStateChange`     | Long   | The timestamp, in milliseconds since epoch (UTC), of the most recent user enrollment state change                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `timestampOfLastEarnOut`         | Long   | The timestamp, in milliseconds since epoch (UTC), of the most recent time the user earned-out. If the user never earned out, its value will be 1                                   |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `modelingPopulation`             | String | Indicates which modeling population the user is, and therefore which models will score the user's ratings:.                                                                  | "CORE" or "EXPANSION"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `modelingGroup` | Int | The ID of the user's modeling group (used for group model scorers). | 0-13 |
