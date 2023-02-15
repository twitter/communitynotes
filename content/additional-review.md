---
title: Additional Community review
geekdocBreadcrumb: false
aliases: ["/additional-review"]
description: Request additional review of Community Notes on your Tweets.
---

[Community Notes](./_index.md) allows volunteer contributors on Twitter to collaboratively add context (called “notes”) to Tweets they believe could be misleading.

If you believe a note rated “helpful” on your Tweet doesn’t add helpful context or shouldn’t be there, you can request additional review. Reviews aren’t done by Twitter — they are done by regular people who use Twitter and have signed up to become Community Notes Contributors.

## How additional reviews work:

- Tweet authors can request additional contributor review of notes that are rated helpful and are being shown on their Tweet.
- When an author requests additional review, the note is shown to contributors on the Community Notes site, which is separate from the Twitter apps and where they may then rate the note.
- All ratings are done by Community Notes contributors, who voluntarily review notes, so there’s no guarantee that more contributors will review it or that the note's rating will change.
- Twitter does not determine the outcome of reviews. The aim of Community Notes is to empower people on Twitter to determine what additional context is helpful.
- If the additional ratings change a note's status such that it is no longer rated "helpful", the note will stop being shown on the Tweet.
- Additional ratings will likely come in within 24 hours following an author's request, but status can change at any time as ratings are received.
- A review of one note does not impact the status of other existing notes or newly written notes on the Tweet. If new notes are added and earn the status of Helpful, they may be shown on the Tweet.
- Notes are subject to the Twitter Rules. If you believe a note violates them, you can report it to Twitter.

## Request additional review

1. <div><strong> Copy your Tweet's link.</strong><label> It should look like this: <code>https://twitter.com/{{yourdisplayname}}/status/123456789</code> </label></div>

2. <div><strong> Paste the link into the form below.</strong></div>

3. <div><strong> Click the `Go to Community Notes` button.</strong></div>

4. <div><strong> In Community Notes, look for the note that you believe requires additional review and click "Request additional review".</strong></div>

<form style="display: flex; flex-direction: column;">
<label for="Tweet URL">Tweet URL</label>
<input name="Tweet URL" type="text" style="font-size: 1rem; margin-bottom: 4px; padding: 1rem; border: none; border-bottom: 2px solid black; background: #eee; border-radius: 2px;" id="input" ></input>
<button onClick="openNotes()" style="padding: 1rem; border-radius: 100px; background-color: black; color: white; font-weight: bold; font-size: 1rem;">Go to Community Notes</button>
</form>

<script>
    var openNotes = () => {
        var input = document.getElementById("input");
        var text = input.value;
        if (text.includes("/status/")) {
            // get the tweet id
            var tweetId = text.split("/status/")[1].split("?")[0];
            if (tweetId.match(/^[0-9]+$/)) {
            window.open("https://twitter.com/i/communitynotes/t/" + tweetId, "_blank");
        } else {
            alert("Invalid Tweet URL");
        }
    } else {
        alert("Invalid Tweet URL");
    }
    }
</script>
