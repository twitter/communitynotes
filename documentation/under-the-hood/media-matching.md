---
title: Media Matching
description: How media matching works
navWeight: 30
---
# Media Matching

Contributors can write [notes on media](../contributing/notes-on-media.md) (images, videos and links) that show on posts that contain matching media. 

### Earning the ability to write notes on media
Contributors can write notes on media if they have sufficient [Writing Impact](../contributing/writing-and-rating-impact.md):
* Contributors with 2+ Writing Impact can write notes on images and videos (though not links), and their notes get associated with other posts with matching media if there is sufficient agreement from raters that the note will be helpful on all posts with matching media
* Contributors with 10+ Writing Impact can write notes on images, videos and links, and their notes are automatically associated with other posts with matching media

### Determining sufficient agreement to match a note to other posts
To be shown to everyone on X, notes need to reach a status of Helpful. Additionally, media notes may need further agreement that the note will be helpful on all posts that contain matching media. This helps prevent cases where a note is specific to a given post, but ends up matching to many other irrelevant posts with the same media, simply because the note author wrote it as a media note. 

When rating a media note, contributors answer this additional question:

![Media match rating question](../images/media-note-match-upgrade.png)

Currently, sufficient agreement is defined as:
* 5+ raters with positive rater factors and 5+ raters with negative rater factors agree the note will be helpful if shown on all posts with matching media
* 80%+ of raters from each factor sign agree the note will be helpful if shown on all posts with matching media

The rater factor used in the calculation is the factor from the [model that determined the note's status](./ranking-notes.md) (e.g. CoreModel, GroupModel01, etc)

### Link matching
Notes on links are new. Some details on how link matching works in this initial experimental phase:
* Links can have many variations — URL shortened (ex.co/a1b2), URL parameters (?source=shareLink), etc. Link matching aims to match on the ultimate URL to which the link resolves, ignoring unnecessary URL parameters.
* Notes are not currently allowed on base domains (like example.com)
* Notes that are not rated Helpful within two weeks will stop matching to new posts.

We will be listening to contributor feedback and may adjust these details.


