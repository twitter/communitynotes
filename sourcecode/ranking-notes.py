import pandas as pd
notes = pd.read_csv('notes-00000.tsv', sep='\t')
ratings = pd.read_csv('ratings-00000.tsv', sep='\t')

## Note: this code snippet's results won't match the results of Birdwatch in production, 
##   because this code snippet doesn't weight ratings by contributors' helpfulness scores.

ratings['helpfulScore'] = 0
ratings.loc[ratings['helpful']==1,'helpfulScore'] = 1
ratings.loc[ratings['helpfulnessLevel']=='SOMEWHAT_HELPFUL','helpfulScore'] = 0.5
ratings.loc[ratings['helpfulnessLevel']=='HELPFUL','helpfulScore'] = 1

ratingsWithNotes = notes.set_index('noteId').join(ratings.set_index('noteId'), lsuffix="\_note", rsuffix="\_rating", how='inner')
ratingsWithNotes['numRatings'] = 1


def getScoredNotesForTweet(
    tweetId,
    minRatingsNeeded = 5,
    minHelpfulnessRatioNeededHelpful = 0.84,
    maxHelpfulnessRatioNeededNotHelpful = .29,
    minRatingsToGetTag = 2,
):
    ratingsWithNotesForTweet = ratingsWithNotes[ratingsWithNotes['tweetId']==tweetId]
    scoredNotes = ratingsWithNotesForTweet.groupby('noteId').sum()
    scoredNotes['helpfulnessRatio'] = scoredNotes['helpfulScore']/scoredNotes['numRatings']

    helpfulWhys = ['helpfulOther', 'helpfulInformative', 'helpfulClear',
                   'helpfulGoodSources', 'helpfulEmpathetic', 'helpfulUniqueContext']
    notHelpfulWhys = ['notHelpfulOther', 'notHelpfulOpinionSpeculationOrBias', 'notHelpfulSourcesMissingOrUnreliable',
                      'notHelpfulMissingKeyPoints', 'notHelpfulArgumentativeOrInflammatory', 'notHelpfulIncorrect',
                      'notHelpfulOffTopic', 'notHelpfulHardToUnderstand', 'notHelpfulSpamHarassmentOrAbuse', 'notHelpfulOutdated']
    scoredNotes['ratingStatus'] = 'Needs More Ratings'
    scoredNotes.loc[(scoredNotes['numRatings'] >= minRatingsNeeded) & (scoredNotes['helpfulnessRatio'] >= minHelpfulnessRatioNeededHelpful), 'ratingStatus'] = 'Currently Rated Helpful'
    scoredNotes.loc[(scoredNotes['numRatings'] >= minRatingsNeeded) & (scoredNotes['helpfulnessRatio'] <= maxHelpfulnessRatioNeededNotHelpful), 'ratingStatus'] = 'Currently Not Rated Helpful'
    scoredNotes['firstTag'] = np.nan
    scoredNotes['secondTag'] = np.nan

    def topWhys(row):
        if row['ratingStatus']=='Currently Rated Helpful':
            whyCounts = pd.DataFrame(row[helpfulWhys])
        elif row['ratingStatus']=='Currently Not Rated Helpful':
            whyCounts = pd.DataFrame(row[notHelpfulWhys])
        else:
            return row
        whyCounts.columns = ['tagCounts']
        whyCounts['tiebreakOrder'] = range(len(whyCounts))
        whyCounts = whyCounts[whyCounts['tagCounts'] >= minRatingsToGetTag]
        topTags = whyCounts.sort_values(by=['tagCounts','tiebreakOrder'], ascending=False)[:2]
        if (len(topTags) < 2):
            row['ratingStatus'] = 'Needs More Ratings'
        else:
            row['firstTag'] = topTags.index[0]
            row['secondTag'] = topTags.index[1]
        return row

    scoredNotes = scoredNotes.apply(topWhys, axis=1)

    scoredNotes = scoredNotes.join(notes[['noteId','summary']].set_index('noteId'), lsuffix="_note", rsuffix="_rating", how='inner')

    scoredNotes['orderWithinStatus'] = 'helpfulnessRatio'
    scoredNotes.loc[scoredNotes['ratingStatus']=='Needs More Ratings', 'orderWithinStatus'] = 'createdAtMillis_note'
    statusOrder = {'Currently Rated Helpful':2, 'Needs More Ratings':1, 'Currently Not Rated Helpful':0}
    scoredNotes['statusOrder'] = scoredNotes.apply(lambda x: statusOrder[x['ratingStatus']], axis=1)
    return scoredNotes.sort_values(by=['statusOrder','orderWithinStatus'], ascending=False)
    