package com.twitter.birdwatch.scoring

import com.twitter.birdwatch.thriftscala.{
  BirdwatchNote,
  BirdwatchNoteRating,
  BirdwatchNoteRatingStatus,
  BirdwatchScoredNote,
  BirdwatchUserReputation
}
import com.twitter.conversions.DurationOps._
import com.twitter.scalding.{DateOps, DateRange, Days}
import com.twitter.scalding.typed.{MultiJoin, TypedPipe, UnsortedGrouped}
import com.twitter.scalding_internal.dalv2.DAL

object HelpfulnessScoresUtil {
  implicit val tz = DateOps.UTC

  // This parameter adds pseudocounts to author helpfulness score computations
  val authorHelpfulnessScoreNumeratorPseudocounts: Int = 2
  val authorHelpfulnessScoreDenominatorPseudocounts: Int = 6
  val raterHelpfulnessScoreNumeratorPseudocounts: Int = 2
  val raterHelpfulnessScoreDenominatorPseudocounts: Int = 6

  // How many times to run the iterative author helpfulness algorithm unless overridden.
  val defaultMaxNumAuthorHelpfulnessScoreIterations: Int = 15

  def computeAuthorHelpfulnessScore(
    weightedHelpfulRatingsReceived: Double,
    weightedTotalRatingsReceived: Double
  ): Double = {
    // rawHelpfulRatio is a simple ratio ranging from 0.0 to 1.0
    val rawHelpfulRatio =
      (authorHelpfulnessScoreNumeratorPseudocounts + weightedHelpfulRatingsReceived) /
        (authorHelpfulnessScoreDenominatorPseudocounts + weightedTotalRatingsReceived)

    // Transform the range of the score to [-0.5, 1.0], then set any negative score to exactly 0.
    // This means for the final helpfulness score to be >0, the rawHelpfulRatio must be >1/3.
    math.max(0.0, ((3.0 * rawHelpfulRatio) - 1.0) / 2.0)
  }

  def computeRaterHelpfulnessScore(
    numberOfValidRatingsWhichMatchedLabel: Long,
    numberOfValidRatingsTotal: Long
  ): Double = {
    // rawHelpfulRatio is a simple ratio ranging from 0.0 to 1.0
    val rawHelpfulRatio =
      (raterHelpfulnessScoreNumeratorPseudocounts + numberOfValidRatingsWhichMatchedLabel.toDouble) /
        (raterHelpfulnessScoreDenominatorPseudocounts + numberOfValidRatingsTotal.toDouble)

    // Transform the range of the score to [-0.5, 1.0], then set any negative score to exactly 0.
    // This means for the final helpfulness score to be >0, the rawHelpfulRatio must be >1/3.
    math.max(0.0, ((3.0 * rawHelpfulRatio) - 1.0) / 2.0)
  }

  def getAuthorHelpfulnessScore(helpfulnessScores: BirdwatchUserReputation): Double = {
    helpfulnessScores.authorHelpfulnessScore.getOrElse(0.0)
  }

  def getRaterHelpfulnessScore(helpfulnessScores: BirdwatchUserReputation): Double = {
    helpfulnessScores.raterHelpfulnessScore.getOrElse(0.0)
  }

  def getCombinedHelpfulnessScore(helpfulnessScores: BirdwatchUserReputation): Double = {
    (helpfulnessScores.authorHelpfulnessScore.getOrElse(0.0) +
      helpfulnessScores.raterHelpfulnessScore.getOrElse(0.0)) / 2.0
  }

  type RaterId = Long
  type AuthorId = Long
  // Note: this function returns BirdwatchAggregateRating objects that still need to be aggregated.
  def weightRatingsByHelpfulnessScore(
    ratingsWithNotes: TypedPipe[(BirdwatchNoteRating, BirdwatchNote)],
    helpfulnessScores: TypedPipe[BirdwatchUserReputation],
    helpfulnessScoreFunction: BirdwatchUserReputation => Double,
    aggregateRatingConstructor: BirdwatchNoteRating => BirdwatchAggregateRating =
      BirdwatchAggregateRating.fromNoteRatingThriftWithoutHelpfulQuestionAnswered
  ): TypedPipe[(BirdwatchNote, (RaterId, BirdwatchAggregateRating))] = {
    joinHelpfulnessScoreOfRaterWithRatings(
      ratingsWithNotes,
      helpfulnessScores,
      helpfulnessScoreFunction)
      .flatMap {
        case (note, (rating, helpfulnessScoreOfRater)) =>
          rating.userId.map { raterId =>
            (
              note,
              (
                raterId,
                BirdwatchAggregateRating.weightByHelpfulnessScore(
                  aggregateRatingConstructor(rating),
                  helpfulnessScoreOfRater
                )
              )
            )
          }
      }
  }

  type UserScore = Double
  def joinHelpfulnessScoreOfRaterWithRatings(
    ratingsWithNotes: TypedPipe[(BirdwatchNoteRating, BirdwatchNote)],
    helpfulnessScores: TypedPipe[BirdwatchUserReputation],
    helpfulnessFunction: BirdwatchUserReputation => Double
  ): TypedPipe[(BirdwatchNote, (BirdwatchNoteRating, UserScore))] = {
    ratingsWithNotes
      .flatMap {
        case (rating, note) =>
          rating.userId.map { raterId =>
            (
              raterId,
              (rating, note)
            )
          }
      }
      .group
      .leftJoin(helpfulnessScores.groupBy { _.userId })
      .mapValues {
        case ((rating, note), helpfulnessScoreOfRaterOpt) =>
          (
            note,
            (
              rating,
              helpfulnessScoreOfRaterOpt
                .map { helpfulnessFunction(_) }.getOrElse(0.0)
            )
          )
      }
      .values
  }

  // Note: this function returns BirdwatchAggregateRating objects that still need to be aggregated.
  def weightNotesByHelpfulnessScore(
    notesWithRatingOptions: TypedPipe[(BirdwatchNote, Option[BirdwatchNoteRating])],
    helpfulnessScores: TypedPipe[BirdwatchUserReputation],
    helpfulnessScoreFunction: BirdwatchUserReputation => Double,
    aggregateRatingConstructor: BirdwatchNoteRating => BirdwatchAggregateRating =
      BirdwatchAggregateRating.fromNoteRatingThriftWithoutHelpfulQuestionAnswered
  ): TypedPipe[(BirdwatchNote, BirdwatchAggregateRating)] = {
    notesWithRatingOptions
      .map {
        case (note, ratingOption) =>
          (
            ratingOption.flatMap(_.userId),
            (ratingOption, note)
          )
      }
      .group
      .leftJoin(helpfulnessScores.groupBy { rep => Some(rep.userId) })
      .map {
        case (_, ((ratingOption, note), helpfulnessScoreOfRaterOption)) =>
          val aggRating = ratingOption
            .map { rating =>
              BirdwatchAggregateRating.weightByHelpfulnessScore(
                aggregateRatingConstructor(rating),
                helpfulnessScoreOfRaterOption
                  .map(helpfulnessScoreFunction(_)).getOrElse(0.0)
              )
            }.getOrElse(BirdwatchAggregateRating())
          (note, aggRating)
      }
  }

  def labelNotesWithWeights(
    ratingsWithNotes: TypedPipe[(BirdwatchNoteRating, BirdwatchNote)],
    helpfulnessScores: TypedPipe[BirdwatchUserReputation],
    helpfulnessScoreFunction: BirdwatchUserReputation => Double
  ): TypedPipe[BirdwatchScoredNote] = {
    weightRatingsByHelpfulnessScore(ratingsWithNotes, helpfulnessScores, helpfulnessScoreFunction)
      .map { case (note, (raterId, aggRating)) => (note, aggRating) }
      .group(Ordering.by(_.noteId))
      .sum(BirdwatchAggregateRating.ratingSemigroup)
      .map {
        case (note, aggRating) =>
          NoteScoringUtil.scoreNote(note, aggRating)
      }
  }

  def reweightAuthorHelpfulnessScoresNTimes(
    ratingsWithNotes: TypedPipe[(BirdwatchNoteRating, BirdwatchNote)],
    previousAuthorHelpfulnessScores: TypedPipe[BirdwatchUserReputation],
    maxAuthorHelpfulnessIterations: Option[Int]
  ): TypedPipe[BirdwatchUserReputation] = {
    (1 to maxAuthorHelpfulnessIterations.getOrElse(defaultMaxNumAuthorHelpfulnessScoreIterations))
      .foldLeft(previousAuthorHelpfulnessScores) { (rep, _) =>
        reweightAuthorHelpfulnessScores(ratingsWithNotes, rep)
      }
  }

  def getAuthorAggregateRatingsAveragedOverRaterAuthorPairs(
    ratingsPerRaterAuthorPair: TypedPipe[((RaterId, AuthorId), BirdwatchAggregateRating)]
  ): UnsortedGrouped[AuthorId, BirdwatchAggregateRating] = {
    ratingsPerRaterAuthorPair.group
      .sum(BirdwatchAggregateRating.ratingSemigroup)
      .map { // average helpfulness rating for each rater author pair
        case ((ratingUserId, noteAuthorId), aggregateRating) =>
          val averageRating = aggregateRating.copy(
            // total contains the actual number of ratings this rater has made on this author,
            //  so we'll divide the weightedHelpful and weightedTotal sums by total in order
            //  to get the average weighted values over all ratings of this rater on this author.
            weightedHelpful = aggregateRating.weightedHelpful.map {
              _ / aggregateRating.total.getOrElse(1L)
            },
            weightedTotal = aggregateRating.weightedTotal.map {
              _ / aggregateRating.total.getOrElse(1L)
            }
          )
          (noteAuthorId, averageRating)
      }
      .group
      .sum(BirdwatchAggregateRating.ratingSemigroup)
  }

  def reweightAuthorHelpfulnessScores(
    ratingsWithNotes: TypedPipe[(BirdwatchNoteRating, BirdwatchNote)],
    previousAuthorHelpfulnessScores: TypedPipe[BirdwatchUserReputation]
  ): TypedPipe[BirdwatchUserReputation] = {
    getAuthorAggregateRatingsAveragedOverRaterAuthorPairs(
      weightRatingsByHelpfulnessScore(
        ratingsWithNotes,
        previousAuthorHelpfulnessScores,
        getAuthorHelpfulnessScore,
        BirdwatchAggregateRating.fromNoteRatingThriftWithoutHelpfulQuestionAnswered
      ).map { case (note, (raterId, aggRating)) => ((raterId, note.userId), aggRating) }
    ).map {
      case (authorId, aggWeightedRatingsReceived) =>
        BirdwatchUserReputation(
          userId = authorId,
          aggregateRatingReceived =
            Some(BirdwatchAggregateRating.toAggregateRatingThrift(aggWeightedRatingsReceived)),
          authorHelpfulnessScore = Some(
            computeAuthorHelpfulnessScore(
              aggWeightedRatingsReceived.weightedHelpful.getOrElse(0.0),
              aggWeightedRatingsReceived.weightedTotal.getOrElse(0.0)
            )
          )
        )
    }
  }

  def computeRawAuthorHelpfulnessScores(
    ratingsWithNotes: TypedPipe[(BirdwatchNoteRating, BirdwatchNote)]
  ) = {
    getAuthorAggregateRatingsAveragedOverRaterAuthorPairs(
      ratingsWithNotes.flatMap {
        case (rating, note) =>
          rating.userId.map { ratingUserId =>
            (
              (ratingUserId, note.userId),
              BirdwatchAggregateRating.weightByHelpfulnessScore(
                BirdwatchAggregateRating.fromNoteRatingThriftWithoutHelpfulQuestionAnswered(rating),
                1.0 // give every author weight 1 at the start.
              )
            )
          }
      }
    ).map {
      case (userId, aggregateRating) =>
        BirdwatchUserReputation(
          userId = userId,
          aggregateRatingReceived =
            Some(BirdwatchAggregateRating.toAggregateRatingThrift(aggregateRating)),
          authorHelpfulnessScore = Some(
            computeAuthorHelpfulnessScore(
              aggregateRating.weightedHelpful.getOrElse(0.0),
              aggregateRating.weightedTotal.getOrElse(0.0)
            )
          )
        )
    }
  }

  def computeRaterHelpfulnessScores(
    ratingsWithNotes: TypedPipe[(BirdwatchNoteRating, BirdwatchNote)],
    labeledNotes: TypedPipe[BirdwatchScoredNote],
    authorHelpfulnessScores: TypedPipe[BirdwatchUserReputation]
  ): TypedPipe[BirdwatchUserReputation] = {
    // Ratings only update rater reputation if the rating was one of the first
    //  NoteScoringUtil.minRatingsToGetRatingsStatus (=5 currently) ratings,
    //  and if the rating was made within 48 hours of the note's creation, to avoid
    //  boosting rater helpfulness score by simply rating old labeled notes the same as the label.
    val durationAfterNoteCreationWhenRatingsCountTowardsRaterHelpfulnessScore = 2.days.inMillis
    val latestTimestampWhereRatingUpdatesRaterHelpfulnessScorePerNote = ratingsWithNotes
      .map {
        case (rating, note) => ((note.noteId, note.createdAt), rating.createdAt)
      }
      .group
      .sortedTake(NoteScoringUtil.minRatingsToGetRatingStatus)
      .map {
        case ((noteId, noteCreatedAt), earliestRatingTimestamps) =>
          val latestTimestampWhereRatingUpdatesRaterHelpfulnessScore =
            if (earliestRatingTimestamps.size == NoteScoringUtil.minRatingsToGetRatingStatus) {
              math.min(
                earliestRatingTimestamps.max,
                noteCreatedAt + durationAfterNoteCreationWhenRatingsCountTowardsRaterHelpfulnessScore
              )
            } else {
              noteCreatedAt + durationAfterNoteCreationWhenRatingsCountTowardsRaterHelpfulnessScore
            }
          (noteId, latestTimestampWhereRatingUpdatesRaterHelpfulnessScore)
      }

    val scoredNotesWithLabels = labeledNotes.flatMap { scoredNote =>
      for {
        ratingStatus <- scoredNote.birdwatchNoteRatingStatus
        aggRating <- scoredNote.aggregateRatingReceived
        if ratingStatus != BirdwatchNoteRatingStatus.NeedsMoreRatings
      } yield (scoredNote.noteId, (aggRating, ratingStatus))
    }.group

    val ratingsWithWeights =
      joinHelpfulnessScoreOfRaterWithRatings(
        ratingsWithNotes,
        authorHelpfulnessScores,
        getAuthorHelpfulnessScore
      ).map {
        case (note, (rating, authorHelpfulnessScore)) =>
          (note.noteId, (rating, authorHelpfulnessScore))
      }.group

    // Join each scoredNote with each time it was rated and the author helpfulness score of the rater.
    MultiJoin(
      scoredNotesWithLabels,
      ratingsWithWeights,
      latestTimestampWhereRatingUpdatesRaterHelpfulnessScorePerNote)
      .flatMap {
        case (
              noteId,
              (
                (noteAggRating, ratingStatus),
                (rating, authorHelpfulnessScore),
                latestTimestampWhereRatingUpdatesRaterHelpfulnessScore
              )
            ) =>
          val singleRatingsAsAggRating = BirdwatchAggregateRating.weightByHelpfulnessScore(
            BirdwatchAggregateRating.fromNoteRatingThrift(rating),
            authorHelpfulnessScore
          )
          // Compute what the consensus helpfulRatio is from all other raters besides this rater
          val helpfulMinusRater = noteAggRating.weightedHelpful.getOrElse(0.0) -
            singleRatingsAsAggRating.weightedHelpful.getOrElse(0.0)
          val totalMinusRater = noteAggRating.weightedTotal.getOrElse(0.0) -
            singleRatingsAsAggRating.weightedTotal.getOrElse(0.0)
          val helpfulRatioMinusRater =
            if (totalMinusRater == 0) 0 else helpfulMinusRater / totalMinusRater

          // Determine what the note's label would be without the current rater's rating.
          val labelWithoutRaterOpt =
            if (helpfulRatioMinusRater >= NoteScoringUtil.minRawHelpfulnessRatioToBeRatedHelpful) {
              Some(true)
            } else if (helpfulRatioMinusRater <= NoteScoringUtil.maxRawHelpfulnessRatioToBeRatedUnhelpful) {
              Some(false)
            } else {
              // The note doesn't count towards rater helpfulness if it would lose its label
              //   without the ratings from this particular rater.
              None
            }

          // Input: an individual note rating
          // Output: what this rater rated the note, and the consensus ratio of all other raters
          for {
            ratedHelpful <- singleRatingsAsAggRating.helpful
            ratedNotHelpful <- singleRatingsAsAggRating.notHelpful
            raterId <- rating.userId
            labelWithoutRater <- labelWithoutRaterOpt
            // Throw out ratings that left the "helpful" question unanswered.
            // Also throw out ratings that happened too long after note creation time.
            if (ratedHelpful == 1 || ratedNotHelpful == 1) &&
              (rating.createdAt <= latestTimestampWhereRatingUpdatesRaterHelpfulnessScore)
          } yield {
            val ratedHelpfulByRater = (ratedHelpful == 1)
            (
              raterId,
              (
                if (labelWithoutRater == ratedHelpfulByRater) 1L else 0L,
                1L // Included so that the next .sum can compute total ratings
              )
            )
          }
      }
      .group
      .sum
      .map {
        case (raterId, (numberOfValidRatingsWhichMatchedLabel, numberOfValidRatingsTotal)) =>
          val raterHelpfulnessScore = computeRaterHelpfulnessScore(
            numberOfValidRatingsWhichMatchedLabel,
            numberOfValidRatingsTotal
          )
          BirdwatchUserReputation(
            userId = raterId,
            raterHelpfulnessScore = Some(raterHelpfulnessScore)
          )

      }
  }

  /*
  Put both of them on the same object with outer join.
   */
  def combineHelpfulnessScores(
    authorHelpfulnessScores: TypedPipe[BirdwatchUserReputation],
    raterHelpfulnessScores: TypedPipe[BirdwatchUserReputation]
  ): TypedPipe[BirdwatchUserReputation] = {
    authorHelpfulnessScores
      .groupBy { _.userId }.outerJoin(raterHelpfulnessScores.groupBy { _.userId })
      .map {
        case (userId, (authorHelpfulnessScoreOpt, raterHelpfulnessScoresOpt)) =>
          authorHelpfulnessScoreOpt
            .getOrElse(
              BirdwatchUserReputation(
                userId = userId,
                authorHelpfulnessScore = None
              )).copy(
              raterHelpfulnessScore = raterHelpfulnessScoresOpt.flatMap(_.raterHelpfulnessScore)
            )
      }
  }

  def getHelpfulnessScores()(implicit dateRange: DateRange) = {
    DAL
      .readMostRecentSnapshot(
        BirdwatchUserReputationDailyScalaDataset,
        DateRange(dateRange.start - Days(7), dateRange.end + Days(1))
      ).toTypedPipe
  }
}
