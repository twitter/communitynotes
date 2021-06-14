package com.twitter.birdwatch.scoring

import com.twitter.algebird.Semigroup
import com.twitter.birdwatch.exporter.{BirdwatchNoteRatingTsv, BirdwatchTsvConverter}
import com.twitter.birdwatch.thriftscala
import com.twitter.birdwatch.thriftscala.BirdwatchNoteRating

case class BirdwatchAggregateRating(
  total: Option[Long] = Some(0L),
  agree: Option[Long] = Some(0L),
  disagree: Option[Long] = Some(0L),
  helpful: Option[Long] = Some(0L),
  notHelpful: Option[Long] = Some(0L),
  helpfulOther: Option[Long] = Some(0L),
  helpfulInformative: Option[Long] = Some(0L),
  helpfulClear: Option[Long] = Some(0L),
  helpfulEmpathetic: Option[Long] = Some(0L),
  helpfulGoodSources: Option[Long] = Some(0L),
  helpfulUniqueContext: Option[Long] = Some(0L),
  notHelpfulOther: Option[Long] = Some(0L),
  notHelpfulIncorrect: Option[Long] = Some(0L),
  notHelpfulSourcesMissingOrUnreliable: Option[Long] = Some(0L),
  notHelpfulOpinionSpeculationOrBias: Option[Long] = Some(0L),
  notHelpfulMissingKeyPoints: Option[Long] = Some(0L),
  notHelpfulOutdated: Option[Long] = Some(0L),
  notHelpfulHardToUnderstand: Option[Long] = Some(0L),
  notHelpfulArgumentativeOrInflammatory: Option[Long] = Some(0L),
  notHelpfulOffTopic: Option[Long] = Some(0L),
  notHelpfulSpamHarassmentOrAbuse: Option[Long] = Some(0L),
  latestRatingCreatedAt: Option[Long] = Some(0L),
  weightedTotal: Option[Double] = None,
  weightedHelpful: Option[Double] = None)

object BirdwatchAggregateRating {
  // the semigroup defines addition of BirdwatchAggregateRatings
  val ratingSemigroup = new Semigroup[BirdwatchAggregateRating] {
    override def plus(
      i: BirdwatchAggregateRating,
      j: BirdwatchAggregateRating
    ): BirdwatchAggregateRating = {
      BirdwatchAggregateRating(
        total = (i.total ++ j.total).reduceOption(_ + _),
        agree = (i.agree ++ j.agree).reduceOption(_ + _),
        disagree = (i.disagree ++ j.disagree).reduceOption(_ + _),
        helpful = (i.helpful ++ j.helpful).reduceOption(_ + _),
        notHelpful = (i.notHelpful ++ j.notHelpful).reduceOption(_ + _),
        helpfulOther = (i.helpfulOther ++ j.helpfulOther).reduceOption(_ + _),
        helpfulInformative = (i.helpfulInformative ++ j.helpfulInformative).reduceOption(_ + _),
        helpfulClear = (i.helpfulClear ++ j.helpfulClear).reduceOption(_ + _),
        helpfulEmpathetic = (i.helpfulEmpathetic ++ j.helpfulEmpathetic).reduceOption(_ + _),
        helpfulGoodSources = (i.helpfulGoodSources ++ j.helpfulGoodSources).reduceOption(_ + _),
        helpfulUniqueContext =
          (i.helpfulUniqueContext ++ j.helpfulUniqueContext).reduceOption(_ + _),
        notHelpfulOther = (i.notHelpfulOther ++ j.notHelpfulOther).reduceOption(_ + _),
        notHelpfulIncorrect = (i.notHelpfulIncorrect ++ j.notHelpfulIncorrect).reduceOption(_ + _),
        notHelpfulSourcesMissingOrUnreliable =
          (i.notHelpfulSourcesMissingOrUnreliable ++ j.notHelpfulSourcesMissingOrUnreliable)
            .reduceOption(_ + _),
        notHelpfulOpinionSpeculationOrBias =
          (i.notHelpfulOpinionSpeculationOrBias ++ j.notHelpfulOpinionSpeculationOrBias)
            .reduceOption(_ + _),
        notHelpfulMissingKeyPoints =
          (i.notHelpfulMissingKeyPoints ++ j.notHelpfulMissingKeyPoints).reduceOption(_ + _),
        notHelpfulOutdated = (i.notHelpfulOutdated ++ j.notHelpfulOutdated).reduceOption(_ + _),
        notHelpfulHardToUnderstand =
          (i.notHelpfulHardToUnderstand ++ j.notHelpfulHardToUnderstand).reduceOption(_ + _),
        notHelpfulArgumentativeOrInflammatory =
          (i.notHelpfulArgumentativeOrInflammatory ++ j.notHelpfulArgumentativeOrInflammatory)
            .reduceOption(_ + _),
        notHelpfulOffTopic = (i.notHelpfulOffTopic ++ j.notHelpfulOffTopic).reduceOption(_ + _),
        notHelpfulSpamHarassmentOrAbuse =
          (i.notHelpfulSpamHarassmentOrAbuse ++ j.notHelpfulSpamHarassmentOrAbuse)
            .reduceOption(_ + _),
        latestRatingCreatedAt = (i.latestRatingCreatedAt ++ j.latestRatingCreatedAt)
          .reduceOption(Math.max(_: Long, _: Long)),
        weightedTotal = (i.weightedTotal ++ j.weightedTotal).reduceOption(_ + _),
        weightedHelpful = (i.weightedHelpful ++ j.weightedHelpful).reduceOption(_ + _),
      )
    }
  }

  def weightByHelpfulnessScore(
    aggRating: BirdwatchAggregateRating,
    helpfulnessScore: Double
  ): BirdwatchAggregateRating = {
    aggRating.copy(
      weightedTotal = aggRating.total.map(_ * helpfulnessScore),
      weightedHelpful = aggRating.helpful.map(_ * helpfulnessScore)
    )
  }

  def fromAggregateRatingThrift(
    thriftAggregateRating: thriftscala.BirdwatchAggregateRating
  ): BirdwatchAggregateRating = {
    BirdwatchAggregateRating(
      total = thriftAggregateRating.total,
      agree = thriftAggregateRating.agree,
      disagree = thriftAggregateRating.disagree,
      helpful = thriftAggregateRating.helpful,
      notHelpful = thriftAggregateRating.notHelpful,
      helpfulOther = thriftAggregateRating.helpfulOther,
      helpfulInformative = thriftAggregateRating.helpfulInformative,
      helpfulClear = thriftAggregateRating.helpfulClear,
      helpfulEmpathetic = thriftAggregateRating.helpfulEmpathetic,
      helpfulGoodSources = thriftAggregateRating.helpfulGoodSources,
      helpfulUniqueContext = thriftAggregateRating.helpfulUniqueContext,
      notHelpfulOther = thriftAggregateRating.notHelpfulOther,
      notHelpfulIncorrect = thriftAggregateRating.notHelpfulIncorrect,
      notHelpfulSourcesMissingOrUnreliable =
        thriftAggregateRating.notHelpfulSourcesMissingOrUnreliable,
      notHelpfulOpinionSpeculationOrBias = thriftAggregateRating.notHelpfulOpinionSpeculationOrBias,
      notHelpfulMissingKeyPoints = thriftAggregateRating.notHelpfulMissingKeyPoints,
      notHelpfulOutdated = thriftAggregateRating.notHelpfulOutdated,
      notHelpfulHardToUnderstand = thriftAggregateRating.notHelpfulHardToUnderstand,
      notHelpfulArgumentativeOrInflammatory =
        thriftAggregateRating.notHelpfulArgumentativeOrInflammatory,
      notHelpfulOffTopic = thriftAggregateRating.notHelpfulOffTopic,
      notHelpfulSpamHarassmentOrAbuse = thriftAggregateRating.notHelpfulSpamHarassmentOrAbuse,
      latestRatingCreatedAt = thriftAggregateRating.latestRatingCreatedAt,
      weightedTotal = thriftAggregateRating.weightedTotal,
      weightedHelpful = thriftAggregateRating.weightedHelpful
    )
  }

  def toAggregateRatingThrift(
    aggregateRating: BirdwatchAggregateRating
  ): thriftscala.BirdwatchAggregateRating = {
    thriftscala.BirdwatchAggregateRating(
      total = aggregateRating.total,
      agree = aggregateRating.agree,
      disagree = aggregateRating.disagree,
      helpful = aggregateRating.helpful,
      notHelpful = aggregateRating.notHelpful,
      helpfulOther = aggregateRating.helpfulOther,
      helpfulInformative = aggregateRating.helpfulInformative,
      helpfulClear = aggregateRating.helpfulClear,
      helpfulEmpathetic = aggregateRating.helpfulEmpathetic,
      helpfulGoodSources = aggregateRating.helpfulGoodSources,
      helpfulUniqueContext = aggregateRating.helpfulUniqueContext,
      notHelpfulOther = aggregateRating.notHelpfulOther,
      notHelpfulIncorrect = aggregateRating.notHelpfulIncorrect,
      notHelpfulSourcesMissingOrUnreliable = aggregateRating.notHelpfulSourcesMissingOrUnreliable,
      notHelpfulOpinionSpeculationOrBias = aggregateRating.notHelpfulOpinionSpeculationOrBias,
      notHelpfulMissingKeyPoints = aggregateRating.notHelpfulMissingKeyPoints,
      notHelpfulOutdated = aggregateRating.notHelpfulOutdated,
      notHelpfulHardToUnderstand = aggregateRating.notHelpfulHardToUnderstand,
      notHelpfulArgumentativeOrInflammatory = aggregateRating.notHelpfulArgumentativeOrInflammatory,
      notHelpfulOffTopic = aggregateRating.notHelpfulOffTopic,
      notHelpfulSpamHarassmentOrAbuse = aggregateRating.notHelpfulSpamHarassmentOrAbuse,
      latestRatingCreatedAt = aggregateRating.latestRatingCreatedAt,
      weightedTotal = aggregateRating.weightedTotal,
      weightedHelpful = aggregateRating.weightedHelpful
    )
  }

  def fromNoteRatingTsv(tsvNoteRating: BirdwatchNoteRatingTsv): BirdwatchAggregateRating = {
    BirdwatchAggregateRating(
      total = Some(1L),
      agree = Some(tsvNoteRating.agree),
      disagree = Some(tsvNoteRating.disagree),
      helpful = Some(tsvNoteRating.helpful),
      notHelpful = Some(tsvNoteRating.notHelpful),
      helpfulOther = Some(tsvNoteRating.helpfulOther),
      helpfulInformative = Some(tsvNoteRating.helpfulInformative),
      helpfulClear = Some(tsvNoteRating.helpfulClear),
      helpfulEmpathetic = Some(tsvNoteRating.helpfulEmpathetic),
      helpfulGoodSources = Some(tsvNoteRating.helpfulGoodSources),
      helpfulUniqueContext = Some(tsvNoteRating.helpfulUniqueContext),
      notHelpfulOther = Some(tsvNoteRating.notHelpfulOther),
      notHelpfulIncorrect = Some(tsvNoteRating.notHelpfulIncorrect),
      notHelpfulSourcesMissingOrUnreliable =
        Some(tsvNoteRating.notHelpfulSourcesMissingOrUnreliable),
      notHelpfulOpinionSpeculationOrBias = Some(tsvNoteRating.notHelpfulOpinionSpeculationOrBias),
      notHelpfulMissingKeyPoints = Some(tsvNoteRating.notHelpfulMissingKeyPoints),
      notHelpfulOutdated = Some(tsvNoteRating.notHelpfulOutdated),
      notHelpfulHardToUnderstand = Some(tsvNoteRating.notHelpfulHardToUnderstand),
      notHelpfulArgumentativeOrInflammatory =
        Some(tsvNoteRating.notHelpfulArgumentativeOrInflammatory),
      notHelpfulOffTopic = Some(tsvNoteRating.notHelpfulOffTopic),
      notHelpfulSpamHarassmentOrAbuse = Some(tsvNoteRating.notHelpfulSpamHarassmentOrAbuse),
      latestRatingCreatedAt = Some(tsvNoteRating.createdAtMillis),
      weightedTotal = None,
      weightedHelpful = None
    )
  }

  def fromNoteRatingThrift(thriftNoteRating: BirdwatchNoteRating): BirdwatchAggregateRating = {
    fromNoteRatingTsv(BirdwatchTsvConverter.convertNoteRatingFromThrift(thriftNoteRating))
  }

  def fromNoteRatingThriftWithoutHelpfulQuestionAnswered(
    thriftNoteRating: BirdwatchNoteRating
  ): BirdwatchAggregateRating = {
    val rawAggregateRating = fromNoteRatingThrift(thriftNoteRating)

    // If the helpful question didn't get answered, count the 'total' as 0.
    if ((rawAggregateRating.helpful.getOrElse(0L) == 0)
      && (rawAggregateRating.notHelpful.getOrElse(0L) == 0)) {
      rawAggregateRating.copy(
        total = Some(0L),
        weightedTotal = Some(0.0)
      )
    } else {
      rawAggregateRating
    }
  }
}
