package com.twitter.birdwatch.scoring

import com.twitter.algebird.Semigroup
import com.twitter.birdwatch.exporter.BirdwatchTsvConverter
import com.twitter.birdwatch.thriftscala.{
  BirdwatchNoteHelpfulTags,
  BirdwatchNoteNotHelpfulTags,
  BirdwatchNoteRating,
  BirdwatchNoteRatingDataV1Aggregate,
  BirdwatchNoteRatingDataV2Aggregate,
  BirdwatchNoteRatingHelpfulnessLevel,
  BirdwatchAggregateRating => ThriftRatingAggregate,
  VisibleContributorStats
}
import com.twitter.birdwatch.thriftscala.BirdwatchNoteRatingHelpfulnessLevel._
import org.apache.thrift.TEnum

private object CustomMath {
  def sumMaps[T](
    a: collection.Map[T, Long],
    b: collection.Map[T, Long]
  ): collection.Map[T, Long] = {
    a.foldLeft(b) {
      case (current, (key, value)) => current + (key -> { value + current.getOrElse(key, 0L) })
    }
  }

  def max(a: Option[Long], b: Option[Long]): Option[Long] = {
    (a ++ b).reduceOption(Math.max(_: Long, _: Long))
  }
}

private class V1V2CombinedAggregateRating(
  v1: Option[BirdwatchNoteRatingDataV1Aggregate] = None,
  v2: Option[BirdwatchNoteRatingDataV2Aggregate] = None) {

  private def v1HelpfulnessLevel(level: BirdwatchNoteRatingHelpfulnessLevel): Long = {
    v1.map { v =>
        level match {
          case Helpful => v.helpful
          case NotHelpful => v.notHelpful
          case SomewhatHelpful => 0L
        }
      }.getOrElse(0L)
  }

  private def v2HelpfulnessLevel(level: BirdwatchNoteRatingHelpfulnessLevel): Long = {
    v2.flatMap(_.helpfulnessLevel.get(level.value)).getOrElse(0L)
  }

  private def helpfulCount: Long = {
    v1HelpfulnessLevel(Helpful) + v2HelpfulnessLevel(Helpful)
  }

  private def somewhatHelpfulCount: Long = {
    v2HelpfulnessLevel(SomewhatHelpful)
  }

  private def notHelpfulCount: Long = {
    v1HelpfulnessLevel(NotHelpful) + v2HelpfulnessLevel(NotHelpful)
  }

  def hasHelpfulnessIndicator: Boolean = {
    helpfulCount > 0 || somewhatHelpfulCount > 0 || notHelpfulCount > 0
  }

  def helpful: Double = {
    // TODO (BWATCH-262): adjust coefficients so v1 and v2 ratings roughly have the same influence
    v1HelpfulnessLevel(Helpful) +
      v2HelpfulnessLevel(Helpful) + 0.5 * v2HelpfulnessLevel(SomewhatHelpful)
  }

  def total: Double = {
    Seq(v1.map(_.total), v2.map(_.total)).flatten.sum.toDouble
  }

  def helpfulTags: Map[BirdwatchNoteHelpfulTags, Long] = {
    (v1.map(_.helpfulTags) ++ v2.map(_.helpfulTags))
      .reduceOption { (a, b) =>
        CustomMath.sumMaps(a, b)
      }.map(_.map {
        case (k, v) => BirdwatchNoteHelpfulTags(k) -> v
      }.toMap).getOrElse(Map.empty)
  }

  def notHelpfulTags: Map[BirdwatchNoteNotHelpfulTags, Long] = {
    (v1.map(_.notHelpfulTags) ++ v2.map(_.notHelpfulTags))
      .reduceOption { (a, b) =>
        CustomMath.sumMaps(a, b)
      }.map(_.map {
        case (k, v) => BirdwatchNoteNotHelpfulTags(k) -> v
      }.toMap).getOrElse(Map.empty)
  }

  def latestRatingCreatedAt: Option[Long] = {
    (v1, v2) match {
      case (Some(a), Some(b)) =>
        CustomMath.max(a.latestRatingCreatedAt, b.latestRatingCreatedAt)
      case (Some(a), None) => a.latestRatingCreatedAt
      case (None, Some(b)) => b.latestRatingCreatedAt
      case (None, None) => None
    }
  }

  def contributorStats: VisibleContributorStats = VisibleContributorStats(
    helpfulCount = Some(helpfulCount),
    somewhatHelpfulCount = Some(somewhatHelpfulCount),
    notHelpfulCount = Some(notHelpfulCount)
  )
}

/*
    This class defines a rating-type agnostic interface that can be used within algorithms.

      Fields:
        rawHelpful: an aggregated value of helpfulness
        rawTotal: total number of ratings that were used to calculate `rawHelpful`
        weightedTotal: weighted value of helpfulness
        weightedHelpful: weighted value of helpfulness
 */
case class RatingAggregate(
  v1: Option[BirdwatchNoteRatingDataV1Aggregate] = None,
  v2: Option[BirdwatchNoteRatingDataV2Aggregate] = None,
  weightedTotal: Double = 0.0,
  weightedHelpful: Double = 0.0) {

  // Note: It's not always going to be possible to combine different ratings together. V1 and V2
  // is a special case because of their high correlation in questions asked in the user form.
  // To make this obvious, we are using a class that specifically defines how these two rating types
  // can be combined. In the future if we have new rating types, we need to either define a similar
  // set of rules that includes the new rating type, and/or deprecate old rating types.
  private val v1v2 = new V1V2CombinedAggregateRating(v1, v2)

  private def sumV1(
    a: BirdwatchNoteRatingDataV1Aggregate,
    b: BirdwatchNoteRatingDataV1Aggregate
  ) = {
    BirdwatchNoteRatingDataV1Aggregate(
      total = a.total + b.total,
      helpful = a.helpful + b.helpful,
      notHelpful = a.notHelpful + b.notHelpful,
      agree = a.agree + b.agree,
      disagree = a.disagree + b.disagree,
      helpfulTags = CustomMath.sumMaps(a.helpfulTags, b.helpfulTags),
      notHelpfulTags = CustomMath.sumMaps(a.notHelpfulTags, b.notHelpfulTags),
      latestRatingCreatedAt = CustomMath.max(a.latestRatingCreatedAt, b.latestRatingCreatedAt)
    )
  }

  private def sumV2(
    a: BirdwatchNoteRatingDataV2Aggregate,
    b: BirdwatchNoteRatingDataV2Aggregate
  ) = {
    BirdwatchNoteRatingDataV2Aggregate(
      total = a.total + b.total,
      helpfulnessLevel = CustomMath.sumMaps(a.helpfulnessLevel, b.helpfulnessLevel),
      helpfulTags = CustomMath.sumMaps(a.helpfulTags, b.helpfulTags),
      notHelpfulTags = CustomMath.sumMaps(a.notHelpfulTags, b.notHelpfulTags),
      latestRatingCreatedAt = CustomMath.max(a.latestRatingCreatedAt, b.latestRatingCreatedAt)
    )
  }

  def rawTotal: Double = v1v2.total

  def rawHelpful: Double = v1v2.helpful

  def helpfulTags: Map[BirdwatchNoteHelpfulTags, Long] = v1v2.helpfulTags

  def notHelpfulTags: Map[BirdwatchNoteNotHelpfulTags, Long] = v1v2.notHelpfulTags

  def hasHelpfulnessIndicator: Boolean = v1v2.hasHelpfulnessIndicator

  def latestRatingCreatedAt: Option[Long] = v1v2.latestRatingCreatedAt

  def contributorStats: VisibleContributorStats = v1v2.contributorStats

  def toThrift: ThriftRatingAggregate = {
    ThriftRatingAggregate(
      total = Some(rawTotal.toLong),
      helpful = contributorStats.helpfulCount,
      notHelpful = contributorStats.notHelpfulCount,
      somewhatHelpful = contributorStats.somewhatHelpfulCount,
      latestRatingCreatedAt = latestRatingCreatedAt,
      weightedTotal = Some(weightedTotal),
      weightedHelpful = Some(weightedHelpful),
      v1 = v1,
      v2 = v2
    )
  }

  def weightedCopy(
    weight: Double,
    requireHelpfulnessIndicator: Boolean = false
  ): RatingAggregate = {
    val zeroWeight = requireHelpfulnessIndicator && !hasHelpfulnessIndicator
    if (zeroWeight) {
      RatingAggregate()
    } else {
      this.copy(weightedTotal = rawTotal * weight, weightedHelpful = rawHelpful * weight)
    }
  }

  def +(that: RatingAggregate) = RatingAggregate(
    v1 = (this.v1 ++ that.v1).reduceOption(sumV1),
    v2 = (this.v2 ++ that.v2).reduceOption(sumV2),
    weightedTotal = this.weightedTotal + that.weightedTotal,
    weightedHelpful = this.weightedHelpful + that.weightedHelpful
  )
}

object RatingAggregate {
  // the semigroup defines addition of RatingAggregates
  val ratingSemigroup = new Semigroup[RatingAggregate] {
    override def plus(i: RatingAggregate, j: RatingAggregate): RatingAggregate =
      i + j
  }

  def apply(thriftNoteRating: BirdwatchNoteRating): RatingAggregate = {
    def convertTags[T <: TEnum](tagsOpt: Option[collection.Set[T]]): Map[Int, Long] = {
      tagsOpt
        .map(tags =>
          tags
            .zip(Seq.fill(tags.size)(1L))
            .toMap
            .map { case (k, v) => k.getValue -> v }).getOrElse(Map.empty)
    }
    val (
      v1: Option[BirdwatchNoteRatingDataV1Aggregate],
      v2: Option[BirdwatchNoteRatingDataV2Aggregate],
    ) = thriftNoteRating.version match {
      case None | Some(1L) =>
        (
          Some(
            BirdwatchNoteRatingDataV1Aggregate(
              total = 1L,
              // TODO (BWATCH-263): move `boolToInt` function to a common lib out of BirdwatchTsvConverter
              helpful = BirdwatchTsvConverter.boolToInt(thriftNoteRating.helpful.contains(true)),
              notHelpful =
                BirdwatchTsvConverter.boolToInt(thriftNoteRating.helpful.contains(false)),
              agree = BirdwatchTsvConverter.boolToInt(thriftNoteRating.agree.contains(true)),
              disagree = BirdwatchTsvConverter.boolToInt(thriftNoteRating.agree.contains(true)),
              helpfulTags = convertTags(thriftNoteRating.helpfulTags),
              notHelpfulTags = convertTags(thriftNoteRating.notHelpfulTags),
              latestRatingCreatedAt = Some(thriftNoteRating.createdAt)
            )
          ),
          None
        )
      case Some(2L) =>
        val helpfulnessLevel = thriftNoteRating.helpfulnessLevel
          .map(l => Map(l.value -> 1L))
          .getOrElse(Map.empty)
        (
          None,
          Some(
            BirdwatchNoteRatingDataV2Aggregate(
              total = 1L,
              helpfulnessLevel = helpfulnessLevel,
              helpfulTags = convertTags(thriftNoteRating.helpfulTags),
              notHelpfulTags = convertTags(thriftNoteRating.notHelpfulTags),
              latestRatingCreatedAt = Some(thriftNoteRating.createdAt)
            ))
        )
      case _ =>
        (None, None)
    }
    RatingAggregate(v1 = v1, v2 = v2)
  }
}