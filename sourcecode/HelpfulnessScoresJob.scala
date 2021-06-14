package com.twitter.birdwatch.scoring

import com.twitter.birdwatch.exporter.DatasetUtils
import com.twitter.birdwatch.thriftscala.BirdwatchUserReputation
import com.twitter.scalding.{Args, Days, Job, RichDate}
import com.twitter.scalding.typed.TypedPipe
import com.twitter.scalding_internal.dalv2.DALWrite.{D, ExplicitEndTime, Version, WriteExtension}
import com.twitter.scalding_internal.job.analytics_batch.AnalyticsBatchJob
import com.twitter.scalding_internal.job.{HasDateRange, TwitterUtcDateJob}
import com.twitter.scalding_internal.multiformat.format.keyval.KeyVal

trait HelpfulnessScoresJob extends Job with HasDateRange {
  import HelpfulnessScoresUtil._

  val numPartFilesToWrite = 1
  val version: Version = args
    .optional("version").map { version =>
      ExplicitEndTime(RichDate(version.toLong))
    }.getOrElse(ExplicitEndTime(RichDate.now))

  def writeHelpfulnessScores(helpfulnessScores: TypedPipe[BirdwatchUserReputation]) = {
    helpfulnessScores
      .shard(numPartFilesToWrite)
      .writeDALSnapshot(
        BirdwatchUserReputationDailyScalaDataset,
        D.Hourly,
        D.Suffix("birdwatch_user_reputation_daily"),
        D.Parquet,
        dateRange.end
      )

    helpfulnessScores
      .map { helpfulnessScores => KeyVal(helpfulnessScores.userId, helpfulnessScores) }
      .shard(numPartFilesToWrite)
      .writeDALVersionedKeyVal(
        BirdwatchUserReputationScalaDataset,
        D.Suffix("birdwatch_user_reputation"),
        version
      )
  }

  def runJob(args: Args): Unit = {
    val (ratingsWithNotes, _) = DatasetUtils.getNotesAndRatingsData()
    val rawAuthorHelpfulnessScores: TypedPipe[BirdwatchUserReputation] = {
      computeRawAuthorHelpfulnessScores(ratingsWithNotes)
    }
    val finalAuthorHelpfulnessScores = reweightAuthorHelpfulnessScoresNTimes(
      ratingsWithNotes,
      rawAuthorHelpfulnessScores,
      args.optional("maxAuthorHelpfulnessIterations").map(_.toInt)
    )

    val notesLabeledUsingAuthorHelpfulnessScores =
      labelNotesWithWeights(
        ratingsWithNotes,
        finalAuthorHelpfulnessScores,
        getAuthorHelpfulnessScore)
    val raterHelpfulnessScores =
      computeRaterHelpfulnessScores(
        ratingsWithNotes,
        notesLabeledUsingAuthorHelpfulnessScores,
        finalAuthorHelpfulnessScores)

    val combinedHelpfulnessScores = combineHelpfulnessScores(
      finalAuthorHelpfulnessScores,
      raterHelpfulnessScores
    )

    writeHelpfulnessScores(combinedHelpfulnessScores)
  }
}

/**
scalding remote run --target \
 birdwatch/scoring/src/main/scala/com/twitter/birdwatch/scoring:scoring-deploy \
 --main-class com.twitter.birdwatch.scoring.HelpfulnessScoresDateJob \
 --user birdwatch-service --submitter hadoopnest1.atla.twitter.com --cluster proc-atla \
 -- --date YYYY-MM-DD
 */
class HelpfulnessScoresDateJob(args: Args)
    extends TwitterUtcDateJob(args)
    with HelpfulnessScoresJob {
  runJob(args)
}

class HelpfulnessScoresBatchJob(args: Args)
    extends AnalyticsBatchJob(args)
    with HelpfulnessScoresJob {
  override def batchIncrement = Days(1)
  override def firstTime = RichDate("2021-06-08")
  runJob(args)
}
