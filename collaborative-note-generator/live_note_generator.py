import concurrent.futures
import datetime
import hashlib
import json
import re
from typing import Optional

from .constants import (
  ContextForGeneration,
  Evaluation,
  GrokRejectorResult,
  LiveNoteGenerationResult,
  LiveNoteTrackingStats,
  LiveNoteVersion,
  NotificationInfo,
  RatingStatus,
  RejectionDecision,
  Source,
  Suggestion,
  SuggestionEvaluation,
  UpdateDecision,
  format_prompt_for_logging,
  format_response_for_logging,
  liveNoteClassificationInaccurate,
  liveNoteClassificationMainPointHoldsButInaccurate,
  liveNoteClassificationNoInaccuraciesFound,
  liveNoteClassificationOpinion,
  liveNoteClassificationOpinionButInaccurate,
)


# Canonical classification strings the model must emit; anything else cannot be
# persisted (maps to a null classification thrift-side), so we retry generation.
_VALID_CLASSIFICATIONS = frozenset(
  {
    liveNoteClassificationOpinion,
    liveNoteClassificationOpinionButInaccurate,
    liveNoteClassificationMainPointHoldsButInaccurate,
    liveNoteClassificationInaccurate,
    liveNoteClassificationNoInaccuraciesFound,
  }
)
from .media_pipeline import (
  check_media_comparison_pipeline_eligibility,
  generate_media_match_analysis,
)
from .notes_data_client import NotesDataClient
from .prompts import (
  build_generation_prompt,
  build_update_decider_prompt,
  get_grok_rejector_prompt,
  get_live_note_candidate_rejector_prompt,
  get_live_note_generation_prompt,
)

from llm.grok_client import GrokClient, SimpleGrokEAPIClient


def _category_is_misleading(category) -> bool:
  """True for an M / Misleading category in any form (M, MISINFORMED_..., MISLEADING)."""
  norm = str(category or "").strip().upper().replace("_", "").replace(" ", "")
  if norm in ("NM", "NOTMISLEADING"):
    return False
  return norm in ("M", "MISLEADING") or "MISLEAD" in norm or "MISINFORM" in norm


def _category_is_not_misleading(category) -> bool:
  norm = str(category or "").strip().upper().replace("_", "").replace(" ", "")
  return norm in ("NM", "NOTMISLEADING")


def _is_m_to_nm_transition(context, new_live_note_result) -> bool:
  """True when the previous published version was M and the new candidate is NM."""
  versions = context.past_live_note_versions_with_suggestions
  if not versions:
    return False
  return _category_is_misleading(versions[0].category) and _category_is_not_misleading(
    new_live_note_result.category
  )


class LiveNoteGenerator:
  def __init__(
    self,
    logger,
    llm_client: GrokClient = None,
    notes_data_client: NotesDataClient = None,
    max_retries: int = 10,
    media_eligibility_llm_client: GrokClient = None,
    rejector_llm_client: Optional[GrokClient] = None,
    generation_llm_client: Optional[GrokClient] = None,
  ):
    self.logger = logger
    if llm_client is None:
      llm_client = SimpleGrokEAPIClient()
    self.llm_client = llm_client
    # Client used for the live note generation step (initial candidate and
    # media-pipeline regeneration). Defaults to the main client when not set.
    self.generation_llm_client = generation_llm_client or self.llm_client
    self.media_eligibility_llm_client = media_eligibility_llm_client or self.llm_client
    if notes_data_client is None:
      raise ValueError("notes_data_client must be provided")
    self.notes_data_client = notes_data_client
    self.max_retries = max_retries
    self.rejector_llm_client = rejector_llm_client
    self._set_version_info_for_model_and_prompt()

  def _set_version_info_for_model_and_prompt(self):
    prompt_base = get_live_note_generation_prompt(
      ContextForGeneration(
        tweet_id="tweet_id",
        note_contents=[],
        past_live_note_versions_with_suggestions=[],
      ),
    )
    version_info_str = f"{self.generation_llm_client.get_model_info()}/{prompt_base}"
    self.version_info = hashlib.sha256(version_info_str.encode("utf-8")).hexdigest()

  def hydrate_context_for_tweet(
    self,
    tweet_id,
    tracking_stats: Optional[LiveNoteTrackingStats] = None,
    include_suggestions: bool = True,
  ) -> ContextForGeneration:
    self._increment_stat(tracking_stats, "hydrate_context.attempts")

    live_note_version_id = self.notes_data_client.get_new_snowflake_id()
    if include_suggestions:
      past_live_note_versions_with_suggestions = (
        self.notes_data_client.get_hydrated_previous_live_note_versions(tweet_id)
      )
    else:
      past_live_note_versions_with_suggestions = (
        self.notes_data_client.get_previous_live_note_versions(tweet_id)
      )
    note_contents = self.notes_data_client.get_note_contents(tweet_id)
    try:
      note_request_and_reply_info = self.notes_data_client.get_note_request_and_reply_info(tweet_id)
    except Exception as e:
      self.logger.warning(
        f"Failed to fetch note request and reply info for tweet {tweet_id} (non-fatal): {e}"
      )
      note_request_and_reply_info = None
    self._increment_stat(tracking_stats, "hydrate_context.successes")
    return ContextForGeneration(
      tweet_id=tweet_id,
      live_note_version_id=live_note_version_id,
      note_contents=note_contents,
      past_live_note_versions_with_suggestions=past_live_note_versions_with_suggestions,
      note_request_and_reply_info=note_request_and_reply_info,
    )

  def generate_candidate_live_note(
    self,
    context: ContextForGeneration,
    tracking_stats: Optional[LiveNoteTrackingStats] = None,
  ) -> Optional[LiveNoteVersion]:
    self._increment_stat(tracking_stats, "generate_candidate.attempts")
    grok_live_note_result = self.sample_live_note(context=context, tracking_stats=tracking_stats)

    if grok_live_note_result is None:
      self._increment_stat(tracking_stats, "generate_candidate.failures")
      return None
    grok_live_note_result.version_id = context.live_note_version_id
    grok_live_note_result.created_at_ms = int(1000 * datetime.datetime.now().timestamp())

    if len(context.past_live_note_versions_with_suggestions) > 0:
      grok_live_note_result.suggestions = context.past_live_note_versions_with_suggestions[
        0
      ].suggestions

    self._increment_stat(tracking_stats, "generate_candidate.successes")
    return grok_live_note_result

  def check_if_post_has_few_enough_previous_crnh_live_note_versions(
    self,
    context: ContextForGeneration,
    tracking_stats: LiveNoteTrackingStats,
    max_previous_crnh_live_note_versions: int = 1,
  ) -> bool:
    past_crnh_live_note_versions = 0
    for live_note_version in context.past_live_note_versions_with_suggestions:
      if live_note_version.scoring_result is not None:
        if live_note_version.scoring_result.status == RatingStatus.CurrentlyRatedNotHelpful:
          past_crnh_live_note_versions += 1
    self.logger.info(
      f"Post {context.tweet_id} has {past_crnh_live_note_versions} previous CRNH live note versions. Max allowed: {max_previous_crnh_live_note_versions}"
    )
    eligible = past_crnh_live_note_versions <= max_previous_crnh_live_note_versions
    if eligible:
      self._increment_stat(tracking_stats, "check_eligibility.successes")
    else:
      self._increment_stat(tracking_stats, "check_eligibility.abort_too_many_previous_crnh_on_post")
    return eligible

  def _increment_stat(
    self,
    tracking_stats: Optional[LiveNoteTrackingStats],
    stat_key: str,
    count: int = 1,
  ) -> None:
    if tracking_stats is None:
      return
    tracking_stats.generator_stats[stat_key] = (
      tracking_stats.generator_stats.get(stat_key, 0) + count
    )

  def initialize_tracking_stats(self) -> LiveNoteTrackingStats:
    return LiveNoteTrackingStats(
      generator_stats={},
      tracking_start_ms=int(1000 * datetime.datetime.now().timestamp()),
    )

  def end_tracking_stats(
    self,
    tracking_stats: LiveNoteTrackingStats,
    failure_reason: Optional[str] = None,
    intended_failure: bool = False,
  ) -> LiveNoteTrackingStats:
    tracking_stats.tracking_end_ms = int(1000 * datetime.datetime.now().timestamp())
    if failure_reason is not None:
      tracking_stats.generator_failure = failure_reason
    tracking_stats.intended_failure = intended_failure
    return tracking_stats

  def generate_live_note(
    self,
    tweet_id,
    include_suggestions: bool = True,
    enable_media_pipeline: bool = True,
    num_rejector_samples: int = 0,
  ) -> LiveNoteGenerationResult:
    self.logger.info(f"Generating live note for tweet {tweet_id}")

    tracking_stats = self.initialize_tracking_stats()

    # Hydration touches external stores (Strato/MH) and previously could raise an
    # uncaught exception that escaped the generation future entirely -- leaving the
    # completion callback with no tracking_stats ("no generator stats"). Catch it
    # and return a proper failure result so the failure is recorded with stats.
    try:
      context = self.hydrate_context_for_tweet(
        tweet_id,
        tracking_stats=tracking_stats,
        include_suggestions=include_suggestions,
      )
    except Exception as e:
      self.logger.error(
        f"Error hydrating context for tweet {tweet_id}: {e}. Skipping generation.",
        exc_info=True,
      )
      return LiveNoteGenerationResult(
        live_note_version=None,
        tracking_stats=self.end_tracking_stats(tracking_stats, f"Error hydrating context: {e}"),
      )

    if not self.check_if_post_has_few_enough_previous_crnh_live_note_versions(
      context, tracking_stats=tracking_stats
    ):
      self.logger.info(
        f"Post {context.tweet_id} has too many previous CRNH live note versions. Skipping generation."
      )
      return LiveNoteGenerationResult(
        live_note_version=None,
        tracking_stats=self.end_tracking_stats(
          tracking_stats,
          "Post has too many previous CRNH live note versions.",
          intended_failure=True,
        ),
      )

    new_live_note_version = self.generate_candidate_live_note(
      context=context, tracking_stats=tracking_stats
    )
    if new_live_note_version is None:
      self.logger.info(
        f"Error generating candidate live note for post {context.tweet_id}. Skipping generation."
      )
      return LiveNoteGenerationResult(
        live_note_version=None,
        tracking_stats=self.end_tracking_stats(
          tracking_stats, "Error generating candidate live note."
        ),
      )

    if enable_media_pipeline:
      new_live_note_version = self.media_comparison_pipeline(
        context=context,
        new_live_note_version=new_live_note_version,
        tracking_stats=tracking_stats,
      )

    new_live_note_version.num_rejector_samples = num_rejector_samples

    # Run grok rejector evaluation and quality rejection in parallel
    run_grok_rejector = (
      new_live_note_version.num_rejector_samples > 0 and self.rejector_llm_client is not None
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      rejection_future = executor.submit(
        self.decide_whether_to_reject, context, new_live_note_version, tracking_stats
      )
      grok_rejector_future = (
        executor.submit(self.query_grok_rejector, context, new_live_note_version, tracking_stats)
        if run_grok_rejector
        else None
      )
      new_live_note_version.rejection_decision = rejection_future.result()
      if grok_rejector_future is not None:
        try:
          new_live_note_version.evaluation = grok_rejector_future.result()
        except Exception as e:
          self.logger.warning(
            f"Grok rejector future failed for post {context.tweet_id} (non-fatal): {e}"
          )
          new_live_note_version.evaluation = Evaluation()
    if new_live_note_version.rejection_decision.should_reject:
      new_live_note_version.update_decision = UpdateDecision(
        should_update=False,
        update_explanation="Rejected by rejector.",
        difference_from_previous="Rejected by rejector.",
      )
    else:
      new_live_note_version.update_decision = self.decide_whether_to_update(
        context, new_live_note_version, tracking_stats=tracking_stats
      )

    if new_live_note_version.update_decision.should_update:
      new_live_note_version.notifications = self.determine_notifications(
        context, new_live_note_version, tracking_stats=tracking_stats
      )
    else:
      new_live_note_version.notifications = NotificationInfo(
        users_who_added_suggestions=[], users_whose_suggestions_were_accepted=[]
      )
    return LiveNoteGenerationResult(
      live_note_version=new_live_note_version,
      tracking_stats=self.end_tracking_stats(tracking_stats),
    )

  def get_newly_added_suggestions(
    self,
    context: ContextForGeneration,
  ) -> list[Suggestion]:
    """
    Returns suggestions that were added for the first time on the previous most recent version.
    Intended to be used to determine users to notify for added suggestions, although it should
    only be called when a new version is being published.
    """
    if len(context.past_live_note_versions_with_suggestions) == 0:
      return []
    elif len(context.past_live_note_versions_with_suggestions) == 1:
      return context.past_live_note_versions_with_suggestions[0].suggestions
    else:
      all_suggestions = context.past_live_note_versions_with_suggestions[0].suggestions
      previous_suggestions = context.past_live_note_versions_with_suggestions[1].suggestions
      previous_suggestions_by_id = {
        suggestion.suggestion_id: suggestion for suggestion in previous_suggestions
      }
      return [
        suggestion
        for suggestion in all_suggestions
        if suggestion.suggestion_id not in previous_suggestions_by_id
      ]

  def get_newly_accepted_suggestions(
    self,
    context: ContextForGeneration,
    new_live_note_result: LiveNoteVersion,
  ) -> list[Suggestion]:
    # Get suggestions that were accepted for the first time this version.
    current_accepted_suggestions = []
    if new_live_note_result.suggestion_evaluations is not None:
      for suggestion in new_live_note_result.suggestions:
        if suggestion.suggestion_id in new_live_note_result.suggestion_evaluations:
          if new_live_note_result.suggestion_evaluations[suggestion.suggestion_id].is_incorporated:
            current_accepted_suggestions.append(suggestion)

    # Filter out suggestions that were accepted in a previous version.
    newly_accepted_suggestions = []
    for suggestion in current_accepted_suggestions:
      for previous_version in context.past_live_note_versions_with_suggestions:
        if previous_version.suggestion_evaluations is None:
          continue
        if suggestion.suggestion_id in previous_version.suggestion_evaluations:
          if previous_version.suggestion_evaluations[suggestion.suggestion_id].is_incorporated:
            break
      else:
        newly_accepted_suggestions.append(suggestion)

    return newly_accepted_suggestions

  def determine_notifications(
    self,
    context: ContextForGeneration,
    new_live_note_result: LiveNoteVersion,
    tracking_stats: LiveNoteTrackingStats,
  ) -> NotificationInfo:
    # TODO: update logic to fix any edge cases involving non-published versions once new devStore is live.
    self._increment_stat(tracking_stats, "determine_notifications.attempts")

    if new_live_note_result.suggestions is None:
      result = NotificationInfo(
        users_who_added_suggestions=[], users_whose_suggestions_were_accepted=[]
      )
      self._increment_stat(tracking_stats, "determine_notifications.successes")
      return result

    newly_accepted_suggestions = self.get_newly_accepted_suggestions(context, new_live_note_result)
    unique_users_whose_suggestions_were_accepted = set(
      [suggestion.user_id for suggestion in newly_accepted_suggestions]
    )

    newly_added_suggestions = self.get_newly_added_suggestions(context)
    suggestion_evaluations = new_live_note_result.suggestion_evaluations or {}
    unique_users_who_added_rejected_suggestions = set(
      [
        suggestion.user_id
        for suggestion in newly_added_suggestions
        if suggestion.suggestion_id not in suggestion_evaluations
        or not suggestion_evaluations[suggestion.suggestion_id].is_incorporated
      ]
    )
    unique_users_for_added_notification = (
      unique_users_who_added_rejected_suggestions - unique_users_whose_suggestions_were_accepted
    )

    result = NotificationInfo(
      users_who_added_suggestions=list(unique_users_for_added_notification),
      users_whose_suggestions_were_accepted=list(unique_users_whose_suggestions_were_accepted),
    )
    self._increment_stat(
      tracking_stats,
      "determine_notifications.num_users_for_added_suggestion_notifications",
      len(unique_users_for_added_notification),
    )
    self._increment_stat(tracking_stats, "determine_notifications.successes")
    return result

  def decide_whether_to_update(
    self,
    context: ContextForGeneration,
    new_live_note_result: LiveNoteVersion,
    tracking_stats: Optional[LiveNoteTrackingStats] = None,
  ) -> UpdateDecision:
    local_tracking_stats = tracking_stats or self.initialize_tracking_stats()
    self._increment_stat(local_tracking_stats, "decide_update.attempts")
    if len(context.past_live_note_versions_with_suggestions) == 0:
      self._increment_stat(local_tracking_stats, "decide_update.successes")
      self._increment_stat(local_tracking_stats, "decide_update.accepted")
      return UpdateDecision(
        should_update=True,
        update_explanation="Initial version.",
        difference_from_previous="Initial version.",
      )
    previous_published_version_id = context.past_live_note_versions_with_suggestions[0].version_id

    # Hard gate: if story assessment says SAME_STORY, bypass the update decider
    # entirely — unless the generator incorporated a suggestion with few ratings
    assessment = (new_live_note_result.story_assessment or "").strip()
    assessment_upper = assessment.upper()
    story_unchanged = assessment_upper.startswith("NO_NEW_INFO") or assessment_upper.startswith(
      "SAME_STORY"
    )
    if story_unchanged and assessment:
      if _is_m_to_nm_transition(context, new_live_note_result):
        # Switching an over-classified M note to NM is valuable even with no new
        # info, so don't let the SAME_STORY/NO_NEW_INFO gate block it -- fall
        # through to the normal update decider below.
        self.logger.info(
          f"M->NM transition for post {context.tweet_id} -- bypassing SAME_STORY hard gate."
        )
        self._increment_stat(local_tracking_stats, "decide_update.m_to_nm_bypass")
      elif self._should_bypass_hard_gate_for_suggestion(context, new_live_note_result):
        self.logger.info(
          f"Suggestion bypass: story assessment says SAME_STORY for post {context.tweet_id} "
          f"but incorporated suggestion — letting decider decide."
        )
        self._increment_stat(local_tracking_stats, "decide_update.suggestion_bypass")
      else:
        self.logger.info(
          f"Hard gate: story assessment says SAME_STORY for post {context.tweet_id} — forcing NO_UPDATE."
        )
        self._increment_stat(local_tracking_stats, "decide_update.hard_gated")
        result = UpdateDecision(
          should_update=False,
          update_explanation="Hard-gated: generator self-assessment reported NO_NEW_INFO.",
          difference_from_previous="Generator found no substantively new information.",
        )
        result.previous_published_version_id = previous_published_version_id
        return result

    update_decision = self.sample_update_decision(
      context,
      new_live_note_result,
      tracking_stats=local_tracking_stats,
    )
    if update_decision is None:
      self._increment_stat(local_tracking_stats, "decide_update.failures")
      update_decision = UpdateDecision(
        should_update=False,
        update_explanation="Error getting update decision",
        difference_from_previous="Error getting difference explanation",
      )
    else:
      self._increment_stat(local_tracking_stats, "decide_update.successes")
    if update_decision.should_update:
      self._increment_stat(local_tracking_stats, "decide_update.accepted")
    else:
      self._increment_stat(local_tracking_stats, "decide_update.rejected")
    update_decision.previous_published_version_id = previous_published_version_id
    return update_decision

  def _should_bypass_hard_gate_for_suggestion(self, context, new_live_note_result) -> bool:
    """Bypass the hard gate when a suggestion was incorporated and there are few ratings."""
    if not new_live_note_result.suggestion_evaluations:
      return False
    has_incorporated = any(
      ev.is_incorporated
      for ev in new_live_note_result.suggestion_evaluations.values()
      if ev is not None
    )
    if not has_incorporated:
      return False
    previous_version = context.past_live_note_versions_with_suggestions[0]
    return (previous_version.total_ratings or 0) <= 1

  def decide_whether_to_reject(
    self,
    context: ContextForGeneration,
    new_live_note_result: LiveNoteVersion,
    tracking_stats: Optional[LiveNoteTrackingStats] = None,
  ) -> RejectionDecision:
    local_tracking_stats = tracking_stats or self.initialize_tracking_stats()
    self._increment_stat(local_tracking_stats, "decide_reject.attempts")
    rejection_decision = self.sample_rejection_decision(
      context, new_live_note_result, tracking_stats=local_tracking_stats
    )
    if rejection_decision is None:
      self._increment_stat(local_tracking_stats, "decide_reject.failures")
      return RejectionDecision(
        should_reject=True,
        rejection_reason="Error getting rejection decision",
        retryable=True,
      )
    if rejection_decision.should_reject:
      self._increment_stat(local_tracking_stats, "decide_reject.rejected")
    else:
      self._increment_stat(local_tracking_stats, "decide_reject.accepted")
    if rejection_decision.retryable:
      self._increment_stat(local_tracking_stats, "decide_reject.retryable")
    self._increment_stat(local_tracking_stats, "decide_reject.successes")
    return rejection_decision

  def sample_rejection_decision(
    self,
    context: ContextForGeneration,
    new_live_note_result: LiveNoteVersion,
    tracking_stats: LiveNoteTrackingStats,
  ) -> Optional[RejectionDecision]:
    retries = 0
    while retries < self.max_retries:
      try:
        self._increment_stat(tracking_stats, "decide_reject.llm_call.attempts")
        prompt = get_live_note_candidate_rejector_prompt(new_live_note_result)
        self.logger.info(f"Getting Grok rejection decision for post {context.tweet_id}")
        grok_response = self.llm_client.call(prompt)
        result = parse_answer_from_grok_reject_response(grok_response)
        self._increment_stat(tracking_stats, "decide_reject.llm_call.successes")
        return result
      except Exception as e:
        self._increment_stat(tracking_stats, "decide_reject.llm_call.failures")
        self._increment_stat(tracking_stats, "decide_reject.llm_call.exceptions")
        if isinstance(e, ValueError):
          self._increment_stat(tracking_stats, "decide_reject.llm_call.parse_errors")
        self.logger.error(
          f"Error getting rejection decision for post {context.tweet_id}: {e}. Retries left: {self.max_retries - retries}",
          exc_info=True,
        )
        retries += 1
    self.logger.info(
      f"Failed to generate live note for post {context.tweet_id} after {self.max_retries} retries. Returning None."
    )
    return None

  def _query_single_grok_rejector(
    self,
    tweet_id: str,
    proposed_note: str,
    tracking_stats: Optional[LiveNoteTrackingStats],
  ) -> GrokRejectorResult:
    """Query the grok rejector model once and parse the score.

    Uses a single attempt (no retries beyond the LLM client's own retry).
    Returns a result with error set on failure — never raises.
    """
    self._increment_stat(tracking_stats, "grok_rejector.llm_call.attempts")
    result = GrokRejectorResult()
    try:
      prompt = get_grok_rejector_prompt(tweet_id, proposed_note)
      response = self.rejector_llm_client.call(prompt, temperature=1.0)
      if response is None:
        raise ValueError("Grok rejector returned None")
      parsed = json.loads(response, strict=False)
      result.score = float(parsed["score"])
      result.reasoning = parsed.get("reasoning")
      self._increment_stat(tracking_stats, "grok_rejector.llm_call.successes")
    except Exception as e:
      self.logger.warning(
        f"Grok rejector query failed for post {tweet_id} (non-fatal): {type(e).__name__}: {str(e)[:200]}"
      )
      result.error = f"{type(e).__name__}: {e}"
      self._increment_stat(tracking_stats, "grok_rejector.llm_call.failures")
    return result

  def query_grok_rejector(
    self,
    context: ContextForGeneration,
    new_live_note_result: LiveNoteVersion,
    tracking_stats: Optional[LiveNoteTrackingStats] = None,
  ) -> Evaluation:
    """Call the grok rejector model N times in parallel and average scores.

    Fully fault-tolerant: never raises, never delays the critical path.
    Returns Evaluation with whatever scores succeeded; None mean if all failed.
    """
    num_samples = new_live_note_result.num_rejector_samples
    if num_samples <= 0 or self.rejector_llm_client is None:
      return Evaluation()

    self._increment_stat(tracking_stats, "grok_rejector.attempts")
    self.logger.info(
      f"Querying grok rejector for post {context.tweet_id} with {num_samples} samples"
    )

    try:
      with concurrent.futures.ThreadPoolExecutor(max_workers=num_samples) as executor:
        futures = [
          executor.submit(
            self._query_single_grok_rejector,
            context.tweet_id,
            new_live_note_result.short_live_note,
            tracking_stats,
          )
          for _ in range(num_samples)
        ]
        sample_results = [f.result() for f in futures]
    except Exception as e:
      self.logger.warning(
        f"Grok rejector executor failed for post {context.tweet_id} (non-fatal): {e}"
      )
      self._increment_stat(tracking_stats, "grok_rejector.executor_failures")
      return Evaluation()

    # Mean score from whatever succeeded — drop Nones
    scores = [r.score for r in sample_results if r.score is not None]
    mean_score = sum(scores) / len(scores) if scores else None

    errors = [r.error for r in sample_results if r.error]
    self.logger.info(
      f"[GROK_REJECTOR_SUMMARY] post={context.tweet_id} "
      f"mean_score={mean_score} "
      f"scores={[r.score for r in sample_results]} "
      f"errors={errors}"
    )
    self._increment_stat(tracking_stats, "grok_rejector.successes")
    if errors:
      self._increment_stat(tracking_stats, "grok_rejector.samples_with_errors", len(errors))

    return Evaluation(
      grok_rejector_results=sample_results,
      mean_grok_rejector_score=mean_score,
    )

  def sample_update_decision(
    self,
    context: ContextForGeneration,
    new_live_note_result: LiveNoteVersion,
    tracking_stats: LiveNoteTrackingStats,
  ) -> Optional[UpdateDecision]:
    retries = 0
    while retries < self.max_retries:
      try:
        self._increment_stat(tracking_stats, "decide_update.llm_call.attempts")
        prompt = build_update_decider_prompt(context, new_live_note_result)
        self.logger.info(
          f"Getting Grok update decision for post {context.tweet_id}. Prompt:\n{format_prompt_for_logging(prompt)}"
        )
        grok_response = self.llm_client.call(prompt)
        self.logger.info(
          f"Grok response for update decision for post {context.tweet_id}:\n{format_response_for_logging(grok_response)}"
        )
        result = parse_answer_from_grok_update_decision_response(grok_response)
        self._increment_stat(tracking_stats, "decide_update.llm_call.successes")
        return result
      except Exception as e:
        self._increment_stat(tracking_stats, "decide_update.llm_call.failures")
        self._increment_stat(tracking_stats, "decide_update.llm_call.exceptions")
        if isinstance(e, ValueError):
          self._increment_stat(tracking_stats, "decide_update.llm_call.parse_errors")
        self.logger.error(
          f"Error getting update decision for post {context.tweet_id}: {e}. Retries left: {self.max_retries - retries}",
          exc_info=True,
        )
        retries += 1
    self.logger.info(
      f"Failed to generate live note for post {context.tweet_id} after {self.max_retries} retries. Returning None."
    )
    return None

  def media_comparison_pipeline(
    self,
    context: ContextForGeneration,
    new_live_note_version: LiveNoteVersion,
    tracking_stats: LiveNoteTrackingStats = None,
    include_citation_urls: bool = False,
  ) -> LiveNoteVersion:
    """Run the media comparison pipeline and regenerate with comparison context."""
    source_urls = []
    for s in new_live_note_version.parsed_sources or []:
      if not s.url:
        continue
      if not include_citation_urls and getattr(s, "source_type", None) == "grok_citation":
        continue
      source_urls.append(s.url)
    if not source_urls:
      return new_live_note_version

    # Step 1: Check if media comparison pipeline is needed
    self._increment_stat(tracking_stats, "media_pipeline.filter.attempts")
    try:
      should_run, explanation = check_media_comparison_pipeline_eligibility(
        self.logger,
        self.media_eligibility_llm_client,
        context.tweet_id,
        new_live_note_version.short_live_note,
        source_urls,
      )
    except RuntimeError as e:
      self.logger.error(f"Media filter failed for post {context.tweet_id}, skipping pipeline: {e}")
      self._increment_stat(tracking_stats, "media_pipeline.filter.failures")
      return new_live_note_version
    self._increment_stat(tracking_stats, "media_pipeline.filter.successes")
    self.logger.info(
      f"Media filter for post {context.tweet_id}: should_run={should_run}, "
      f"explanation={explanation}"
    )
    if not should_run:
      return new_live_note_version
    self._increment_stat(tracking_stats, "media_pipeline.filter.triggered")

    # Step 2: Run media comparison (all URLs analyzed together)
    self._increment_stat(tracking_stats, "media_pipeline.comparison.attempts")
    per_url_results = generate_media_match_analysis(
      self.logger,
      self.llm_client,
      context.tweet_id,
      source_urls,
    )
    self._increment_stat(tracking_stats, "media_pipeline.comparison.successes")
    per_url_summary = ", ".join(f"{r.url}: {r.same_media.value}" for r in per_url_results)
    self.logger.info(
      f"Media comparison result for post {context.tweet_id}: "
      f"{len(per_url_results)} URLs analyzed [{per_url_summary}]"
    )

    # Step 3: Regenerate with media comparison context
    self._increment_stat(tracking_stats, "media_pipeline.regeneration.attempts")
    context_with_media = ContextForGeneration(
      tweet_id=context.tweet_id,
      note_contents=context.note_contents,
      past_live_note_versions_with_suggestions=context.past_live_note_versions_with_suggestions,
      live_note_version_id=context.live_note_version_id,
      media_comparison_results=per_url_results,
    )

    regenerated = self.sample_live_note(
      context_with_media, tracking_stats, "generate_candidate_with_media_context"
    )
    if regenerated is None:
      self._increment_stat(tracking_stats, "media_pipeline.regeneration.failures")
      return new_live_note_version

    self._increment_stat(tracking_stats, "media_pipeline.regeneration.successes")
    regenerated.version_id = new_live_note_version.version_id
    regenerated.created_at_ms = new_live_note_version.created_at_ms
    regenerated.suggestions = new_live_note_version.suggestions
    return regenerated

  def sample_live_note(
    self,
    context: ContextForGeneration,
    tracking_stats: LiveNoteTrackingStats = None,
    stat_prefix: str = "generate_candidate",
  ) -> Optional[LiveNoteVersion]:
    retries = 0
    while retries < self.max_retries:
      try:
        self._increment_stat(tracking_stats, f"{stat_prefix}.llm_call.attempts")
        prompt = build_generation_prompt(context)
        self.logger.info(
          f"Getting Grok draft live note generation for post {context.tweet_id}. Prompt:\n{format_prompt_for_logging(prompt)}"
        )
        full_resp = self.generation_llm_client.call(prompt, full_response=True)
        grok_response, citation_urls = _extract_text_and_citations(full_resp)
        self.logger.info(
          f"Grok response for live note generation for post {context.tweet_id}:\n{format_response_for_logging(grok_response)}"
        )

        try:
          result = parse_json_generation_response(grok_response, self.logger)
        except ValueError:
          self._increment_stat(tracking_stats, f"{stat_prefix}.llm_call.parse_errors")
          self._increment_stat(tracking_stats, f"{stat_prefix}.llm_call.failures")
          retries += 1
          continue

        # The classification must be one of the canonical values, otherwise it
        # cannot be persisted (it maps to a null classification thrift-side).
        # Strip whitespace; if it still isn't valid, retry the generation rather
        # than publishing a note with an unrecognized/non-persistable classification.
        normalized_classification = (result.live_note_classification or "").strip()
        if normalized_classification not in _VALID_CLASSIFICATIONS:
          self.logger.warning(
            f"Invalid classification {result.live_note_classification!r} for post "
            f"{context.tweet_id}; retrying generation. Retries left: {self.max_retries - retries}"
          )
          self._increment_stat(tracking_stats, f"{stat_prefix}.llm_call.invalid_classification")
          self._increment_stat(tracking_stats, f"{stat_prefix}.llm_call.failures")
          retries += 1
          continue
        result.live_note_classification = normalized_classification

        # Merge citation URLs into parsed_sources
        _merge_citation_urls(result, citation_urls)

        # JSON parser extracts story_assessment from the JSON payload; fall
        # back to XML tag extraction for responses that used XML format.
        if not result.story_assessment:
          story_assessment_match = re.search(
            r"<STORY_ASSESSMENT>(.*?)</STORY_ASSESSMENT>", grok_response, re.DOTALL
          )
          if story_assessment_match:
            result.story_assessment = story_assessment_match.group(1).strip()

        if not self.notes_data_client.check_note_character_limit(result.short_live_note):
          self._increment_stat(
            tracking_stats,
            f"{stat_prefix}.generated_short_live_note_exceeds_character_limit",
          )
          retries += 1
          continue

        self._increment_stat(tracking_stats, f"{stat_prefix}.llm_call.successes")
        return result
      except Exception as e:
        self._increment_stat(tracking_stats, f"{stat_prefix}.llm_call.failures")
        self._increment_stat(tracking_stats, f"{stat_prefix}.llm_call.exceptions")
        if isinstance(e, ValueError):
          self._increment_stat(tracking_stats, f"{stat_prefix}.llm_call.parse_errors")
        self.logger.error(
          f"Error generating live note for post {context.tweet_id}: {e}. Retries left: {self.max_retries - retries}",
          exc_info=True,
        )
      retries += 1
    self.logger.info(
      f"Failed to generate live note for post {context.tweet_id} after {self.max_retries} retries. Returning None."
    )
    return None


def parse_json_generation_response(response, logger):
  """Parse a JSON generation response into a LiveNoteVersion.

  Expects raw JSON (or JSON with code fences stripped as a safety fallback).
  Raises ValueError if JSON parsing fails.
  """
  text = response.strip()

  # Strip markdown code fences
  if "```json" in text:
    start = text.find("```json") + 7
    end = text.find("```", start)
    if end > start:
      text = text[start:end].strip()
  elif "```" in text:
    start = text.find("```") + 3
    end = text.find("```", start)
    if end > start:
      text = text[start:end].strip()

  # strict=False tolerates literal control characters (e.g. the blank lines the
  # prompt instructs the model to put in "detail"), which are invalid in strict
  # JSON strings and were the dominant cause of generation parse failures.
  try:
    data = json.loads(text, strict=False)
  except json.JSONDecodeError:
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
      try:
        data = json.loads(json_match.group(), strict=False)
      except json.JSONDecodeError:
        raise ValueError("Failed to parse JSON from generation response")
    else:
      raise ValueError("No JSON object found in generation response")

  classification = data.get("classification", "")
  category = data.get("category", "")
  proposed_note = data.get("proposed_note", "")
  detail = data.get("detail", "")
  sources_considered_raw = data.get("sources_considered", [])
  story_assessment = data.get("story_assessment")
  suggestion_explanations_raw = data.get("suggestion_explanations")

  if not proposed_note:
    raise ValueError("JSON response missing 'proposed_note'")
  if not classification:
    raise ValueError("JSON response missing 'classification'")
  if not category:
    raise ValueError("JSON response missing 'category'")

  sources_considered_str = (
    json.dumps(sources_considered_raw, indent=2) if sources_considered_raw else "[]"
  )
  parsed_sources = _parse_sources_from_json(sources_considered_raw)

  suggestion_evaluations = {}
  if suggestion_explanations_raw and isinstance(suggestion_explanations_raw, list):
    for item in suggestion_explanations_raw:
      if not isinstance(item, dict):
        continue
      raw_sid = item.get("suggestion_id")
      if raw_sid is None:
        continue
      try:
        sid = int(raw_sid)
        status = item.get("incorporated_status", "NO")
        suggestion_evaluations[sid] = SuggestionEvaluation(
          is_incorporated=status in ("YES", "PARTIALLY"),
          incorporated_status=status,
          incorporated_explanation=item.get("incorporated_explanation"),
          decision_explanation=item.get("decision_explanation"),
        )
      except (KeyError, ValueError, TypeError):
        continue

  return LiveNoteVersion(
    live_note_classification=classification,
    category=category,
    short_live_note=proposed_note,
    long_live_note=detail,
    sources_considered=sources_considered_str,
    suggestion_evaluations=suggestion_evaluations if suggestion_evaluations else {},
    parsed_sources=parsed_sources,
    story_assessment=story_assessment,
  )


def _parse_sources_from_json(sources_list):
  """Parse sources from JSON array into Source objects."""
  if not sources_list or not isinstance(sources_list, list):
    return []
  sources = []
  for item in sources_list:
    if not isinstance(item, dict):
      continue
    sources.append(
      Source(
        url=item.get("url"),
        explanation=item.get("summary_and_impact_on_analysis"),
        source_type=item.get("source_type"),
        source_detail=item.get("source_detail"),
      )
    )
  return sources


def _parse_json_response(response: str) -> dict:
  """Extract and parse a JSON object from an LLM response.

  Handles responses with ```json code fences, bare JSON, or JSON embedded
  in other text.
  """
  text = response.strip()

  if "```json" in text:
    start = text.find("```json") + 7
    end = text.find("```", start)
    if end > start:
      text = text[start:end].strip()
  elif "```" in text:
    start = text.find("```") + 3
    end = text.find("```", start)
    if end > start:
      text = text[start:end].strip()

  try:
    return json.loads(text, strict=False)
  except json.JSONDecodeError:
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
      return json.loads(json_match.group(), strict=False)
    raise ValueError("No JSON object found in response")


def parse_answer_from_grok_update_decision_response(response: str) -> UpdateDecision:
  data = _parse_json_response(response)
  should_update_str = data.get("should_update", "NO")
  return UpdateDecision(
    should_update=(should_update_str == "YES"),
    update_explanation=data.get("update_explanation", ""),
    difference_from_previous=data.get("difference_explanation", ""),
  )


def parse_answer_from_grok_reject_response(response: str) -> RejectionDecision:
  data = _parse_json_response(response)
  should_reject_str = data.get("reject", "YES")
  retryable_str = data.get("retryable", "NO")
  if should_reject_str not in ("YES", "NO"):
    raise ValueError(f"Invalid reject value: {should_reject_str} (must be YES or NO)")
  if retryable_str not in ("YES", "NO"):
    raise ValueError(f"Invalid retryable value: {retryable_str} (must be YES or NO)")
  return RejectionDecision(
    should_reject=should_reject_str == "YES",
    rejection_reason=data.get("reject_reason", ""),
  )


def _extract_text_and_citations(full_resp) -> tuple[str, list[str]]:
  """Extract response text and citation URLs from a full Grok API response.

  Handles both dict (full_response=True) and str (legacy) return types.
  Citations are in output[-1]["content"][0]["annotations"] as
  {"type": "url_citation", "url": "..."} objects.
  """
  if isinstance(full_resp, str):
    return full_resp, []
  try:
    content_block = full_resp["output"][-1]["content"][0]
    text = content_block["text"]
  except (KeyError, IndexError, TypeError):
    return str(full_resp), []
  annotations = content_block.get("annotations", []) or []
  citation_urls = [
    a["url"] for a in annotations if a.get("type") == "url_citation" and a.get("url")
  ]
  return text, citation_urls


def _merge_citation_urls(result: LiveNoteVersion, citation_urls: list[str]) -> None:
  """Add citation URLs to parsed_sources, skipping any already present."""
  if not citation_urls:
    return
  if result.parsed_sources is None:
    result.parsed_sources = []
  existing_urls = {s.url for s in result.parsed_sources if s.url}
  for url in citation_urls:
    if url and url not in existing_urls:
      result.parsed_sources.append(Source(url=url, source_type="grok_citation"))
      existing_urls.add(url)
