import datetime
import hashlib
import json
import re
import textwrap
from typing import Optional

from .constants import (
  ContextForGeneration,
  LiveNoteGenerationResult,
  LiveNoteTrackingStats,
  LiveNoteVersion,
  NotificationInfo,
  RatingStatus,
  RejectionDecision,
  Suggestion,
  SuggestionEvaluation,
  UpdateDecision,
)
from .notes_data_client import NotesDataClient
from .prompts import (
  build_generation_prompt,
  build_update_decider_prompt,
  get_evaluate_whether_single_suggestion_is_incorporated_prompt,
  get_live_note_candidate_rejector_prompt,
  get_live_note_generation_prompt,
)

from llm.grok_client import GrokClient, SimpleGrokEAPIClient


_PROMPT_PREFIX = " »   "
_PROMPT_CONT = " »       "
_RESPONSE_PREFIX = " «   "
_RESPONSE_CONT = " «       "
_WRAP_WIDTH = 180


def _wrap_line(line: str, prefix: str, continuation: str) -> str:
  """Wrap a single long line, using prefix for first and continuation for rest."""
  if len(prefix + line) <= _WRAP_WIDTH or not line.strip():
    return prefix + line
  wrapped = textwrap.wrap(line, width=_WRAP_WIDTH - len(prefix))
  if not wrapped:
    return prefix + line
  return prefix + wrapped[0] + "".join(f"\n{continuation}{w}" for w in wrapped[1:])


def _format_prompt(text: str) -> str:
  """Prefix every line with » and word-wrap long lines for readability."""
  return "\n".join(_wrap_line(line, _PROMPT_PREFIX, _PROMPT_CONT) for line in text.split("\n"))


def _format_response(text: str) -> str:
  """Prefix every line with « and word-wrap long lines for readability."""
  return "\n".join(_wrap_line(line, _RESPONSE_PREFIX, _RESPONSE_CONT) for line in text.split("\n"))


class LiveNoteGenerator:
  def __init__(
    self,
    logger,
    llm_client: GrokClient = None,
    notes_data_client: NotesDataClient = None,
    max_retries: int = 10,
  ):
    self.logger = logger
    if llm_client is None:
      llm_client = SimpleGrokEAPIClient()
    self.llm_client = llm_client
    if notes_data_client is None:
      raise ValueError("notes_data_client must be provided")
    self.notes_data_client = notes_data_client
    self.max_retries = max_retries
    self._set_version_info_for_model_and_prompt()

  def _set_version_info_for_model_and_prompt(self):
    prompt_base = get_live_note_generation_prompt(
      ContextForGeneration(
        tweet_id="tweet_id",
        note_contents=[],
        past_live_note_versions_with_suggestions=[],
      ),
    )
    version_info_str = f"{self.llm_client.get_model_info()}/{prompt_base}"
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
    self._increment_stat(tracking_stats, "hydrate_context.successes")
    return ContextForGeneration(
      tweet_id=tweet_id,
      live_note_version_id=live_note_version_id,
      note_contents=note_contents,
      past_live_note_versions_with_suggestions=past_live_note_versions_with_suggestions,
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

  def merge_post_hoc_suggestion_evaluations_with_grok_generated_suggestion_evaluations(
    self,
    grok_generated_suggestion_evaluations: dict[int, SuggestionEvaluation],
    post_hoc_suggestion_evaluations: dict[int, SuggestionEvaluation],
  ) -> dict[int, SuggestionEvaluation]:
    """
    Merge post-hoc suggestion evaluations with Grok-generated suggestion evaluations.
    Enables the rest of the code to work whether Grok-generated suggestion evaluations are requested or not.
    """
    if grok_generated_suggestion_evaluations is None:
      grok_generated_suggestion_evaluations = {}

    for suggestion_id, suggestion_evaluation in post_hoc_suggestion_evaluations.items():
      if suggestion_id in grok_generated_suggestion_evaluations:
        grok_generated_suggestion_evaluations[
          suggestion_id
        ].post_hoc_incorporated_status = suggestion_evaluation.incorporated_status
        grok_generated_suggestion_evaluations[
          suggestion_id
        ].post_hoc_incorporated_explanation = suggestion_evaluation.incorporated_explanation
        grok_generated_suggestion_evaluations[
          suggestion_id
        ].post_hoc_decision_explanation = suggestion_evaluation.decision_explanation
        if grok_generated_suggestion_evaluations[suggestion_id].incorporated_status is None:
          grok_generated_suggestion_evaluations[
            suggestion_id
          ].incorporated_status = suggestion_evaluation.incorporated_status
          grok_generated_suggestion_evaluations[
            suggestion_id
          ].incorporated_explanation = suggestion_evaluation.incorporated_explanation
          grok_generated_suggestion_evaluations[
            suggestion_id
          ].decision_explanation = suggestion_evaluation.decision_explanation
      else:
        grok_generated_suggestion_evaluations[suggestion_id] = suggestion_evaluation
    return grok_generated_suggestion_evaluations

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
  ) -> LiveNoteGenerationResult:
    self.logger.info(f"Generating live note for tweet {tweet_id}")

    tracking_stats = self.initialize_tracking_stats()

    context = self.hydrate_context_for_tweet(
      tweet_id,
      tracking_stats=tracking_stats,
      include_suggestions=include_suggestions,
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

    new_live_note_version.rejection_decision = self.decide_whether_to_reject(
      context, new_live_note_version, tracking_stats=tracking_stats
    )
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

    post_hoc_suggestion_evaluations = self.determine_if_suggestions_are_incorporated_post_hoc(
      context, new_live_note_version, tracking_stats=tracking_stats
    )
    new_live_note_version.suggestion_evaluations = (
      self.merge_post_hoc_suggestion_evaluations_with_grok_generated_suggestion_evaluations(
        new_live_note_version.suggestion_evaluations, post_hoc_suggestion_evaluations
      )
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

  def determine_if_suggestions_are_incorporated_post_hoc(
    self,
    context: ContextForGeneration,
    new_live_note_result: LiveNoteVersion,
    tracking_stats: LiveNoteTrackingStats,
    only_check_suggestions_from_latest_version: bool = False,
  ) -> dict[int, SuggestionEvaluation]:
    """
    Parallelizable. Currently written in series for simplicity.
    Also, can use a simpler model for this (e.g. small model if no tool use)
    """
    if (
      only_check_suggestions_from_latest_version
      and len(context.past_live_note_versions_with_suggestions) > 0
    ):
      previous_versions_to_use = [context.past_live_note_versions_with_suggestions[0]]
    else:
      previous_versions_to_use = context.past_live_note_versions_with_suggestions

    self._increment_stat(tracking_stats, "post_hoc_suggestions_eval.attempts")
    suggestion_evaluations = {}
    for previous_live_note_version in previous_versions_to_use:
      for suggestion in previous_live_note_version.suggestions:
        suggestion_evaluations[
          suggestion.suggestion_id
        ] = self.determine_if_suggestion_is_incorporated_post_hoc(
          previous_live_note_version,
          new_live_note_result,
          suggestion,
          tracking_stats=tracking_stats,
        )
    self._increment_stat(tracking_stats, "post_hoc_suggestions_eval.successes")
    return suggestion_evaluations

  def determine_if_suggestion_is_incorporated_post_hoc(
    self,
    previous_live_note_version: LiveNoteVersion,
    new_live_note_result: LiveNoteVersion,
    suggestion: Suggestion,
    tracking_stats: LiveNoteTrackingStats,
  ) -> SuggestionEvaluation:
    try:
      self._increment_stat(tracking_stats, "post_hoc_suggestions.llm_call.attempts")
      prompt = get_evaluate_whether_single_suggestion_is_incorporated_prompt(
        previous_live_note_version, new_live_note_result, suggestion
      )
      self.logger.info(
        f"Evaluating whether suggestion {suggestion.suggestion_id} is incorporated. Prompt:\n{_format_prompt(prompt)}"
      )
      grok_response = self.llm_client.call(prompt)
      self.logger.info(
        f"Grok response for suggestion {suggestion.suggestion_id} evaluation:\n{_format_response(grok_response)}"
      )
      result = parse_answer_from_grok_post_hoc_suggestion_evaluation_response(grok_response)
      self._increment_stat(tracking_stats, "post_hoc_suggestions.llm_call.successes")
      return result
    except Exception as e:
      self._increment_stat(tracking_stats, "post_hoc_suggestions.llm_call.failures")
      self._increment_stat(tracking_stats, "post_hoc_suggestions.llm_call.exceptions")
      if isinstance(e, ValueError):
        self._increment_stat(tracking_stats, "post_hoc_suggestions.llm_call.parse_errors")
      raise

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
      if self._should_bypass_hard_gate_for_suggestion(context, new_live_note_result):
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
        self.logger.info(
          f"Getting Grok rejection decision for post {context.tweet_id}. Prompt:\n{_format_prompt(prompt)}"
        )
        grok_response = self.llm_client.call(prompt)
        self.logger.info(
          f"Grok response for rejection decision for post {context.tweet_id}:\n{_format_response(grok_response)}"
        )
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
          f"Getting Grok update decision for post {context.tweet_id}. Prompt:\n{_format_prompt(prompt)}"
        )
        grok_response = self.llm_client.call(prompt)
        self.logger.info(
          f"Grok response for update decision for post {context.tweet_id}:\n{_format_response(grok_response)}"
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

  def sample_live_note(
    self,
    context: ContextForGeneration,
    tracking_stats: LiveNoteTrackingStats = None,
  ) -> Optional[LiveNoteVersion]:
    retries = 0
    while retries < self.max_retries:
      try:
        self._increment_stat(tracking_stats, "generate_candidate.llm_call.attempts")
        prompt = build_generation_prompt(context)
        self.logger.info(
          f"Getting Grok draft live note generation for post {context.tweet_id}. Prompt:\n{_format_prompt(prompt)}"
        )
        grok_response = self.llm_client.call(prompt)
        self.logger.info(
          f"Grok response for live note generation for post {context.tweet_id}:\n{_format_response(grok_response)}"
        )

        try:
          result = parse_answer_from_grok_generation_response(grok_response, self.logger)
        except ValueError:
          self._increment_stat(tracking_stats, "generate_candidate.llm_call.parse_errors")
          self._increment_stat(tracking_stats, "generate_candidate.llm_call.failures")
          retries += 1
          continue

        # Parse story assessment from generator response
        story_assessment_match = re.search(
          r"<STORY_ASSESSMENT>(.*?)</STORY_ASSESSMENT>", grok_response, re.DOTALL
        )
        if story_assessment_match:
          result.story_assessment = story_assessment_match.group(1).strip()

        if not self.notes_data_client.check_note_character_limit(result.short_live_note):
          self._increment_stat(
            tracking_stats,
            "generate_candidate.generated_short_live_note_exceeds_character_limit",
          )
          retries += 1
          continue

        self._increment_stat(tracking_stats, "generate_candidate.llm_call.successes")
        return result
      except Exception as e:
        self._increment_stat(tracking_stats, "generate_candidate.llm_call.failures")
        self._increment_stat(tracking_stats, "generate_candidate.llm_call.exceptions")
        if isinstance(e, ValueError):
          self._increment_stat(tracking_stats, "generate_candidate.llm_call.parse_errors")
        self.logger.error(
          f"Error generating live note for post {context.tweet_id}: {e}. Retries left: {self.max_retries - retries}",
          exc_info=True,
        )
      retries += 1
    self.logger.info(
      f"Failed to generate live note for post {context.tweet_id} after {self.max_retries} retries. Returning None."
    )
    return None


def _parse_str_from_tag(response: str, tag_name: str, missing_ok: bool = False) -> str:
  tag_match = re.search(rf"<{tag_name}>(.*?)</{tag_name}>", response, re.DOTALL)
  if not tag_match:
    if missing_ok:
      return None
    raise ValueError(f"Could not find {tag_name} in response: {response}")
  tag_str = tag_match.group(1).strip()
  return tag_str


def parse_answer_from_grok_post_hoc_suggestion_evaluation_response(
  response: str,
) -> SuggestionEvaluation:
  incorporated_status_str = _parse_str_from_tag(response, "INCORPORATED")
  explanation_str = _parse_str_from_tag(response, "INCORPORATED_EXPLANATION")
  return SuggestionEvaluation(
    is_incorporated=incorporated_status_str in set(["YES", "PARTIALLY"]),
    incorporated_status=incorporated_status_str,
    incorporated_explanation=explanation_str,
  )


def parse_answer_from_grok_update_decision_response(response: str) -> UpdateDecision:
  should_update_str = _parse_str_from_tag(response, "SHOULD_UPDATE")
  update_explanation_str = _parse_str_from_tag(response, "UPDATE_EXPLANATION")
  difference_from_previous_str = _parse_str_from_tag(response, "DIFFERENCE_EXPLANATION")
  return UpdateDecision(
    should_update=(should_update_str == "YES"),
    update_explanation=update_explanation_str,
    difference_from_previous=difference_from_previous_str,
  )


def parse_answer_from_grok_reject_response(response: str) -> RejectionDecision:
  should_reject_str = _parse_str_from_tag(response, "REJECT")
  reject_reason_str = _parse_str_from_tag(response, "REJECT_REASON")
  retryable_str = _parse_str_from_tag(response, "RETRYABLE")
  if should_reject_str not in set(["YES", "NO"]):
    raise ValueError(f"Invalid should_reject value: {should_reject_str} (must be YES or NO)")
  if retryable_str not in set(["YES", "NO"]):
    raise ValueError(f"Invalid retryable value: {retryable_str} (must be YES or NO)")
  return RejectionDecision(
    should_reject=should_reject_str == "YES", rejection_reason=reject_reason_str
  )


def parse_suggestion_explanations_from_grok_response(
  response: str, logger
) -> dict[int, SuggestionEvaluation]:
  suggestion_explanations_str = _parse_str_from_tag(
    response, "SUGGESTION_EXPLANATIONS", missing_ok=True
  )
  if suggestion_explanations_str is None:
    return {}
  try:
    suggestion_explanations_list = json.loads(suggestion_explanations_str)

    suggestion_evaluations_dict = {}
    for explanation in suggestion_explanations_list:
      suggestion_evaluations_dict[int(explanation["suggestion_id"])] = SuggestionEvaluation(
        is_incorporated=explanation["incorporated_status"] == "YES"
        or explanation["incorporated_status"] == "PARTIALLY",
        incorporated_status=explanation["incorporated_status"],
        incorporated_explanation=explanation["incorporated_explanation"]
        if "incorporated_explanation" in explanation
        else None,
        decision_explanation=explanation["decision_explanation"],
      )
    return suggestion_evaluations_dict
  except Exception as e:
    logger.error(
      f"Error parsing suggestion explanations from Grok response: {e}. Response: {suggestion_explanations_str}",
      exc_info=True,
    )
    return {}


def parse_answer_from_grok_generation_response(response: str, logger) -> LiveNoteVersion:
  """
  Parses the answer from the Grok response. Raises a ValueError if the answer is not valid.
  """
  live_note_classification_str = _parse_str_from_tag(response, "CLASSIFICATION")
  proposed_note_str = _parse_str_from_tag(response, "PROPOSED_NOTE")
  category_str = _parse_str_from_tag(response, "CATEGORY")
  detail_str = _parse_str_from_tag(response, "DETAIL")
  sources_considered_str = _parse_str_from_tag(response, "SOURCES_CONSIDERED")
  suggestion_evaluations = parse_suggestion_explanations_from_grok_response(response, logger)

  return LiveNoteVersion(
    live_note_classification=live_note_classification_str,
    category=category_str,
    short_live_note=proposed_note_str,
    long_live_note=detail_str,
    sources_considered=sources_considered_str,
    suggestion_evaluations=suggestion_evaluations,
  )
