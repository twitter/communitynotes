"""Media comparison pipeline for detecting mismatched fact-check sources.

Per-URL pipeline: analyzes each source URL independently against the post's media.
Each consensus run gets independent P1 (analyze source), P2 (analyze post), and
P3 (compare) calls. Default is 3 independent runs per URL.

Public functions:
  check_media_comparison_pipeline_eligibility - Filter prompt to decide if pipeline is needed.
  generate_media_match_analysis - Run the per-URL pipeline and return per-URL results.
"""

import concurrent.futures
import re
import threading
import time

from .constants import (
  MediaComparisonVotes,
  MediaMatchVerdict,
  UrlMediaComparisonResult,
  format_response_for_logging,
)


PROMPT_FILTER_SPECIFIC_MEDIA = """\
You are reviewing a community note that fact-checks a social media post.

**Question:** Does this note make claims about the SPECIFIC MEDIA in the post that require proving the media's origin or identity?

Answer **YES** only if ALL of these are true:
1. There is an image or video in the post
2. The note makes a claim that specifically relates to the media in the post (not just about facts/events/people)
3. The claim requires proving the media is from a specific source, event, time, location, or creator
4. The note would be UNCONVINCING if the source showed similar but different media

**YES examples:**
- "This depicts [a different event], not [claimed event]" — proving media provenance
- "An event of the type described in this post occurred, but the picture is of [a different]" — proving media provenance
- "This video is actually from [different event], not [claimed event]" — proving media provenance
- "This photo is by [creator X], not [claimed creator Y]" — attribution dispute
- "The photo shows [X]" — explicit reference to specific media identity
- "This image shows [X location] in [Y year]" — identifying specific media origin
- "This [screenshot] is from [source Y]" - identifying specific media origin
- "The image is AI-generated as you can see by this [expert analysis] of it here" — source must match post image, otherwise would be unconvincing and possibly unrelated
- Notes about media that could easily be confused (many similar boats, buildings, explosions)

Answer **NO** if ANY of these apply:
- The post has no media (text-only)
- The note makes claims about facts that can be convincingly verified by an article (dates, quotes, statement, events, policies)
- The note is about what people DID or SAID (not about the media itself)
- The note addresses a claim MADE IN the media OR corrects a misinterpretation of what was said/shown — any source confirming the actual content works, regardless of which recording captured it
- The note just needs to show "an example of X" rather than "this exact X"
- The note debunks a fake/manipulated image (the authentic original WON'T match)
- The note adds context to an opinion post
- The note could cite a text article and still be just as convincing

**NO examples:**
- "This claim about [policy/event] is false because [factual evidence]" — article suffices
- "This quote is from [date], not recent" — any source with the quote works
- "The image has been digitally altered [to change feature X]" — ANY authentic image showing the real feature X proves the alteration, even if the note mentions a specific original source
- "The image is AI-generated as can be seen by [the person having 6 fingers]" — evidence is in the image in the post itself
- "This person did/said [X]" — sources about the person work, media match irrelevant
- "[Person] is quoted out of context" — full quote from any source works, so long as the quote is clearly and certainly from the same event
- "The video claims [X happened], which is unverified/false" — debunking the claim works regardless of which video made it
- "The [audio/video] is misinterpreted — [person] actually said [X]" — any source with the correct transcript/translation works

Output only <specific_media>YES</specific_media> or <specific_media>NO</specific_media>, followed by a one-sentence explanation.

**Community Note:** {note_text}

**Cited Sources:** {source_urls}

**X Post:** https://x.com/i/status/{tweet_id}
"""


def parse_filter_response(filter_response: str) -> tuple[bool, str]:
  """Parse the filter prompt response.

  Args:
    filter_response: Raw LLM response containing <specific_media>YES/NO</specific_media>

  Returns:
    Tuple of (should_run_media_pipeline: bool, filter_explanation: str)
  """
  tag_match = re.search(
    r"<specific_media>(YES|NO)</specific_media>", filter_response, re.IGNORECASE
  )
  if not tag_match:
    # Default to NO if we can't parse - avoid unnecessary pipeline runs
    return False, "Could not parse filter response"

  should_run_media_pipeline = tag_match.group(1).upper() == "YES"

  # Extract explanation (text after the tag)
  filter_explanation = filter_response[tag_match.end() :].strip()
  # Take first sentence/line
  filter_explanation = filter_explanation.split("\n")[0].strip()

  return should_run_media_pipeline, filter_explanation


# ═══════════════════════════════════════════════════════════════════════════
# Per-URL pipeline prompts
# ═══════════════════════════════════════════════════════════════════════════

PROMPT_ANALYZE_SINGLE_SOURCE = """\
Below is a single link that was cited as a source related to media in a social \
media post. Analyze the media embedded in or linked from this page:

1. Find images/videos on the page relevant to the topic. Ignore ads/unrelated content.
2. Use your tools (view_image, browse_page, etc.) to view each piece of media.
3. Output a detailed 500-1000 char description of each relevant media item: what you \
see, environment, colors, camera angles, objects, motion. Base analysis entirely on \
what you see, not surrounding text.

Link to analyze: {url}
"""

PROMPT_ANALYZE_X_POST = """\
Review the media in this X post and output a detailed 500-1000 char description \
of each piece of media. What's happening, environment, colors, camera angles. \
Base analysis entirely on the media itself, not text in replies or notes.

Post: https://x.com/i/status/{tweet_id}

Use x_thread_fetch, then view_x_video or view_x_image on the media URLs.
"""

PROMPT_COMPARE_SINGLE_URL = """\
You analyzed media from an article/source:
{p1_output}

And media in an X Post:
{p2_output}

Answer two questions about THIS SPECIFIC source ({url}):

1. SAME MEDIA: Is the X post's media the same media as in this source? Minor \
publishing differences (blur, crop, subtitles, logos) are OK. Substantial scene \
differences mean NOT the same.

2. SAME INCIDENT: Does the media depict the same incident? Different angles or \
different buildings/structures mean NOT the same incident.

Output in <SAME_MEDIA><SHORT_ANSWER>YES or NO</SHORT_ANSWER>analysis</SAME_MEDIA>
and <SAME_INCIDENT><SHORT_ANSWER>YES or NO</SHORT_ANSWER>analysis</SAME_INCIDENT>
"""

_NUM_CONSENSUS_RUNS = 3
_MAX_WORKERS_PER_TWEET = 128
_CACHE_TTL_SECS = 3600  # 1 hour

# In-memory cache for media description LLM results.
# Value: (result_string, timestamp)
_description_cache: dict[str, tuple[str, float]] = {}
_cache_lock = threading.Lock()
_key_locks: dict[str, threading.Lock] = {}  # per-key locks to prevent duplicate LLM calls


def _cached_call(llm_client, prompt: str, cache_key: str, logger=None):
  """Call LLM with caching. Returns cached result if within TTL.

  Uses per-key locking so concurrent requests for the same key wait for
  the first caller rather than making duplicate LLM calls.
  """
  # Fast path: check cache under global lock
  with _cache_lock:
    entry = _description_cache.get(cache_key)
    if entry and time.time() - entry[1] < _CACHE_TTL_SECS:
      age_secs = int(time.time() - entry[1])
      if logger:
        logger.info(
          f"  Media cache HIT: {cache_key} (age={age_secs}s, {len(entry[0])} chars, {len(_description_cache)} entries in cache)"
        )
      return entry[0]
    # Get or create a per-key lock
    if cache_key not in _key_locks:
      _key_locks[cache_key] = threading.Lock()
    key_lock = _key_locks[cache_key]

  # Serialize callers for the same key — only one makes the LLM call
  with key_lock:
    # Re-check cache (another thread may have populated it while we waited)
    with _cache_lock:
      entry = _description_cache.get(cache_key)
      if entry and time.time() - entry[1] < _CACHE_TTL_SECS:
        if logger:
          logger.info(f"  Media cache HIT (after wait): {cache_key}")
        return entry[0]

    if logger:
      logger.info(f"  Media cache MISS: {cache_key} — calling LLM")

    result = llm_client.call(prompt)

    if not result:
      if logger:
        logger.warning(f"  Media cache SKIP: {cache_key} — LLM returned empty/None, not caching")
      return result

    with _cache_lock:
      _description_cache[cache_key] = (result, time.time())
      cache_size = len(_description_cache)
      # Evict expired entries periodically
      if cache_size > 0 and cache_size % 50 == 0:
        now = time.time()
        expired = [k for k, (_, ts) in _description_cache.items() if now - ts >= _CACHE_TTL_SECS]
        for k in expired:
          del _description_cache[k]
          _key_locks.pop(k, None)
        # Also clean up orphaned key locks (no cache entry)
        orphaned_locks = [k for k in _key_locks if k not in _description_cache]
        for k in orphaned_locks:
          del _key_locks[k]
        if expired and logger:
          logger.info(
            f"  Media cache evicted {len(expired)} expired entries, {len(orphaned_locks)} orphaned locks"
          )

    if logger:
      logger.info(
        f"  Media cache STORE: {cache_key} ({len(result or '')} chars, {cache_size} entries in cache)"
      )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Internals
# ═══════════════════════════════════════════════════════════════════════════


def _parse_verdict(tag: str, response: str) -> MediaMatchVerdict:
  """Extract a YES/NO verdict from an XML tag in the comparison response."""
  match = re.search(
    rf"<{tag}>.*?<SHORT_ANSWER>(YES|NO)</SHORT_ANSWER>", response, re.DOTALL | re.IGNORECASE
  )
  if match:
    return MediaMatchVerdict(match.group(1).upper())
  return MediaMatchVerdict.INCONCLUSIVE


def _count_verdicts(verdicts: list[MediaMatchVerdict]) -> MediaComparisonVotes:
  """Count verdicts into vote tallies."""
  return MediaComparisonVotes(
    yes_votes=verdicts.count(MediaMatchVerdict.YES),
    no_votes=verdicts.count(MediaMatchVerdict.NO),
    error_votes=verdicts.count(MediaMatchVerdict.INCONCLUSIVE),
  )


def _get_consensus(votes: MediaComparisonVotes) -> MediaMatchVerdict:
  """Determine consensus from vote counts (majority wins, else INCONCLUSIVE)."""
  if votes.yes_votes > votes.no_votes and votes.yes_votes > votes.error_votes:
    return MediaMatchVerdict.YES
  if votes.no_votes > votes.yes_votes and votes.no_votes > votes.error_votes:
    return MediaMatchVerdict.NO
  return MediaMatchVerdict.INCONCLUSIVE


_FILTER_MAX_RETRIES = 3


def check_media_comparison_pipeline_eligibility(
  logger, llm_client, tweet_id: str, note_text: str, source_urls: list[str]
) -> tuple[bool, str]:
  """Call Grok to determine whether the media comparison pipeline should run.

  Retries up to _FILTER_MAX_RETRIES times on parse failures or LLM errors.
  Raises RuntimeError if all retries fail (caller should handle gracefully).
  """
  prompt = PROMPT_FILTER_SPECIFIC_MEDIA.format(
    note_text=note_text,
    source_urls="\n".join(source_urls),
    tweet_id=tweet_id,
  )
  logger.info(f"Running media filter for post {tweet_id} with {len(source_urls)} source URLs")

  last_error = None
  for attempt in range(_FILTER_MAX_RETRIES):
    try:
      response = llm_client.call(prompt)
      logger.info(
        f"Media filter response for post {tweet_id} (attempt {attempt + 1}):\n"
        f"{format_response_for_logging(response)}"
      )
      should_run, explanation = parse_filter_response(response)
      if explanation == "Could not parse filter response":
        logger.warning(
          f"Media filter for post {tweet_id}: could not parse response on attempt {attempt + 1}/{_FILTER_MAX_RETRIES}, retrying"
        )
        last_error = f"Unparseable response: {(response or '')[:200]}"
        continue
      return should_run, explanation
    except Exception as e:
      logger.warning(
        f"Media filter for post {tweet_id}: error on attempt {attempt + 1}/{_FILTER_MAX_RETRIES}: {e}"
      )
      last_error = str(e)

  error_msg = (
    f"Media filter failed after {_FILTER_MAX_RETRIES} retries for post {tweet_id}: {last_error}"
  )
  logger.error(error_msg)
  raise RuntimeError(error_msg)


def _is_self_post_url(url: str, tweet_id: str) -> bool:
  """Check if a URL points to the post itself (x.com or twitter.com status URL with this tweet ID)."""
  return ("x.com/" in url or "twitter.com/" in url) and f"/{tweet_id}" in url


def generate_media_match_analysis(
  logger,
  llm_client,
  tweet_id: str,
  source_urls: list[str],
  num_consensus_runs: int = _NUM_CONSENSUS_RUNS,
) -> list[UrlMediaComparisonResult]:
  """Run per-URL media comparison pipeline with full independent consensus.

  Each consensus run gets its own independent source-media-analysis, post-media-analysis,
  and comparison calls to capture variance in media description.

  Phase 1 (all in parallel): N × post-media-analysis + N × source-media-analysis per URL
  Phase 2 (all in parallel): N × comparison per URL

  Returns:
    List of UrlMediaComparisonResult, one per source URL.
  """
  n = num_consensus_runs

  # Skip URLs that are the post itself
  filtered = [u for u in source_urls if not _is_self_post_url(u, tweet_id)]
  if len(filtered) < len(source_urls):
    skipped = [u for u in source_urls if _is_self_post_url(u, tweet_id)]
    logger.info(
      f"Media pipeline for post {tweet_id}: skipping {len(skipped)} self-post URLs: {skipped}"
    )
    source_urls = filtered

  num_urls = len(source_urls)
  if num_urls == 0:
    logger.info(f"Media pipeline for post {tweet_id}: no source URLs to analyze after filtering")
    return []

  logger.info(
    f"Media pipeline for post {tweet_id}: {n}× post-media-analysis + {n}× source-media-analysis "
    f"for {num_urls} URLs in parallel, then {n}× comparison per URL"
  )

  # Phase 1: All post-media-analysis and source-media-analysis calls in parallel
  post_media_analysis_prompt = PROMPT_ANALYZE_X_POST.format(tweet_id=tweet_id)
  phase1_workers = min(n + n * num_urls, _MAX_WORKERS_PER_TWEET)

  with concurrent.futures.ThreadPoolExecutor(max_workers=phase1_workers) as executor:
    post_media_analysis_futures = [
      executor.submit(
        _cached_call,
        llm_client,
        post_media_analysis_prompt,
        f"post-media-analysis:{tweet_id}:{i}",
        logger,
      )
      for i in range(n)
    ]
    source_media_analysis_futures_by_url = {
      url: [
        executor.submit(
          _cached_call,
          llm_client,
          PROMPT_ANALYZE_SINGLE_SOURCE.format(url=url),
          f"url-media-analysis:{url}:{i}",
          logger,
        )
        for i in range(n)
      ]
      for url in source_urls
    }

  # Collect post-media-analysis results
  post_analyses = []
  for i, fut in enumerate(post_media_analysis_futures):
    try:
      post_analyses.append(fut.result())
    except Exception as e:
      logger.error(f"[post {tweet_id}] Post-media-analysis run {i} failed: {e}")
      post_analyses.append(None)
  logger.info(
    f"[post {tweet_id}] Post-media-analysis: {sum(1 for p in post_analyses if p)}/{n} succeeded"
  )

  # Collect source-media-analysis results
  source_analyses_by_url = {}
  for url in source_urls:
    outputs = []
    for i, fut in enumerate(source_media_analysis_futures_by_url[url]):
      try:
        outputs.append(fut.result())
      except Exception as e:
        logger.error(f"[post {tweet_id}] Source-media-analysis run {i} for {url} failed: {e}")
        outputs.append(None)
    source_analyses_by_url[url] = outputs
    logger.info(
      f"[post {tweet_id}]   Source-media-analysis for {url}: {sum(1 for o in outputs if o)}/{n} succeeded"
    )

  # Phase 2: Media-comparison calls — pair source-media-analysis[i] with post-media-analysis[i], all in parallel
  media_comparison_futures = {}  # (url, run_index) -> future
  phase2_workers = min(n * num_urls, _MAX_WORKERS_PER_TWEET)

  with concurrent.futures.ThreadPoolExecutor(max_workers=phase2_workers) as executor:
    for url in source_urls:
      for i in range(n):
        source_out = source_analyses_by_url[url][i]
        post_out = post_analyses[i]
        if source_out and post_out:
          prompt = PROMPT_COMPARE_SINGLE_URL.format(
            p1_output=source_out,
            p2_output=post_out,
            url=url,
          )
          media_comparison_futures[(url, i)] = executor.submit(llm_client.call, prompt)

  # Collect comparison results and compute consensus per URL
  results: list[UrlMediaComparisonResult] = []

  for url in source_urls:
    media_verdicts = []
    incident_verdicts = []
    for i in range(n):
      key = (url, i)
      if key not in media_comparison_futures:
        media_verdicts.append(MediaMatchVerdict.INCONCLUSIVE)
        incident_verdicts.append(MediaMatchVerdict.INCONCLUSIVE)
        continue
      try:
        response = media_comparison_futures[key].result()
        media_verdicts.append(_parse_verdict("SAME_MEDIA", response))
        incident_verdicts.append(_parse_verdict("SAME_INCIDENT", response))
      except Exception as e:
        logger.error(f"[post {tweet_id}] Media-comparison run {i} for {url} failed: {e}")
        media_verdicts.append(MediaMatchVerdict.INCONCLUSIVE)
        incident_verdicts.append(MediaMatchVerdict.INCONCLUSIVE)

    media_votes = _count_verdicts(media_verdicts)
    incident_votes = _count_verdicts(incident_verdicts)
    result = UrlMediaComparisonResult(
      url=url,
      same_media=_get_consensus(media_votes),
      same_incident=_get_consensus(incident_votes),
      same_media_votes=media_votes,
      same_incident_votes=incident_votes,
    )
    results.append(result)
    logger.info(
      f"[post {tweet_id}]   {url}: same_media={result.same_media.value} same_incident={result.same_incident.value}"
    )

  return results
