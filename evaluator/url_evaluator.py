import html
import logging
import os
import re
from typing import Callable, FrozenSet, List


logger = logging.getLogger("birdwatch")
logger.setLevel(logging.INFO)

# IANA's list of top-level domains (https://data.iana.org/TLD/tlds-alpha-by-domain.txt).
_TLD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tlds-alpha-by-domain.txt")


def _load_tlds(path: str = _TLD_FILE) -> FrozenSet[str]:
  """Return the top-level domains in IANA's list, lower-cased."""
  with open(path, encoding="ascii") as handle:
    return frozenset(
      line.strip().lower() for line in handle if line.strip() and not line.startswith("#")
    )


_TLDS = _load_tlds()


def _is_url_like(match: str) -> bool:
  """
  Whether text matched by the URL pattern should be checked as a URL.

  Text with a scheme or a leading "www." always is. Text without either, such as
  "example.com/page", is only when its host ends in a real top-level domain. That is the
  rule X's own text library (twitter-text) applies when it decides what to render as a
  link, and it keeps words joined by a full stop ("ended.The", "Three.js",
  "settings.json") from being checked, and failed, as if they were links.
  """
  lowered = match.lower()
  if lowered.startswith(("http://", "https://", "www.")):
    return True
  host = match.split("/", 1)[0].rstrip(".")
  return host.rsplit(".", 1)[-1].lower() in _TLDS


def check_all_urls_for_note(note_text: str, check_url_fn: Callable[[str], bool]) -> bool:
  """
  Check all URLs in the note text to see if they are valid.
  For each URL, there are multiple variants (e.g. with and without trailing punctuation).
  If at least one variant of each URL is valid, return True.
  If there is any URL with no valid variant, return False.

  Args:
      note_text (str): The text of the note to check.
      check_url_fn (Callable[[str], bool]): A function to check if a individual URL
          returns a 200 status code, after e.g. following redirects.
  """
  note_text_unescaped = unescape(note_text)
  urls = _extract_urls(note_text_unescaped)

  if len(urls) == 0:
    logger.info(f"No URLs found in note text: {note_text_unescaped}")
    return False

  for url_variant_list in urls:
    # For each URL variant
    at_least_one_good_url_variant = False
    for url_variant in url_variant_list:
      if check_url_fn(url_variant):
        at_least_one_good_url_variant = True
        break
    if not at_least_one_good_url_variant:
      logger.info(
        f"No valid URL found for any variant of {url_variant_list} in note text: {note_text_unescaped}"
      )
      return False
  logger.info(f"All URLs in note text are valid: {note_text_unescaped}")
  return True


def _extract_urls(text: str) -> List[List[str]]:
  """
  Return every URL-like substring from *text*.
  Return a List of Lists: each inner List contains multiple possible variants
  of an individual URL.
  """
  pattern = re.compile(
    r"""
        (?:
            https?://               # optional scheme
        )?
        (?:www\.)?                  # optional www
        [\w\-._~%]+                 # subdomain or domain name chars
        \.[a-zA-Z]{2,}              # dot + top level domain (≥2 letters)
        (?:/[^\s]*)?                # optional query/fragment
        """,
    re.VERBOSE,
  )

  raw_matches = pattern.findall(text)

  # Strip common trailing punctuation that often follows URLs in note text
  # Return both variants (with and without trailing punctuation) for each URL.
  strip_trailing = ".,;:!?)]}\"'"
  results = []
  for match in raw_matches:
    if not _is_url_like(match):
      continue
    # Create a list of variants for each match
    variants = [match]
    stripped_variant = match.rstrip(strip_trailing)
    if stripped_variant != match:
      variants.append(stripped_variant)
    results.append(variants)
  return results


def unescape(text: str) -> str:
  """Remove layers of HTML escaping so the text matches natural language."""
  return html.unescape(html.unescape(text)) if isinstance(text, str) else text
