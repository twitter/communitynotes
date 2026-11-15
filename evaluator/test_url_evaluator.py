import pytest

from url_evaluator import _extract_urls, check_all_urls_for_note


def _first_variants(text):
  return [variants[0] for variants in _extract_urls(text)]


@pytest.mark.parametrize(
  "text",
  [
    "The clip was made with Three.js and is not real footage. https://example.org/a",
    "The file is called settings.json. https://example.org/a",
    "It ended.The video is from 2019. https://example.org/a",
    "The fossil is a T.rex, not a Spinosaurus. https://example.org/a",
  ],
)
def test_words_joined_by_a_full_stop_are_not_urls(text):
  assert _first_variants(text) == ["https://example.org/a"]
  # Only the real link is checked, so a note is not failed for ordinary text.
  assert check_all_urls_for_note(text, lambda url: url.startswith("https://example.org"))


@pytest.mark.parametrize(
  "text, expected",
  [
    ("See example.com for details.", ["example.com"]),
    ("See www.example.com/page.", ["www.example.com/page."]),
    ("Source: reuters.com/world/some-story", ["reuters.com/world/some-story"]),
    ("Listed on pump.fun today.", ["pump.fun"]),
    ("https://en.wikipedia.org/wiki/Foo_(film)", ["https://en.wikipedia.org/wiki/Foo_(film)"]),
  ],
)
def test_real_addresses_are_still_found(text, expected):
  assert _first_variants(text) == expected


def test_a_dead_bare_address_still_fails_the_note():
  text = "Debunked at example.com/gone and https://example.org/a"
  assert not check_all_urls_for_note(text, lambda url: "gone" not in url)


def test_a_note_without_any_url_still_fails():
  assert not check_all_urls_for_note("It ended.The video is from 2019.", lambda url: True)
