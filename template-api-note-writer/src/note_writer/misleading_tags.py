import json
from typing import List

from data_models import MisleadingTag, Post
from note_writer.llm_util import get_grok_response


def get_misleading_tags(
    post_with_context_description: str, note_text: str, retries: int = 3
) -> List[MisleadingTag]:
    misleading_why_tags_prompt = _get_prompt_for_misleading_why_tags(
        post_with_context_description, note_text
    )
    while retries > 0:
        try:
            misleading_why_tags_str = get_grok_response(misleading_why_tags_prompt)
            misleading_why_tags = json.loads(misleading_why_tags_str)["misleading_tags"]
            return [MisleadingTag(tag) for tag in misleading_why_tags]
        except Exception as e:
            retries -= 1
            if retries == 0:
                raise e
    raise Exception("Failed to get misleading tags for note.")


def _get_prompt_for_misleading_why_tags(post_with_context_description: str, note: str):
    return f"""Below will be a post on X, and a proposed community note that \
adds additional context to the potentially misleading post. \
Your task will be to identify which of the following tags apply to the post and note. \
You may choose as many tags as apply, but you must choose at least one. \
You must respond in valid JSON format, with a list of which of the following options apply:
- "factual_error":  # the post contains a factual error
- "manipulated_media":  # the post contains manipulated/fake/out-of-context media
- "outdated_information":  # the post contains outdated information
- "missing_important_context":  # the post is missing important context
- "disputed_claim_as_fact":  # including unverified claims
- "misinterpreted_satire":  # the post is satire that may likely be misinterpreted as fact
- "other":  # the post contains other misleading reasons

Example valid JSON response:
{{
    "misleading_tags": ["factual_error", "outdated_information", "missing_important_context"]
}}

OTHER = 0
FACTUAL_ERROR = 1
MANIPULATED_MEDIA = 2
OUTDATED_INFORMATION = 3
MISSING_IMPORTANT_CONTEXT = 4
DISPUTED_CLAIM_AS_FACT = 5
MISINTERPRETED_SATIRE = 6

The post and note are as follows:

{post_with_context_description}

```
Proposed community note:
```
{note}
```
"""
