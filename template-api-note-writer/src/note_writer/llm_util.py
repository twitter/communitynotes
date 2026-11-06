import json
import os

import dotenv
import requests


def _make_request(payload: dict) -> str:
    """
    Call the xAI Responses API and return the output text.
    Currently extremely simple and includes no retry logic.
    """
    responses_url = "https://api.x.ai/v1/responses"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
    }
    response = requests.post(responses_url, headers=headers, json=payload, timeout=300)
    if response.status_code != 200:
        raise Exception(f"Error making request: {response.status_code} {response.text}")
    data = response.json()
    if data.get("output_text"):
        return data["output_text"]
    return "".join(
        block.get("text", "")
        for item in data.get("output", [])
        if item.get("type") == "message"
        for block in item.get("content", [])
        if block.get("type") == "output_text"
    )


def get_grok_search_json_response(
    prompt: str,
    schema: dict,
    model: str = "grok-4.3",
) -> dict:
    """
    One call that lets Grok search the web and X, and answer with JSON
    matching `schema`.
    """
    payload = {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "tools": [
            {"type": "web_search", "enable_image_understanding": True},
            {
                "type": "x_search",
                "enable_image_understanding": True,
                "enable_video_understanding": True,
            },
        ],
        "text": {
            "format": {"type": "json_schema", "name": "note", "schema": schema, "strict": True}
        },
        "store": False,
    }
    return json.loads(_make_request(payload))


if __name__ == "__main__":
    dotenv.load_dotenv()
    print(
        get_grok_search_json_response(
            "Provide me a digest of world news in the last 2 hours, with a link to a source for each item.",
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"summary": {"type": "string"}, "url": {"type": "string"}},
                            "required": ["summary", "url"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["items"],
                "additionalProperties": False,
            },
        )
    )
