import os

import dotenv
import requests


def _make_request(payload: dict):
    """
    Currently extremely simple and includes no retry logic.
    """
    chat_completions_url = "https://api.x.ai/v1/chat/completions"
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
}
    response = requests.post(chat_completions_url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Error making request: {response.status_code} {response.text}")
    return response.json()["choices"][0]["message"]["content"]


def get_grok_response(prompt: str, temperature: float = 0.8, model: str = "grok-3-latest"):
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a highly intelligent AI assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        "model": model,
        "temperature": temperature,
    }
    return _make_request(payload)


def grok_describe_image(image_url: str, temperature: float = 0.01, model: str = "grok-2-vision-latest"):
    """
    Currently just describe image on its own. There are many possible
    improvements to consider making, e.g. passing in the post text or
    other context and describing the image and post text together.
    """
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": "What's in this image?",
                    },
                ],
            },
        ],
        "model": model,
        "temperature": temperature,
    }
    return _make_request(payload)


def get_grok_live_search_response(prompt: str, temperature: float = 0.8, model= "grok-3-latest"):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "search_parameters": {
            "mode": "on",
        },
        "model": model,
        "temperature": temperature,
    }
    return _make_request(payload)


if __name__ == "__main__":
    dotenv.load_dotenv()
    print(
        get_grok_live_search_response(
            "Provide me a digest of world news in the last 2 hours. Please respond with links to each source next to the claims that the source supports."
        )
    )
