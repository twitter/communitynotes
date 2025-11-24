from openai import OpenAI


class LLMClient:
    """Client for interacting with xAI's Grok models via OpenAI-compatible API."""
    
    def __init__(self, api_key: str):
        """
        Initialize the LLM client with the provided API key.
        
        Args:
            api_key: The xAI API key for authentication
        """
        self._api_key: str = api_key
        self._client: OpenAI | None = None
    
    @property
    def client(self) -> OpenAI:
        """Get OpenAI client configured for xAI API."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url="https://api.x.ai/v1",
            )
        return self._client
    
    def get_grok_response(self, prompt: str, temperature: float = 0.8, model: str = "grok-3-latest") -> str:
        """
        Get a response from Grok for a given prompt.
        
        Args:
            prompt: The prompt to send to Grok
            temperature: Temperature for response generation (default: 0.8)
            model: Model to use (default: "grok-3-latest")
            
        Returns:
            The generated response text
        """
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a highly intelligent AI assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=temperature,
        )
        return response.choices[0].message.content
    
    def grok_describe_image(self, image_url: str, temperature: float = 0.01, model: str = "grok-2-vision-latest") -> str:
        """
        Describe an image using Grok's vision capabilities.
        
        Currently just describe image on its own. There are many possible
        improvements to consider making, e.g. passing in the post text or
        other context and describing the image and post text together.
        
        Args:
            image_url: URL of the image to describe
            temperature: Temperature for response generation (default: 0.01)
            model: Model to use (default: "grok-2-vision-latest")
            
        Returns:
            The image description text
        """
        response = self.client.chat.completions.create(
            messages=[
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
            model=model,
            temperature=temperature,
        )
        return response.choices[0].message.content
    
    def get_grok_live_search_response(self, prompt: str, temperature: float = 0.8, model: str = "grok-3-latest") -> str:
        """
        Get a response from Grok with live web search enabled.
        
        Args:
            prompt: The prompt to send to Grok
            temperature: Temperature for response generation (default: 0.8)
            model: Model to use (default: "grok-3-latest")
            
        Returns:
            The generated response text with live search results
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            extra_body={
                "search_parameters": {
                    "mode": "on",
                },
            },
            model=model,
            temperature=temperature,
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    import dotenv
    import os

    dotenv.load_dotenv()
    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        raise ValueError("XAI_API_KEY environment variable is required")
    
    llm_client = LLMClient(api_key=xai_api_key)
    print(
        llm_client.get_grok_live_search_response(
            "Provide me a digest of world news in the last 2 hours. Please respond with links to each source next to the claims that the source supports."
        )
    )
