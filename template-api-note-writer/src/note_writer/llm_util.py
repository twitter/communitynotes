import asyncio
import dotenv
import os
from typing import Any

from xai_sdk import AsyncClient
from xai_sdk.chat import user, system
from xai_sdk.tools import web_search, x_search


class LLMClient:
    """Client for interacting with xAI's Grok models via xAI SDK."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "grok-4-fast-community-notes-r35",
        enable_web_image_understanding: bool = False,
        enable_x_image_understanding: bool = False,
        enable_x_video_understanding: bool = False
    ):
        """
        Initialize the LLM client with the provided API key.
        
        Args:
            api_key: The xAI API key for authentication
            enable_web_image_understanding: Enable image understanding in web search (default: False)
            enable_x_image_understanding: Enable image understanding in X search (default: False)
            enable_x_video_understanding: Enable video understanding in X search (default: False)
        """
        self._api_key: str = api_key
        self._client: AsyncClient | None = None
        self._model: str = model
        self._enable_web_image_understanding: bool = enable_web_image_understanding
        self._enable_x_image_understanding: bool = enable_x_image_understanding
        self._enable_x_video_understanding: bool = enable_x_video_understanding
    
    @property
    def client(self) -> AsyncClient:
        """Get xAI async client."""
        if self._client is None:
            self._client = AsyncClient(api_key=self._api_key, api_host="research-models.api.x.ai")
        return self._client
    
    async def get_grok_response(
        self, 
        prompt: str, 
        temperature: float = 0.8,
        timeout: float | None = None,
    ) -> tuple[str, list[Any], list[str]]:
        """
        Get a response from Grok for a given prompt with web and X search enabled.
        
        Args:
            prompt: The prompt to send to Grok
            temperature: Temperature for response generation (default: 0.8)
            timeout: Optional timeout in seconds for the streaming operation (default: None for no timeout)
            
        Returns:
            The generated response text
        
        Raises:
            asyncio.TimeoutError: If the operation exceeds the specified timeout
        """
        # Create a chat session with web_search and x_search tools enabled
        chat = self.client.chat.create(
            model=self._model,
            temperature=temperature,
            tools=[
                web_search(
                    enable_image_understanding=self._enable_web_image_understanding,
                ),
                x_search(
                    enable_image_understanding=self._enable_x_image_understanding,
                    enable_video_understanding=self._enable_x_video_understanding
                )
            ],
        )
        
        # Add system and user messages
        chat.append(system("You are a highly intelligent AI assistant."))
        chat.append(user(prompt))
        
        # Stream the response and collect the final content
        async def stream_response():
            final_content: list[str] = []
            is_thinking = True
            tool_calls = []
            citations = []
            async for response, chunk in chat.stream():
                # View the server-side tool calls as they are being made in real-time
                for tool_call in chunk.tool_calls:
                    tool_calls.append(tool_call)
                if chunk.content and is_thinking:
                    is_thinking = False
                if chunk.content and not is_thinking:
                    final_content.append(chunk.content)
                citations.extend(response.citations)
            return final_content, tool_calls, citations
        
        # Apply timeout if specified
        if timeout is not None:
            final_content, tool_calls, citations = await asyncio.wait_for(
                stream_response(), 
                timeout=timeout
            )
        else:
            final_content, tool_calls, citations = await stream_response()
        
        return "".join(final_content), tool_calls, citations


if __name__ == "__main__":
    dotenv.load_dotenv()
    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        raise ValueError("XAI_API_KEY environment variable is required")
    
    async def main():
        llm_client = LLMClient(api_key=xai_api_key)
        result = await llm_client.get_grok_response(
            "Provide me a digest of world news in the last 2 hours. Please respond with links to each source next to the claims that the source supports."
        )
        print(result)
    
    asyncio.run(main())
