import re
import httpx


async def extract_and_validate_urls(text: str, citations: list[str], timeout: int = 10) -> list[tuple[str, int | None]]:
    """
    Extract URLs from text and validate that all of them return 2XX or 3XX status codes.
    
    Args:
        text: The text to extract URLs from
        citations: The citations to validate
        timeout: Request timeout in seconds (default: 10)
    
    Returns:
        A list of tuples (url, status_code) for URLs that failed validation.
        Returns an empty list if all URLs are valid or if no URLs are found.
    
    Example:
        >>> text = "Check this out: https://example.com and https://example.org"
        >>> failed_urls = await extract_and_validate_urls(text)
        >>> if failed_urls:
        ...     print(f"Failed URLs: {failed_urls}")
    """
    # Regular expression to match URLs (http/https)
    url_pattern = r'https?://[^\s<>"\'\)]+[^\s<>"\'\)\.,;:!?]'
    urls = re.findall(url_pattern, text)
    
    # If no URLs found, return empty list (nothing to validate)
    if not urls:
        return []
    
    # Validate each URL and collect failures
    failed_urls = []
    async with httpx.AsyncClient() as client:
        for url in urls:
            if url in citations:
                continue
            is_valid, status_code = await validate_url(url, client=client, timeout=timeout)
            if not is_valid:
                failed_urls.append((url, status_code))
    
    return failed_urls


async def validate_url(url: str, client: httpx.AsyncClient | None = None, timeout: int = 10) -> tuple[bool, int | None]:
    """
    Validate that a URL returns a 2XX or 3XX status code.
    
    Args:
        url: The URL to validate
        client: Optional httpx.AsyncClient to use (if None, a new client will be created)
        timeout: Request timeout in seconds (default: 10)
    
    Returns:
        A tuple of (is_valid, status_code) where:
        - is_valid: True if status code is 2XX or 3XX, False otherwise
        - status_code: The HTTP status code, or None if request failed
    
    Example:
        >>> is_valid, status_code = await validate_url("https://example.com")
        >>> if is_valid:
        ...     print(f"URL is valid with status code {status_code}")
    """
    try:
        if client is None:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=timeout, follow_redirects=True)
                status_code = response.status_code
                is_valid = 200 <= status_code < 400
                return is_valid, status_code
        else:
            response = await client.get(url, timeout=timeout, follow_redirects=True)
            status_code = response.status_code
            is_valid = 200 <= status_code < 400
            return is_valid, status_code
    except httpx.HTTPError:
        return False, None

