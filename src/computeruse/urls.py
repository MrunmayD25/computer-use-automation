"""Shared HTTP URL validation for the policy and CLI input boundaries."""

from urllib.parse import SplitResult, urlsplit


def parse_http_url(value: str) -> SplitResult:
    """Parse an HTTP(S) URL.

    Raises
    ------
    ValueError
        If the URL is malformed or contains unsafe components.
    """
    if (
        any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in value
        )
        or "\\" in value
    ):
        raise ValueError("URL contains whitespace or control characters")
    url = urlsplit(value)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or "%" in url.hostname
        or url.username is not None
        or url.password is not None
        or url.port == 0
    ):
        raise ValueError("URL must use HTTP(S) without embedded credentials")
    return url


def http_origin(value: str) -> str:
    """Return the validated origin with host case and default ports normalized."""
    url = parse_http_url(value)
    host = url.hostname
    if host is None:
        raise ValueError("URL must use HTTP(S) without embedded credentials")
    if ":" in host:
        host = f"[{host}]"
    default_port = 443 if url.scheme == "https" else 80
    port = f":{url.port}" if url.port not in (None, default_port) else ""
    return f"{url.scheme}://{host}{port}"
