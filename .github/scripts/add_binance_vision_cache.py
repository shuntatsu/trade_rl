from __future__ import annotations

from pathlib import Path

path = Path("trade_rl/integrations/binance.py")
text = path.read_text(encoding="utf-8")
if "import hashlib\n" not in text:
    text = text.replace("import csv\n", "import csv\nimport hashlib\n", 1)
if "from pathlib import Path\n" not in text:
    text = text.replace("from enum import StrEnum\n", "from enum import StrEnum\nfrom pathlib import Path\n", 1)
old_init = '''    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 0.25,
    ) -> None:
'''
new_init = '''    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 0.25,
        cache_root: str | Path | None = None,
    ) -> None:
'''
if new_init not in text:
    if old_init not in text:
        raise RuntimeError("BinancePublicTransport constructor was not found")
    text = text.replace(old_init, new_init, 1)
old_assign = '''        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds

    def _request_bytes(self, url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
'''
new_assign = '''        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.cache_root = None if cache_root is None else Path(cache_root)

    def _vision_cache_path(self, url: str) -> Path | None:
        if self.cache_root is None or not url.startswith(f"{_VISION_ROOT}/"):
            return None
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_root / digest[:2] / f"{digest}.bin"

    def _request_bytes(self, url: str) -> bytes:
        cache_path = self._vision_cache_path(url)
        if cache_path is not None and cache_path.is_file():
            payload = cache_path.read_bytes()
            if not payload:
                raise BinanceTransportError(
                    f"cached Binance Vision archive is empty: {cache_path}"
                )
            return payload
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
'''
if new_assign not in text:
    if old_assign not in text:
        raise RuntimeError("BinancePublicTransport request boundary was not found")
    text = text.replace(old_assign, new_assign, 1)
old_return = '''                with urllib.request.urlopen(  # noqa: S310 - fixed HTTPS endpoints
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    return response.read()
'''
new_return = '''                with urllib.request.urlopen(  # noqa: S310 - fixed HTTPS endpoints
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    payload = response.read()
                if not payload:
                    raise BinanceTransportError(
                        f"Binance returned an empty response for {url}"
                    )
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache_path.with_suffix(".tmp")
                    temporary.write_bytes(payload)
                    temporary.replace(cache_path)
                return payload
'''
if new_return not in text:
    if old_return not in text:
        raise RuntimeError("BinancePublicTransport response block was not found")
    text = text.replace(old_return, new_return, 1)
path.write_text(text, encoding="utf-8")
