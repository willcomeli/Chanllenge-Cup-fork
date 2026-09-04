from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_VISION_MODEL = "deepseek-v4-flash-vision-exp"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DOTENV_FILENAMES = (".env", ".env.local")


class JsonChatClient(Protocol):
    model: str

    def complete_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def complete_json_with_images(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        images: list[bytes],
        *,
        mime_type: str = "image/png",
    ) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class LlmConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    base_url: str = DEFAULT_BASE_URL
    resume_enabled: bool = False
    jd_enabled: bool = False

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dotenv_values(root: Path | None = None) -> dict[str, str]:
    root = root or project_root()
    values: dict[str, str] = {}
    for filename in DOTENV_FILENAMES:
        path = root / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in values:
                values[key] = value
    return values


def _setting(name: str, dotenv: dict[str, str]) -> str:
    return os.environ.get(name) or dotenv.get(name) or ""


def _truthy(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _configured_flag(value: str, default: bool) -> bool:
    if not value.strip():
        return default
    return _truthy(value)


def load_llm_config(root: Path | None = None) -> LlmConfig:
    dotenv = _dotenv_values(root)
    api_key = (
        _setting("LLM_API_KEY", dotenv)
        or _setting("DEEPSEEK_API_KEY", dotenv)
        or _setting("OPENAI_API_KEY", dotenv)
    )
    model = _setting("LLM_MODEL", dotenv) or DEFAULT_MODEL
    return LlmConfig(
        api_key=api_key,
        model=model,
        vision_model=_setting("LLM_VISION_MODEL", dotenv) or DEFAULT_VISION_MODEL,
        base_url=_setting("LLM_BASE_URL", dotenv) or DEFAULT_BASE_URL,
        resume_enabled=_truthy(_setting("LLM_RESUME_ENABLED", dotenv) or _setting("RESUME_LLM_ENABLED", dotenv)),
        jd_enabled=_configured_flag(
            _setting("LLM_JD_ENABLED", dotenv) or _setting("JD_LLM_ENABLED", dotenv),
            bool(api_key),
        ),
    )


def llm_config_status(root: Path | None = None) -> dict[str, Any]:
    config = load_llm_config(root)
    return {
        "configured": config.configured,
        "model": config.model,
        "visionModel": config.vision_model,
        "baseUrl": config.base_url,
        "resumeEnabled": config.resume_enabled,
        "jdEnabled": config.jd_enabled,
    }


def write_llm_config(
    api_key: str,
    *,
    model: str = DEFAULT_MODEL,
    vision_model: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    resume_enabled: bool = True,
    root: Path | None = None,
) -> LlmConfig:
    cleaned_key = api_key.strip()
    if not cleaned_key:
        raise ValueError("LLM API key is required")
    cleaned_model = model.strip() or DEFAULT_MODEL
    cleaned_vision_model = (vision_model or "").strip() or DEFAULT_VISION_MODEL
    cleaned_base_url = base_url.strip() or DEFAULT_BASE_URL
    root = root or project_root()
    env_path = root / ".env"
    lines = [
        f"LLM_API_KEY={cleaned_key}",
        f"LLM_BASE_URL={cleaned_base_url}",
        f"LLM_MODEL={cleaned_model}",
        f"LLM_VISION_MODEL={cleaned_vision_model}",
        f"LLM_RESUME_ENABLED={'true' if resume_enabled else 'false'}",
    ]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return LlmConfig(
        api_key=cleaned_key,
        model=cleaned_model,
        vision_model=cleaned_vision_model,
        base_url=cleaned_base_url,
        resume_enabled=resume_enabled,
    )


@dataclass(slots=True)
class ChatCompletionsClient:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: int = 90
    retries: int = 2
    temperature: float = 0.0

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = 90,
        retries: int = 2,
    ) -> "ChatCompletionsClient":
        config = load_llm_config()
        if not config.api_key:
            raise RuntimeError("LLM_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY is required in environment or .env")
        return cls(
            api_key=config.api_key,
            model=model or config.model,
            base_url=base_url or config.base_url,
            timeout=timeout,
            retries=retries,
        )

    def complete_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        return self._complete_payload(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
            }
        )

    def complete_json_with_images(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        images: list[bytes],
        *,
        mime_type: str = "image/png",
    ) -> dict[str, Any]:
        if not images:
            raise ValueError("at least one image is required for multimodal completion")
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}
        ]
        for raw_image in images:
            encoded = base64.b64encode(raw_image).decode("ascii")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                }
            )
        return self._complete_payload(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
            }
        )

    def _complete_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                result = json.loads(raw)
                content = result["choices"][0]["message"]["content"]
                parsed = extract_json_object(content)
                if not isinstance(parsed, dict):
                    raise ValueError("LLM response is not a JSON object")
                return parsed
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                KeyError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM JSON completion failed after {self.retries + 1} attempts: {last_error}") from last_error


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object")
    return parsed
