"""Azure AI Foundry chat client that reports every request's measured token usage."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

PROJECT_ENDPOINT = "https://foundary-tzuc06.services.ai.azure.com/api/projects/firstProject"
OPENAI_ENDPOINT = "https://foundary-tzuc06.openai.azure.com/openai/v1"
TENANT_ID = "16b3c013-d300-468d-ac64-7eda0820b6d3"
SUBSCRIPTION_ID = "ef669702-542a-4abc-95a6-edf9f972cd3c"
SCOPE = "https://cognitiveservices.azure.com/.default"


@dataclass(frozen=True)
class FoundryConfig:
    project_endpoint: str = PROJECT_ENDPOINT
    openai_endpoint: str = OPENAI_ENDPOINT
    deployments: tuple[str, ...] = ("gpt-5-mini", "gpt-5.4", "gpt-5.6-sol")
    tenant_id: str = TENANT_ID
    subscription_id: str = SUBSCRIPTION_ID

    def public(self) -> dict:
        auth = ("AZURE_OPENAI_API_KEY" if os.environ.get("AZURE_OPENAI_API_KEY")
                else f"az login · tenant {self.tenant_id[:8]}… · subscription {self.subscription_id[:8]}…")
        return {"project_endpoint": self.project_endpoint, "openai_endpoint": self.openai_endpoint,
                "deployments": list(self.deployments), "auth": auth,
                "tenant_id": self.tenant_id, "subscription_id": self.subscription_id}


def client(config: FoundryConfig):
    """API key from the server environment if set, otherwise az login pinned to the configured tenant/subscription."""
    from openai import OpenAI
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not key:
        from azure.identity import AzureCliCredential, get_bearer_token_provider
        # az accepts only one of tenant/subscription; the subscription fixes the tenant, verified at startup.
        credential = AzureCliCredential(subscription=config.subscription_id, process_timeout=60)
        key = get_bearer_token_provider(credential, SCOPE)
    return OpenAI(base_url=config.openai_endpoint, api_key=key, max_retries=3, timeout=300)


def complete(api, deployment: str, messages: list[dict], tools: list[dict]) -> tuple[object, dict]:
    """One chat-completions request; returns (assistant message, usage event)."""
    started = time.time()
    response = api.chat.completions.create(model=deployment, messages=messages, tools=tools)
    usage = response.usage
    cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
    reasoning = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
    event = {
        "id": response.id, "model": response.model or deployment,
        "input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens,
        "cache_read_tokens": cached, "cache_write_tokens": 0, "reasoning_tokens": reasoning,
        "fresh_input": usage.prompt_tokens - cached,
        "duration_ms": int((time.time() - started) * 1000),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return response.choices[0].message, event
