"""Per-model token prices (USD per 1M tokens) for cost accounting in traces.

gpt-5.4 family prices verified on developers.openai.com on 2026-10-18; the rest are list prices
at time of writing and only affect the reported cost estimate.
"""

from __future__ import annotations

PRICES: dict[str, tuple[float, float, float]] = {
    # model: (input, cached_input, output)
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.2": (1.75, 0.175, 14.00),
    "gpt-5.1": (1.25, 0.125, 10.00),
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4o-2024-08-06": (2.50, 1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
}

ALLOWED_MODELS = frozenset(PRICES)


def cost_usd(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    inp, cached, out = PRICES.get(model, (0.0, 0.0, 0.0))
    uncached = max(0, input_tokens - cached_tokens)
    return (uncached * inp + cached_tokens * cached + output_tokens * out) / 1_000_000
