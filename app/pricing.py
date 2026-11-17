import logging
import os
from dataclasses import dataclass
from decimal import Decimal

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPricing:
    """Pricing rates for one LLM model, in USD per million tokens."""

    input_price_per_million: Decimal
    output_price_per_million: Decimal


MODEL_PRICING: dict[tuple[str, str], ModelPricing] = {}


def register_model_pricing(
    provider: str,
    model: str,
    input_price_per_million: Decimal,
    output_price_per_million: Decimal,
) -> None:
    """Register pricing for a provider and model."""
    if input_price_per_million < 0 or output_price_per_million < 0:
        raise ValueError("Pricing cannot be negative.")

    key = (provider.lower().strip(), model.strip())
    MODEL_PRICING[key] = ModelPricing(
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
    )


def get_model_pricing(provider: str, model: str) -> ModelPricing | None:
    """Get registered model pricing, or None if not configured."""
    key = (provider.lower().strip(), model.strip())
    return MODEL_PRICING.get(key)


def calculate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> Decimal:
    """Calculate total LLM API cost in USD.

    Uses Decimal for financial precision. Raises ValueError for negative tokens
    or unknown models.
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative.")

    key = (provider.lower().strip(), model.strip())
    pricing = MODEL_PRICING.get(key)

    if pricing is None:
        raise ValueError(
            f"Pricing not configured for provider={provider}, model={model}"
        )

    input_cost = (
        Decimal(input_tokens)
        * pricing.input_price_per_million
        / Decimal("1000000")
    )

    output_cost = (
        Decimal(output_tokens)
        * pricing.output_price_per_million
        / Decimal("1000000")
    )

    return input_cost + output_cost


# Standard verified provider pricing (USD per 1 million tokens)
DEFAULT_PRICING_TABLE: list[tuple[str, str, Decimal, Decimal]] = [
    # Google Gemini models
    ("gemini", "gemini-3-flash-preview", Decimal("0.50"), Decimal("3.00")),
    ("gemini", "gemini-2.5-flash", Decimal("0.10"), Decimal("0.40")),
    ("gemini", "gemini-2.0-flash", Decimal("0.10"), Decimal("0.40")),
    ("gemini", "gemini-1.5-flash", Decimal("0.075"), Decimal("0.30")),
    ("gemini", "gemini-1.5-pro", Decimal("1.25"), Decimal("5.00")),
    # Groq models
    ("groq", "openai/gpt-oss-safeguard-20b", Decimal("0.075"), Decimal("0.30")),
    ("groq", "llama-3.3-70b-versatile", Decimal("0.59"), Decimal("0.79")),
    ("groq", "llama-3.1-8b-instant", Decimal("0.05"), Decimal("0.08")),
    ("groq", "mixtral-8x7b-32768", Decimal("0.24"), Decimal("0.24")),
]

for provider, model, input_price, output_price in DEFAULT_PRICING_TABLE:
    register_model_pricing(provider, model, input_price, output_price)


def load_pricing_from_env() -> None:
    """Read configured model names and optional custom pricing from environment variables."""
    gemini_model = os.getenv("GEMINI_MODEL")
    if gemini_model:
        in_str = os.getenv("GEMINI_INPUT_PRICE_PER_MILLION") or os.getenv("GEMINI_INPUT_PRICE_PER_M")
        out_str = os.getenv("GEMINI_OUTPUT_PRICE_PER_MILLION") or os.getenv("GEMINI_OUTPUT_PRICE_PER_M")
        existing = get_model_pricing("gemini", gemini_model)
        in_price = Decimal(in_str) if in_str else (existing.input_price_per_million if existing else Decimal("0.50"))
        out_price = Decimal(out_str) if out_str else (existing.output_price_per_million if existing else Decimal("3.00"))
        register_model_pricing("gemini", gemini_model, in_price, out_price)

    groq_model = os.getenv("GROQ_MODEL")
    if groq_model:
        in_str = os.getenv("GROQ_INPUT_PRICE_PER_MILLION") or os.getenv("GROQ_INPUT_PRICE_PER_M")
        out_str = os.getenv("GROQ_OUTPUT_PRICE_PER_MILLION") or os.getenv("GROQ_OUTPUT_PRICE_PER_M")
        existing = get_model_pricing("groq", groq_model)
        in_price = Decimal(in_str) if in_str else (existing.input_price_per_million if existing else Decimal("0.075"))
        out_price = Decimal(out_str) if out_str else (existing.output_price_per_million if existing else Decimal("0.30"))
        register_model_pricing("groq", groq_model, in_price, out_price)


load_pricing_from_env()
