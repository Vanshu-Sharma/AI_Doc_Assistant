from decimal import Decimal

import pytest

from app.pricing import (
    calculate_cost,
    get_model_pricing,
    load_pricing_from_env,
    register_model_pricing,
)


def test_calculate_cost():
    register_model_pricing(
        "gemini",
        "test-model",
        Decimal("0.10"),
        Decimal("0.40"),
    )

    cost = calculate_cost(
        provider="gemini",
        model="test-model",
        input_tokens=10000,
        output_tokens=5000,
    )

    assert cost == Decimal("0.003")


def test_input_token_cost_only():
    register_model_pricing(
        "gemini",
        "input-only-model",
        Decimal("0.50"),
        Decimal("3.00"),
    )

    cost = calculate_cost(
        provider="gemini",
        model="input-only-model",
        input_tokens=1000000,  # 1M tokens
        output_tokens=0,
    )

    assert cost == Decimal("0.50")


def test_output_token_cost_only():
    register_model_pricing(
        "groq",
        "output-only-model",
        Decimal("0.10"),
        Decimal("0.80"),
    )

    cost = calculate_cost(
        provider="groq",
        model="output-only-model",
        input_tokens=0,
        output_tokens=500000,  # 0.5M tokens
    )

    assert cost == Decimal("0.40")


def test_negative_tokens():
    with pytest.raises(ValueError, match="Token counts cannot be negative"):
        calculate_cost("gemini", "test-model", -10, 100)

    with pytest.raises(ValueError, match="Token counts cannot be negative"):
        calculate_cost("gemini", "test-model", 100, -10)


def test_unknown_model():
    with pytest.raises(ValueError, match="Pricing not configured"):
        calculate_cost("gemini", "unknown-model", 100, 100)


def test_negative_pricing_registration():
    with pytest.raises(ValueError, match="Pricing cannot be negative"):
        register_model_pricing("gemini", "bad-price-model", Decimal("-0.10"), Decimal("0.40"))

    with pytest.raises(ValueError, match="Pricing cannot be negative"):
        register_model_pricing("gemini", "bad-price-model", Decimal("0.10"), Decimal("-0.40"))


def test_load_pricing_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "custom-gemini-model")
    monkeypatch.setenv("GEMINI_INPUT_PRICE_PER_MILLION", "0.25")
    monkeypatch.setenv("GEMINI_OUTPUT_PRICE_PER_MILLION", "1.50")

    load_pricing_from_env()

    pricing = get_model_pricing("gemini", "custom-gemini-model")
    assert pricing is not None
    assert pricing.input_price_per_million == Decimal("0.25")
    assert pricing.output_price_per_million == Decimal("1.50")

    cost = calculate_cost("gemini", "custom-gemini-model", 1000000, 1000000)
    assert cost == Decimal("1.75")
