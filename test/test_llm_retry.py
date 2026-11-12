
import asyncio

import pytest

from app import llm


class MockAPIError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"API error: {status_code}")
        self.status_code = status_code


def run_generate():
    async def collect():
        return [
            chunk
            async for chunk in llm.generate(
                prompt="Explain Python generators.",
                provider="gemini",
            )
        ]

    return asyncio.run(collect())


@pytest.fixture
def no_sleep(monkeypatch):
    async def fake_sleep(delay):
        pass

    monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_retryable_status_codes(
    monkeypatch,
    no_sleep,
    status_code,
):
    attempts = 0

    async def fake_generate_once(prompt, provider):
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            raise MockAPIError(status_code)

        yield "Success"

    monkeypatch.setattr(
        llm,
        "_generate_once",
        fake_generate_once,
    )

    result = run_generate()

    assert result == ["Success"]
    assert attempts == 2


def test_timeout_retry(monkeypatch, no_sleep):
    attempts = 0

    async def fake_generate_once(prompt, provider):
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            raise TimeoutError("Request timed out")

        yield "Recovered"

    monkeypatch.setattr(
        llm,
        "_generate_once",
        fake_generate_once,
    )

    result = run_generate()

    assert result == ["Recovered"]
    assert attempts == 2


def test_non_retryable_error(monkeypatch, no_sleep):
    attempts = 0

    async def fake_generate_once(prompt, provider):
        nonlocal attempts
        attempts += 1
        raise ValueError("Invalid provider")
        yield

    monkeypatch.setattr(
        llm,
        "_generate_once",
        fake_generate_once,
    )

    with pytest.raises(ValueError):
        run_generate()

    assert attempts == 1


def test_maximum_retries(monkeypatch, no_sleep):
    attempts = 0

    async def fake_generate_once(prompt, provider):
        nonlocal attempts
        attempts += 1
        raise MockAPIError(503)
        yield

    monkeypatch.setattr(
        llm,
        "_generate_once",
        fake_generate_once,
    )

    with pytest.raises(MockAPIError):
        run_generate()

    assert attempts == llm.MAX_RETRIES


def test_no_retry_after_partial_response(
    monkeypatch,
    no_sleep,
):
    attempts = 0

    async def fake_generate_once(prompt, provider):
        nonlocal attempts
        attempts += 1

        yield "Partial response"

        raise MockAPIError(503)

    monkeypatch.setattr(
        llm,
        "_generate_once",
        fake_generate_once,
    )

    with pytest.raises(MockAPIError):
        run_generate()

    assert attempts == 1
