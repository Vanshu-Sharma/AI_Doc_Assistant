#!/usr/bin/env python3
"""Evaluation script for Assignment 7 – Resume Information Extraction.

This script evaluates extraction accuracy against the 20 trainer-provided resumes.
Expected format:
    Evaluation Results
    ------------------
    Total resumes: 20
    Valid responses: 19/20
    Fields correct: 74/80
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from app.llm import extract_resume  # noqa: E402
from app.schema import ResumeData  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_extract")


def normalize_string(val: str | None) -> str | None:
    if val is None:
        return None
    cleaned = str(val).strip()
    return cleaned.lower() if cleaned else None


def normalize_skills(skills: list[str] | None) -> list[str]:
    if not skills:
        return []
    return sorted({str(s).strip().lower() for s in skills if str(s).strip()})


def compare_field(field_name: str, actual_val, expected_val) -> bool:
    if field_name == "skills":
        return normalize_skills(actual_val) == normalize_skills(expected_val)
    elif field_name == "email":
        return normalize_string(actual_val) == normalize_string(expected_val)
    elif field_name == "phone":
        if actual_val is None and expected_val is None:
            return True
        if actual_val is None or expected_val is None:
            return False
        act_digits = "".join(c for c in str(actual_val) if c.isdigit())
        exp_digits = "".join(c for c in str(expected_val) if c.isdigit())
        if act_digits and exp_digits:
            return act_digits == exp_digits
        return str(actual_val).strip() == str(expected_val).strip()
    else:
        if actual_val is None and expected_val is None:
            return True
        if actual_val is None or expected_val is None:
            return False
        return str(actual_val).strip().lower() == str(expected_val).strip().lower()


async def run_evaluation(
    expected_path: Path,
    data_dir: Path,
    provider: str,
    pause_seconds: float,
) -> None:
    if not expected_path.exists():
        print(f"Error: Expected file not found at '{expected_path}'.")
        print(
            "Please place the 20 trainer-provided resumes in data/resumes/ "
            "and create data/resumes/expected.json before running evaluation."
        )
        sys.exit(1)

    try:
        with open(expected_path, encoding="utf-8") as f:
            expected_data = json.load(f)
    except Exception as exc:
        print(f"Error reading '{expected_path}': {exc}")
        sys.exit(1)

    items: list[tuple[str, str, dict]] = []

    if isinstance(expected_data, dict):
        for key, exp_obj in expected_data.items():
            resume_text = exp_obj.get("resume_text")
            if not resume_text:
                file_path = data_dir / key
                if not file_path.exists() and not key.endswith(".txt"):
                    file_path = data_dir / f"{key}.txt"
                if file_path.exists():
                    resume_text = file_path.read_text(encoding="utf-8")
                else:
                    logger.warning("Resume file not found: %s", file_path)
                    continue
            items.append((key, resume_text, exp_obj))
    elif isinstance(expected_data, list):
        for idx, item in enumerate(expected_data):
            key = item.get("filename") or item.get("id") or f"resume_{idx + 1}"
            exp_obj = item.get("expected") or item
            resume_text = item.get("resume_text")
            if not resume_text and "filename" in item:
                file_path = data_dir / item["filename"]
                if file_path.exists():
                    resume_text = file_path.read_text(encoding="utf-8")
            if not resume_text:
                logger.warning("Resume text not found for entry: %s", key)
                continue
            items.append((key, resume_text, exp_obj))
    else:
        print("Error: Invalid expected.json format. Expected a JSON dict or list.")
        sys.exit(1)

    total_resumes = len(items)
    if total_resumes == 0:
        print("Error: No resumes found to evaluate.")
        sys.exit(1)

    valid_responses = 0
    fields_correct = 0
    total_fields = total_resumes * 4
    field_names = ["name", "email", "phone", "skills"]

    logger.info(
        "Starting evaluation of %d resumes using provider '%s' (pause=%.1fs)...",
        total_resumes,
        provider,
        pause_seconds,
    )

    for idx, (key, resume_text, expected_dict) in enumerate(items, start=1):
        logger.info("[%d/%d] Evaluating '%s'...", idx, total_resumes, key)
        actual_output: ResumeData | None = None

        try:
            actual_output = await extract_resume(
                resume_text=resume_text,
                provider=provider,
            )
        except Exception as exc:
            logger.error("Extraction failed for '%s': %s", key, exc)

        if actual_output is not None and isinstance(actual_output, ResumeData):
            valid_responses += 1

            for field in field_names:
                act_val = getattr(actual_output, field)
                exp_val = expected_dict.get(field)
                if compare_field(field, act_val, exp_val):
                    fields_correct += 1
                else:
                    logger.info(
                        "Mismatch for '%s' field '%s': expected=%r, actual=%r",
                        key,
                        field,
                        exp_val,
                        act_val,
                    )

        if idx < total_resumes:
            time.sleep(pause_seconds)

    print()
    print("Evaluation Results")
    print("------------------")
    print(f"Total resumes: {total_resumes}")
    print(f"Valid responses: {valid_responses}/{total_resumes}")
    print(f"Fields correct: {fields_correct}/{total_fields}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate resume structured extraction on dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "resumes",
        help="Directory containing resume text files.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=PROJECT_ROOT / "data" / "resumes" / "expected.json",
        help="Path to expected.json file.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=os.getenv("DEFAULT_PROVIDER", "gemini"),
        choices=["gemini", "groq"],
        help="LLM provider to use for extraction (default: gemini).",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.5,
        help="Pause between requests in seconds to respect rate limits (default: 1.5).",
    )

    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            expected_path=args.expected,
            data_dir=args.data_dir,
            provider=args.provider,
            pause_seconds=args.pause,
        )
    )


if __name__ == "__main__":
    main()
