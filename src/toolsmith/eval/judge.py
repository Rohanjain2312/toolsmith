"""LLM-judge rubric scoring (1-5) of final answers via the Anthropic API, identity-blinded."""

from __future__ import annotations

import csv
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5"

# Deliberately omits any system identity: the judge is told only the request and the answer
# text, never which model produced it — this is what "blinded" means here.
RUBRIC_PROMPT = (
    "You are scoring the quality of an AI assistant's final answer to a travel-ops request, "
    "on a scale of 1 (unhelpful or wrong) to 5 (excellent, fully correct and helpful).\n\n"
    "User request: {user_prompt}\n"
    "Assistant's final answer: {final_answer}\n\n"
    'Respond with ONLY a JSON object: {{"score": <1-5 integer>, "reasoning": "<one sentence>"}}'
)


class JudgeRequestError(RuntimeError):
    """Raised when the judge API request fails or its response can't be parsed/used."""


@dataclass(frozen=True)
class JudgeScore:
    """One rubric judgment: a 1-5 score plus a short justification."""

    score: int
    reasoning: str


def _post_anthropic(body: dict[str, Any], api_key: str) -> dict[str, Any]:
    """POST a request body to the Anthropic Messages API and return the parsed JSON response."""
    request = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except (OSError, urllib.error.URLError) as exc:
        raise JudgeRequestError(f"judge API request failed: {exc}") from exc


def parse_judge_response(raw_text: str) -> JudgeScore:
    """Parse the judge model's raw text response into a JudgeScore."""
    try:
        data = json.loads(raw_text.strip())
        score = int(data["score"])
        reasoning = str(data["reasoning"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise JudgeRequestError(f"could not parse judge response: {raw_text!r}") from exc
    if not 1 <= score <= 5:
        raise JudgeRequestError(f"judge score out of range 1-5: {score}")
    return JudgeScore(score=score, reasoning=reasoning)


def judge_final_answer(
    user_prompt: str, final_answer: str, model: str = DEFAULT_MODEL
) -> JudgeScore:
    """Blind rubric-score one final answer (1-5) via the Anthropic API.

    Model identity is never included in the prompt (see RUBRIC_PROMPT), so the judge cannot
    see which system produced the answer.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise JudgeRequestError("ANTHROPIC_API_KEY is not set")

    prompt = RUBRIC_PROMPT.format(user_prompt=user_prompt, final_answer=final_answer)
    body = {"model": model, "max_tokens": 256, "messages": [{"role": "user", "content": prompt}]}
    payload = _post_anthropic(body, api_key)

    try:
        raw_text = "".join(
            block["text"] for block in payload["content"] if block["type"] == "text"
        )
    except (KeyError, TypeError) as exc:
        raise JudgeRequestError(f"unexpected judge API response shape: {payload}") from exc
    return parse_judge_response(raw_text)


@dataclass(frozen=True)
class CalibrationResult:
    """Agreement between judge scores and human-labeled scores over the same items."""

    mean_absolute_error: float
    exact_match_rate: float
    n: int


def calibrate_against_human_csv(
    csv_path: Path, judge_fn: Callable[[str, str], JudgeScore] = judge_final_answer
) -> CalibrationResult:
    """Run the judge over a human-scored CSV (columns: user_prompt, final_answer, human_score)
    and report how closely the judge's scores track the human labels."""
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    if not rows:
        return CalibrationResult(mean_absolute_error=0.0, exact_match_rate=0.0, n=0)

    errors = []
    exact_matches = 0
    for row in rows:
        judged = judge_fn(row["user_prompt"], row["final_answer"])
        human_score = int(row["human_score"])
        errors.append(abs(judged.score - human_score))
        if judged.score == human_score:
            exact_matches += 1

    n = len(rows)
    return CalibrationResult(
        mean_absolute_error=sum(errors) / n, exact_match_rate=exact_matches / n, n=n
    )
