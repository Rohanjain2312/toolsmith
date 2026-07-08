"""Tests for the LLM-judge rubric scorer and its human-calibration mode (network always mocked)."""

from __future__ import annotations

from pathlib import Path

import pytest

from toolsmith.eval import judge as judge_module
from toolsmith.eval.judge import (
    JudgeRequestError,
    JudgeScore,
    calibrate_against_human_csv,
    judge_final_answer,
    parse_judge_response,
)

# --- parse_judge_response ---


def test_parse_judge_response_happy_path() -> None:
    score = parse_judge_response('{"score": 4, "reasoning": "Correct and clear."}')

    assert score == JudgeScore(score=4, reasoning="Correct and clear.")


def test_parse_judge_response_malformed_json_raises() -> None:
    with pytest.raises(JudgeRequestError):
        parse_judge_response("not json")


def test_parse_judge_response_missing_field_raises() -> None:
    with pytest.raises(JudgeRequestError):
        parse_judge_response('{"score": 4}')


def test_parse_judge_response_out_of_range_score_raises() -> None:
    with pytest.raises(JudgeRequestError):
        parse_judge_response('{"score": 7, "reasoning": "x"}')


def test_parse_judge_response_non_integer_score_raises() -> None:
    with pytest.raises(JudgeRequestError):
        parse_judge_response('{"score": "great", "reasoning": "x"}')


# --- judge_final_answer ---


def test_judge_final_answer_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        judge_module,
        "_post_anthropic",
        lambda body, api_key: {
            "content": [{"type": "text", "text": '{"score": 5, "reasoning": "Great answer."}'}]
        },
    )

    score = judge_final_answer("Weather in Paris?", "Sunny, 22C.")

    assert score.score == 5


def test_judge_final_answer_prompt_never_names_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    captured: dict = {}

    def fake_post(body: dict, api_key: str) -> dict:
        captured.update(body)
        return {"content": [{"type": "text", "text": '{"score": 3, "reasoning": "ok"}'}]}

    monkeypatch.setattr(judge_module, "_post_anthropic", fake_post)

    judge_final_answer("Weather in Paris?", "Sunny, 22C.")

    prompt_text = captured["messages"][0]["content"]
    assert "Weather in Paris?" in prompt_text
    assert "Sunny, 22C." in prompt_text
    for name in ("GPT", "Qwen", "Claude", "gpt-4o", "SFT", "GRPO"):
        assert name not in prompt_text


def test_judge_final_answer_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(JudgeRequestError):
        judge_final_answer("x", "y")


def test_judge_final_answer_unexpected_response_shape_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(judge_module, "_post_anthropic", lambda body, api_key: {"bad": True})

    with pytest.raises(JudgeRequestError):
        judge_final_answer("x", "y")


# --- calibrate_against_human_csv ---


def test_calibrate_against_human_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "human_scores.csv"
    csv_path.write_text(
        "user_prompt,final_answer,human_score\n"
        "Weather in Paris?,Sunny 22C,4\n"
        "Distance to Tokyo?,5000 km,3\n"
    )

    def fake_judge(user_prompt: str, final_answer: str) -> JudgeScore:
        return JudgeScore(score=4, reasoning="stub")

    result = calibrate_against_human_csv(csv_path, judge_fn=fake_judge)

    assert result.n == 2
    assert result.mean_absolute_error == pytest.approx(0.5)  # |4-4| then |4-3|, avg 0.5
    assert result.exact_match_rate == pytest.approx(0.5)


def test_calibrate_against_human_csv_empty_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("user_prompt,final_answer,human_score\n")

    result = calibrate_against_human_csv(csv_path, judge_fn=judge_final_answer)

    assert result.n == 0
