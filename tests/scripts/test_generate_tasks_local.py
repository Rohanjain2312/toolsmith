"""Tests for the local (no live LLM API) task generator.

The most important property tested here isn't template phrasing (that's editorial, not
verifiable) — it's that every generated goal_spec's tool_was_called_with conditions actually
execute successfully against the real sandbox registry. That's what "grounded in real world
data" means in practice: a condition an agent could actually satisfy, not just schema-valid JSON.
"""

from __future__ import annotations

import random

from scripts.generate_tasks_local import generate_all
from scripts.task_generation import common, t1, t2, t3, t4

import toolsmith.tools.sandbox  # noqa: F401  (registers all 12 sandbox tools)
from toolsmith.data.taskspec import ToolWasCalledWithCondition
from toolsmith.env.executor import execute_tool_call

_ALL_TOOL_NAMES = {
    "geocode_city",
    "weather_lookup",
    "flight_search",
    "currency_convert",
    "timezone_info",
    "calendar_create_event",
    "country_info",
    "poi_search",
    "distance_calc",
    "packing_rules",
    "unit_convert",
    "datetime_math",
}


def _assert_goal_spec_executes(specs) -> None:
    for spec in specs:
        for condition in spec.goal_spec:
            assert isinstance(condition, ToolWasCalledWithCondition)
            assert condition.tool_name in _ALL_TOOL_NAMES
            result = execute_tool_call(condition.tool_name, condition.args)
            assert result.ok, (
                f"{spec.id} ({spec.tier}): {condition.tool_name}({condition.args}) failed: "
                f"{result.error}"
            )


def test_common_build_spec_is_schema_valid() -> None:
    spec = common.build_spec(
        "T1", "test prompt", [common.tool_cond("geocode_city", {"city": "Paris"})]
    )
    assert spec.tier == "T1"
    assert spec.split == "train"
    assert spec.min_steps == 0


def test_common_dedupe_drops_case_insensitive_duplicates() -> None:
    paris_cond = [common.tool_cond("geocode_city", {"city": "Paris"})]
    tokyo_cond = [common.tool_cond("geocode_city", {"city": "Tokyo"})]
    a = common.build_spec("T1", "Find Paris.", paris_cond)
    b = common.build_spec("T1", "find paris.", paris_cond)
    c = common.build_spec("T1", "Find Tokyo.", tokyo_cond)

    deduped = common.dedupe([a, b, c])

    assert len(deduped) == 2


def test_t1_generate_produces_valid_executable_tasks() -> None:
    rng = random.Random(common.SEED)
    world = common.load_world()
    specs = t1.generate(rng, world, 5)

    assert all(spec.tier == "T1" for spec in specs)
    assert all(len(spec.goal_spec) == 1 for spec in specs)
    _assert_goal_spec_executes(specs)


def test_t2_generate_produces_valid_executable_tasks() -> None:
    rng = random.Random(common.SEED)
    world = common.load_world()
    specs = t2.generate(rng, world, 5)

    assert all(spec.tier == "T2" for spec in specs)
    assert all(len(spec.goal_spec) == 2 for spec in specs)
    _assert_goal_spec_executes(specs)


def test_t3_generate_produces_valid_executable_tasks() -> None:
    rng = random.Random(common.SEED)
    world = common.load_world()
    specs = t3.generate(rng, world, 5)

    assert all(spec.tier == "T3" for spec in specs)
    assert all(len(spec.goal_spec) == 3 for spec in specs)
    _assert_goal_spec_executes(specs)


def test_t4_generate_produces_valid_executable_tasks() -> None:
    rng = random.Random(common.SEED)
    world = common.load_world()
    specs = t4.generate(rng, world, 5)

    assert all(spec.tier == "T4" for spec in specs)
    assert all(len(spec.goal_spec) == 4 for spec in specs)
    _assert_goal_spec_executes(specs)


def test_generate_all_covers_all_four_tiers() -> None:
    specs = generate_all(per_tier_target=20)

    tiers = {spec.tier for spec in specs}
    assert tiers == {"T1", "T2", "T3", "T4"}
    assert len(specs) > 0


def test_generate_all_ids_are_unique() -> None:
    specs = generate_all(per_tier_target=20)

    ids = [spec.id for spec in specs]
    assert len(ids) == len(set(ids))


def test_generate_all_is_deterministic_given_same_seed() -> None:
    first = generate_all(seed=42, per_tier_target=20)
    second = generate_all(seed=42, per_tier_target=20)

    assert [s.user_prompt for s in first] == [s.user_prompt for s in second]
    assert [s.goal_spec for s in first] == [s.goal_spec for s in second]
