"""Deterministic sandbox implementation of the currency_convert tool."""

from __future__ import annotations

import functools
import json
from pathlib import Path

from pydantic import BaseModel, Field

from toolsmith.tools.schemas import ToolSpec, registry

_WORLDDATA_DIR = Path(__file__).parent / "worlddata"


@functools.cache
def _load_fx_rates() -> dict[str, float]:
    """Load and cache the generated fixed FX rate table (anchored to USD)."""
    return json.loads((_WORLDDATA_DIR / "fx_rates.json").read_text())


class CurrencyNotFoundError(ValueError):
    """Raised when a currency code isn't present in the sandbox FX rate table."""


class CurrencyConvertArgs(BaseModel):
    """Arguments for a currency_convert call."""

    amount: float = Field(gt=0)
    from_currency: str = Field(pattern=r"^[A-Z]{3}$")
    to_currency: str = Field(pattern=r"^[A-Z]{3}$")


class CurrencyConvertResult(BaseModel):
    """Result of a currency_convert call."""

    amount: float
    from_currency: str
    to_currency: str
    converted: float
    rate: float


def currency_convert(args: CurrencyConvertArgs) -> CurrencyConvertResult:
    """Convert an amount between two currencies using the fixed sandbox FX rate table."""
    fx_rates = _load_fx_rates()
    if args.from_currency not in fx_rates:
        raise CurrencyNotFoundError(f"unknown currency: {args.from_currency}")
    if args.to_currency not in fx_rates:
        raise CurrencyNotFoundError(f"unknown currency: {args.to_currency}")

    if args.from_currency == args.to_currency:
        return CurrencyConvertResult(
            amount=args.amount,
            from_currency=args.from_currency,
            to_currency=args.to_currency,
            converted=args.amount,
            rate=1.0,
        )

    rate = fx_rates[args.to_currency] / fx_rates[args.from_currency]
    converted = args.amount * rate

    return CurrencyConvertResult(
        amount=args.amount,
        from_currency=args.from_currency,
        to_currency=args.to_currency,
        converted=converted,
        rate=rate,
    )


# Deferred: tools.real.currency_convert imports these Args/Result classes back from this
# module, so importing it at the top would be circular.
from toolsmith.tools.real.currency_convert import currency_convert_real  # noqa: E402

registry.register(
    ToolSpec(
        name="currency_convert",
        description="Convert an amount between two currencies using a fixed sandbox FX rate table.",
        args_model=CurrencyConvertArgs,
        returns_model=CurrencyConvertResult,
        sandbox_fn=currency_convert,
        real_fn=currency_convert_real,
    )
)
