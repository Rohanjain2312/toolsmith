"""Deterministic sandbox implementation of the currency_convert tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from toolsmith.tools.schemas import ToolSpec, registry

# Fixed FX rate table anchored to USD; replaced by loading
# src/toolsmith/tools/sandbox/worlddata/fx_rates.json in task P1-T14.
_FX_RATES: dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 149.5,
    "INR": 83.1,
    "AUD": 1.52,
}


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
    if args.from_currency not in _FX_RATES:
        raise CurrencyNotFoundError(f"unknown currency: {args.from_currency}")
    if args.to_currency not in _FX_RATES:
        raise CurrencyNotFoundError(f"unknown currency: {args.to_currency}")

    if args.from_currency == args.to_currency:
        return CurrencyConvertResult(
            amount=args.amount,
            from_currency=args.from_currency,
            to_currency=args.to_currency,
            converted=args.amount,
            rate=1.0,
        )

    rate = _FX_RATES[args.to_currency] / _FX_RATES[args.from_currency]
    converted = args.amount * rate

    return CurrencyConvertResult(
        amount=args.amount,
        from_currency=args.from_currency,
        to_currency=args.to_currency,
        converted=converted,
        rate=rate,
    )


registry.register(
    ToolSpec(
        name="currency_convert",
        description="Convert an amount between two currencies using a fixed sandbox FX rate table.",
        args_model=CurrencyConvertArgs,
        returns_model=CurrencyConvertResult,
        sandbox_fn=currency_convert,
    )
)
