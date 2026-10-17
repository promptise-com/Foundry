"""Cost estimates from the manifest's ``[pricing]`` table.

There is no built-in price list: prices change, and a wrong number is worse
than none.  ``[pricing]`` maps model names (``fnmatch`` globs allowed) to USD
per one million input and output tokens::

    [pricing]
    "gpt-5-mini*" = { input = 0.25, output = 2.0 }
"""

from __future__ import annotations

from collections.abc import Mapping
from fnmatch import fnmatchcase

from .manifest import Price

__all__ = ["PriceTable"]


class PriceTable:
    """Looks up a model's price and estimates the cost of a call.

    A model is matched by its exact name first, then by its name without a
    ``provider:`` prefix, then against each glob in table order.
    """

    def __init__(self, prices: Mapping[str, Price] | None = None) -> None:
        self._prices = dict(prices or {})
        self._cache: dict[str, Price | None] = {}

    def __bool__(self) -> bool:
        return bool(self._prices)

    def price(self, model: str | None) -> Price | None:
        """The price of *model*, or ``None`` when the table has none for it."""
        if not model:
            return None
        if model in self._cache:
            return self._cache[model]
        found = self._prices.get(model)
        bare = model.split(":", 1)[1] if ":" in model else model
        if found is None:
            found = self._prices.get(bare)
        if found is None:
            for pattern, price in self._prices.items():
                if fnmatchcase(model, pattern) or fnmatchcase(bare, pattern):
                    found = price
                    break
        self._cache[model] = found
        return found

    def cost(self, model: str | None, prompt_tokens: int, completion_tokens: int) -> float | None:
        """The estimated USD cost of one call, or ``None`` when *model* has no price."""
        price = self.price(model)
        if price is None:
            return None
        return (prompt_tokens * price.input + completion_tokens * price.output) / 1_000_000
