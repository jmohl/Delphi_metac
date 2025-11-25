"""
Shared helpers for forecast_bot.

Stub created during scaffolding.
"""


def clean_indents(text: str) -> str:
    """
    Minimal replacement for forecasting_tools.clean_indents.
    """
    lines = text.splitlines()
    stripped = [line.lstrip() for line in lines]
    return "\n".join(stripped)

