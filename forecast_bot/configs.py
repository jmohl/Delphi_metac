"""
Configuration classes for forecast_bot_v2.

These mirror the v1 config objects without relying on pydantic or forecasting_tools.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class ResearchBotConfig:
    name: str
    web_search_mode: Literal["heavy", "lite", "none", "multi"] = "lite"
    research_prompt_key: str = "default"
    multi_search_models: list[str] | None = None


@dataclass
class ForecasterConfig:
    name: str
    model: str
    binary_prompt_key: str = "default"
    numeric_prompt_key: str = "default"
    multiple_choice_prompt_key: str = "default"


@dataclass
class EndToEndForecasterConfig:
    name: str
    model: str
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    binary_prompt_key: str = "default"
    numeric_prompt_key: str = "default"
    multiple_choice_prompt_key: str = "default"


@dataclass
class BotConfig:
    research_reports_per_question: int = 1
    predictions_per_research_report: int = 1
    use_research_summary_to_forecast: bool = False
    publish_reports_to_metaculus: bool = False
    folder_to_save_reports_to: str = "forecast_bot_v2/reports"
    skip_previously_forecasted_questions: bool = True
    enable_end_to_end_forecaster: bool = True
    end_to_end_forecasters_per_question: int = 0
    llms: dict[str, str] = field(default_factory=dict)


def prepare_configs(configs: list[T], target_count: int, config_type: str) -> list[T]:
    """
    Cycle or truncate configs to match the target count, mirroring v1 behavior.
    """
    if target_count <= 0:
        return []

    if len(configs) == target_count:
        return list(configs)

    if len(configs) < target_count and configs:
        cycled = [configs[i % len(configs)] for i in range(target_count)]
        logger.warning(
            f"{config_type} config count mismatch: {len(configs)} provided, "
            f"{target_count} needed. Cycling configs -> {[c.name for c in cycled]}"
        )
        return cycled

    if len(configs) > target_count:
        logger.warning(
            f"{config_type} config count mismatch: {len(configs)} provided, "
            f"only {target_count} needed. Truncating to first {target_count}."
        )
        return configs[:target_count]

    raise ValueError(f"No {config_type} configs provided to satisfy target count {target_count}.")


__all__ = [
    "ResearchBotConfig",
    "ForecasterConfig",
    "EndToEndForecasterConfig",
    "BotConfig",
    "prepare_configs",
]
