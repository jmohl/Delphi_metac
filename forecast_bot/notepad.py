"""
Config tracking utilities for forecast_bot_v2.

Recreates the queue-based selection logic from v1's MultiConfigNotepad without pydantic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from forecast_bot_v2.configs import (
    EndToEndForecasterConfig,
    ForecasterConfig,
    ResearchBotConfig,
    prepare_configs,
)
from forecast_bot_v2.questions import MetaculusQuestion


@dataclass
class Notepad:
    """
    Tracks per-question config state and consumption order.
    """

    question: MetaculusQuestion
    research_configs_queue: list[ResearchBotConfig] = field(default_factory=list)
    forecaster_configs_per_research: list[list[ForecasterConfig]] = field(default_factory=list)
    end_to_end_configs_queue: list[EndToEndForecasterConfig] = field(default_factory=list)

    def get_next_research_config(self) -> ResearchBotConfig:
        if not self.research_configs_queue:
            raise IndexError("No research configs remaining.")
        return self.research_configs_queue.pop(0)

    def get_next_forecaster_config(self) -> ForecasterConfig:
        if not self.forecaster_configs_per_research:
            raise IndexError("No forecaster configs available for this research report.")

        current_research_configs = self.forecaster_configs_per_research[0]
        if not current_research_configs:
            raise IndexError("Forecaster configs exhausted for this research report.")

        config = current_research_configs.pop(0)
        if not current_research_configs:
            self.forecaster_configs_per_research.pop(0)
        return config

    def get_next_end_to_end_config(self) -> EndToEndForecasterConfig:
        if not self.end_to_end_configs_queue:
            raise IndexError("No end-to-end forecaster configs remaining.")
        return self.end_to_end_configs_queue.pop(0)


def build_notepad(
    question: MetaculusQuestion,
    research_configs: list[ResearchBotConfig],
    forecaster_configs: list[ForecasterConfig],
    end_to_end_configs: list[EndToEndForecasterConfig],
    research_reports_per_question: int,
    predictions_per_research_report: int,
    end_to_end_forecasters_per_question: int,
) -> Notepad:
    """
    Create a Notepad with cycled/truncated config queues matching desired counts.
    """
    research_queue: list[ResearchBotConfig] = []
    forecaster_per_research: list[list[ForecasterConfig]] = []

    if research_reports_per_question > 0:
        research_queue = prepare_configs(
            research_configs, research_reports_per_question, "research"
        )

        if predictions_per_research_report > 0:
            forecaster_template = prepare_configs(
                forecaster_configs, predictions_per_research_report, "forecaster"
            )
            forecaster_per_research = [list(forecaster_template) for _ in range(research_reports_per_question)]

    end_to_end_queue = (
        prepare_configs(end_to_end_configs, end_to_end_forecasters_per_question, "end_to_end")
        if end_to_end_forecasters_per_question > 0
        else []
    )

    return Notepad(
        question=question,
        research_configs_queue=research_queue,
        forecaster_configs_per_research=forecaster_per_research,
        end_to_end_configs_queue=end_to_end_queue,
    )
