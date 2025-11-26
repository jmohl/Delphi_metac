"""
Orchestrates the v2 forecasting pipeline without forecasting_tools.

Usage example (from repo root):
    PYTHONPATH=. python -m forecast_bot.main --mode test_questions --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable

from dotenv import load_dotenv

from forecast_bot.aggregate import AggregationModule
from forecast_bot.configs import (
    BotConfig,
    EndToEndForecasterConfig,
    ForecasterConfig,
    ResearchBotConfig,
)
from forecast_bot.metaculus_client import MetaculusClient
from forecast_bot.notepad import build_notepad, Notepad
from forecast_bot.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    convert_forecasting_tools_question,
)
from forecast_bot.research import ResearchModule
from forecast_bot.forecast import ForecastModule
from forecast_bot.reporting import ForecastReport, Reporter

logger = logging.getLogger(__name__)

# Load environment variables from a .env file
load_dotenv()


TEST_QUESTION_URLS = [
    "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
    #"https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
    #"https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
    #"https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
]


@dataclass
class ResearchPredictionCollection:
    """
    Bundles a research run with its forecasts for aggregation/reporting.
    """

    name: str
    research: str
    predictions: list


class DelphiV2Bot:
    """
    Minimal orchestrator for the v2 pipeline.
    """

    _max_concurrent_forecasts = 5  # Limit parallel end-to-end forecasts
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_forecasts)

    def __init__(
        self,
        bot_config: BotConfig,
        research_configs: list[ResearchBotConfig],
        forecaster_configs: list[ForecasterConfig],
        end_to_end_configs: list[EndToEndForecasterConfig],
        metaculus_client: MetaculusClient | None = None,
    ) -> None:
        self.config = bot_config
        self.research_configs = research_configs
        self.forecaster_configs = forecaster_configs
        self.end_to_end_configs = end_to_end_configs

        self.llms: dict[str, str] = bot_config.llms or {}

        self.metaculus = metaculus_client or MetaculusClient()
        self.research_module = ResearchModule(parent_bot=self)
        self.forecast_module = ForecastModule(parent_bot=self)
        self.aggregation_module = AggregationModule(parent_bot=self)
        self.reporter = Reporter(folder=bot_config.folder_to_save_reports_to, metaculus_client=self.metaculus)

        self._notepads: dict[int, Notepad] = {}

    async def forecast_question(self, question: MetaculusQuestion, dry_run: bool = True) -> ForecastReport:
        logger.info(f"{'='*80}")
        logger.info(f"Starting forecast for question {question.id}: {question.question_text[:80]}...")
        logger.info(f"URL: {question.page_url}")
        logger.info(f"{'='*80}")

        notepad = await self._get_notepad(question)

        research_collections: list[ResearchPredictionCollection] = []
        traditional_predictions: list[Any] = []

        # Traditional research + forecast path
        total_research = self.config.research_reports_per_question
        total_forecasts_per_research = self.config.predictions_per_research_report
        logger.info(f"Starting traditional pipeline: {total_research} research report(s), {total_forecasts_per_research} forecast(s) per report")

        for research_idx in range(self.config.research_reports_per_question):
            logger.info(f"--- Research Report {research_idx + 1}/{total_research} ---")
            research_config = notepad.get_next_research_config()
            logger.info(f"Starting research with config: {research_config.name}")
            research_text = await self.research_module.run_research(question, research_config)
            logger.info(f"Research completed for {research_config.name}")

            preds_for_research = []
            logger.info(f"Generating {total_forecasts_per_research} forecast(s) based on this research...")
            for forecast_idx in range(self.config.predictions_per_research_report):
                logger.info(f"  Forecast {forecast_idx + 1}/{total_forecasts_per_research}...")
                pred = await self._run_forecast_for_question(question, research_text)
                preds_for_research.append(pred)
                traditional_predictions.append(pred.prediction_value)

            research_collections.append(
                ResearchPredictionCollection(
                    name=research_config.name,
                    research=research_text,
                    predictions=preds_for_research,
                )
            )

        # End-to-end forecasts (optional)
        end_to_end_predictions = []
        total_end_to_end = self.config.end_to_end_forecasters_per_question
        if total_end_to_end > 0:
            logger.info(f"--- End-to-End Forecasts ---")
            logger.info(f"Starting {total_end_to_end} end-to-end forecast(s) in parallel...")

            # Get all configs upfront before launching parallel tasks
            end_to_end_configs = [
                notepad.get_next_end_to_end_config()
                for _ in range(self.config.end_to_end_forecasters_per_question)
            ]

            async def _run_single_end_to_end(config, idx):
                async with self._concurrency_limiter:
                    logger.info(f"  End-to-end forecast {idx + 1}/{total_end_to_end} with config: {config.name}")
                    return await self._run_end_to_end_forecast(question, config)

            # Launch forecasts in parallel (bounded by _concurrency_limiter)
            tasks = [
                asyncio.create_task(_run_single_end_to_end(cfg, idx))
                for idx, cfg in enumerate(end_to_end_configs)
            ]
            end_to_end_predictions = await asyncio.gather(*tasks)

            # Add prediction values to traditional_predictions list
            traditional_predictions.extend([pred.prediction_value for pred in end_to_end_predictions])

        logger.info(f"--- Aggregation ---")
        logger.info(f"Aggregating {len(traditional_predictions)} prediction(s)...")
        aggregated = self.aggregation_module.aggregate_predictions(traditional_predictions, question)
        logger.info(f"Aggregated prediction: {aggregated}")

        logger.info(f"--- Creating Unified Explanation ---")
        logger.info(f"Synthesizing explanations from {len(research_collections)} research report(s) and {len(end_to_end_predictions)} end-to-end forecast(s)...")
        explanation = await self.aggregation_module.create_unified_explanation(
            question=question,
            research_prediction_collections=research_collections,
            aggregated_prediction=aggregated,
            end_to_end_predictions=end_to_end_predictions,
            summarizer_model=self.llms.get("summarizer") if self.llms else None,
        )
        logger.info("Unified explanation created")

        logger.info(f"--- Creating Report ---")
        report = ForecastReport(
            question=question,
            prediction=aggregated,
            explanation=explanation,
            research=[rc.research for rc in research_collections],
            forecaster_reasonings=[p.reasoning for rc in research_collections for p in rc.predictions]
            + [p.reasoning for p in end_to_end_predictions],
        )

        logger.info(f"--- Saving and Publishing ---")
        self.reporter.save_report(report)
        if self.config.publish_reports_to_metaculus:
            logger.info("Publishing to Metaculus...")
            self.reporter.post_forecast(report, dry_run=dry_run)
            self.reporter.post_comment(report, dry_run=dry_run)
        else:
            logger.info("Publishing to Metaculus is disabled")

        logger.info(f"{'='*80}")
        logger.info(f"Completed forecast for question {question.id}")
        logger.info(f"{'='*80}")

        return report

    async def _run_forecast_for_question(self, question: MetaculusQuestion, research: str):
        if isinstance(question, BinaryQuestion):
            return await self.forecast_module.run_forecast_on_binary(question, research)
        if isinstance(question, MultipleChoiceQuestion):
            return await self.forecast_module.run_forecast_on_multiple_choice(question, research)
        if isinstance(question, NumericQuestion):
            return await self.forecast_module.run_forecast_on_numeric(question, research)
        raise NotImplementedError(f"Unsupported question type: {type(question)}")

    async def _run_end_to_end_forecast(self, question: MetaculusQuestion, config: EndToEndForecasterConfig):
        if isinstance(question, BinaryQuestion):
            return await self.forecast_module.run_end_to_end_forecast_on_binary(question, config)
        if isinstance(question, MultipleChoiceQuestion):
            return await self.forecast_module.run_end_to_end_forecast_on_multiple_choice(question, config)
        if isinstance(question, NumericQuestion):
            return await self.forecast_module.run_end_to_end_forecast_on_numeric(question, config)
        raise NotImplementedError(f"Unsupported question type: {type(question)}")

    async def _get_notepad(self, question: MetaculusQuestion) -> Notepad:
        if question.id in self._notepads:
            return self._notepads[question.id]

        npd = build_notepad(
            question=question,
            research_configs=self.research_configs,
            forecaster_configs=self.forecaster_configs,
            end_to_end_configs=self.end_to_end_configs,
            research_reports_per_question=self.config.research_reports_per_question,
            predictions_per_research_report=self.config.predictions_per_research_report,
            end_to_end_forecasters_per_question=self.config.end_to_end_forecasters_per_question,
        )
        self._notepads[question.id] = npd
        return npd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Delphi v2 forecasting bot.")
    parser.add_argument(
        "--mode",
        choices=["test_questions", "urls", "tournament"],
        default="test_questions",
        help="Which source of questions to forecast.",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        help="Explicit Metaculus question URLs when --mode urls.",
    )
    parser.add_argument(
        "--tournament-id",
        help="(Optional) Tournament id or slug for --mode tournament. If omitted, uses both CURRENT_MINIBENCH_ID and CURRENT_AI_COMPETITION_ID.",
    )
    parser.add_argument(
        "--config",
        default="forecast_bot/default_configs.py",
        help="Path to a config module exposing get_default_configs().",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Post forecasts/comments to Metaculus (otherwise dry run).",
    )
    return parser.parse_args()


def load_configs(config_path: str):
    """
    Load bot/research/forecast configs from a Python module exposing get_default_configs.
    """
    resolved = os.path.abspath(config_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Config file not found: {resolved}")

    spec = importlib.util.spec_from_file_location("forecast_bot_configs", resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config file: {resolved}")

    config_module = importlib.util.module_from_spec(spec)
    sys.modules["forecast_bot_configs"] = config_module
    spec.loader.exec_module(config_module)

    if not hasattr(config_module, "get_default_configs"):
        raise AttributeError(
            f"Config file {resolved} must define get_default_configs() returning "
            f"(bot_config, research_configs, forecaster_configs, end_to_end_configs)"
        )

    configs = config_module.get_default_configs()
    if len(configs) != 4:
        raise ValueError(
            f"Config file {resolved} returned {len(configs)} values; expected 4 "
            f"(bot_config, research_configs, forecaster_configs, end_to_end_configs)"
        )
    return configs


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress verbose HTTP logs from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)

    args = parse_args()

    bot_config, research_configs, forecaster_configs, end_to_end_configs = load_configs(args.config)

    # CLI publish flag can override config for safety when dry-running.
    if args.publish:
        bot_config.publish_reports_to_metaculus = True

    bot = DelphiV2Bot(
        bot_config=bot_config,
        research_configs=research_configs,
        forecaster_configs=forecaster_configs,
        end_to_end_configs=end_to_end_configs,
    )

    questions: list[MetaculusQuestion] = []
    if args.mode == "test_questions":
        questions = [bot.metaculus.get_question_by_url(u) for u in TEST_QUESTION_URLS]
    elif args.mode == "urls":
        if not args.urls:
            raise ValueError("No URLs provided for --mode urls.")
        questions = [bot.metaculus.get_question_by_url(u) for u in args.urls]
    else:  # tournament
        if args.tournament_id:
            # Legacy behavior: single tournament using custom client
            logger.info(f"Fetching questions from tournament {args.tournament_id}...")
            questions = bot.metaculus.get_questions_from_tournament(args.tournament_id)
        else:
            # New behavior: dual tournaments using forecasting_tools
            from forecasting_tools import MetaculusClient as FTMetaculusClient

            # Instantiate the forecasting_tools client
            ft_client = FTMetaculusClient()

            minibench_questions = []
            ai_competition_questions = []

            try:
                logger.info("Fetching questions from Minibench tournament...")
                ft_minibench = ft_client.get_all_open_questions_from_tournament(
                    FTMetaculusClient.CURRENT_MINIBENCH_ID
                )
                minibench_questions = [convert_forecasting_tools_question(q) for q in ft_minibench]
                logger.info(f"Retrieved {len(minibench_questions)} questions from Minibench")
            except Exception as e:
                logger.error(f"Failed to fetch Minibench questions: {e}")

            try:
                logger.info("Fetching questions from AI Competition tournament...")
                ft_ai_comp = ft_client.get_all_open_questions_from_tournament(
                    FTMetaculusClient.CURRENT_AI_COMPETITION_ID
                )
                ai_competition_questions = [convert_forecasting_tools_question(q) for q in ft_ai_comp]
                logger.info(f"Retrieved {len(ai_competition_questions)} questions from AI Competition")
            except Exception as e:
                logger.error(f"Failed to fetch AI Competition questions: {e}")

            if not minibench_questions and not ai_competition_questions:
                logger.info("No open questions found in either tournament.")

            # Combine and deduplicate questions by ID
            all_questions = minibench_questions + ai_competition_questions
            seen_ids = set()
            questions = []
            for q in all_questions:
                if q.id not in seen_ids:
                    questions.append(q)
                    seen_ids.add(q.id)
                else:
                    logger.debug(f"Skipping duplicate question ID {q.id}")

            logger.info(f"Total unique questions from both tournaments: {len(questions)}")

    logger.info(f"Loaded {len(questions)} question(s) to forecast.")

    reports = []
    for idx, q in enumerate(questions):
        try:
            logger.info(f"{'#'*80}")
            logger.info(f"# Processing Question {idx + 1}/{len(questions)}")
            logger.info(f"# Remaining: {len(questions) - idx - 1} question(s)")
            logger.info(f"{'#'*80}")
            report = await bot.forecast_question(q, dry_run=not bot.config.publish_reports_to_metaculus)
            reports.append(report)
            logger.info(f"Completed question {idx + 1}/{len(questions)}")
        except Exception as e:
            logger.error(f"Failed to forecast question {q.id}: {e}")

    logger.info(f"{'#'*80}")
    logger.info(f"# ALL QUESTIONS COMPLETED")
    logger.info(f"# Successfully completed {len(reports)}/{len(questions)} reports")
    logger.info(f"{'#'*80}")


if __name__ == "__main__":
    asyncio.run(main())
