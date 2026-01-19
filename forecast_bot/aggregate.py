"""
Aggregation utilities for forecast_bot.

Implements mean-based aggregation and reasoning synthesis without forecasting_tools.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

import numpy as np

from forecast_bot.llm_wrappers import GeneralLlm, validate_model_credentials
from forecast_bot.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    ReasonedPrediction,
)

logger = logging.getLogger(__name__)


class AggregationModule:
    """
    Handles prediction aggregation and reasoning synthesis.
    """

    def __init__(self, parent_bot=None) -> None:
        self.parent_bot = parent_bot

    # ------------------------------------------------------------------ #
    # Prediction aggregation
    # ------------------------------------------------------------------ #
    def aggregate_predictions(self, predictions: list[Any], question: MetaculusQuestion) -> Any:
        """
        Aggregate predictions using mean logic similar to v1.
        """
        if isinstance(question, BinaryQuestion):
            logger.info(f"Aggregating {len(predictions)} binary predictions using mean")
            for prediction in predictions:
                assert 0 <= prediction <= 1, "Predictions must be between 0 and 1"
            result = float(statistics.mean(predictions))
            logger.info(f"Binary aggregation result: {result:.2%}")
            return result

        if isinstance(question, NumericQuestion):
            logger.info(f"Aggregating {len(predictions)} numeric distributions")
            result = self._aggregate_numeric(predictions)
            logger.info(f"Numeric aggregation completed")
            return result

        if isinstance(question, MultipleChoiceQuestion):
            logger.info(f"Aggregating {len(predictions)} multiple choice predictions")
            result = self._aggregate_multiple_choice(predictions)
            logger.info(f"Multiple choice aggregation completed")
            return result

        raise NotImplementedError(f"Unsupported question type for aggregation: {type(question)}")

    def _aggregate_multiple_choice(self, predictions: list[PredictedOptionList]) -> PredictedOptionList:
        if not predictions:
            raise ValueError("No multiple-choice predictions to aggregate.")
        combined: dict[str, float] = {}
        for pred in predictions:
            for opt, prob in pred.items():
                combined[opt] = combined.get(opt, 0.0) + prob
        total = sum(combined.values()) or 1.0
        return {opt: val / total for opt, val in combined.items()}

    def _aggregate_numeric(self, predictions: list[NumericDistribution]) -> NumericDistribution:
        if not predictions:
            raise ValueError("No numeric predictions to aggregate.")

        cdfs = [prediction.get_cdf() for prediction in predictions]

        # Extract percentiles from first CDF (they should all match)
        percentile_axis = [p.percentile for p in cdfs[0]]
        values_matrix: list[list[float]] = []

        for cdf in cdfs:
            current_percentiles = [p.percentile for p in cdf]
            # Check that percentiles match (not values!)
            if current_percentiles != percentile_axis:
                raise ValueError("Numeric distributions have mismatched percentile values.")
            # Collect the VALUES at each percentile
            values_matrix.append([p.value for p in cdf])

        # Average the values at each percentile
        mean_values = np.mean(np.array(values_matrix), axis=0).tolist()
        mean_cdf = [Percentile(value=val, percentile=perc)
                    for val, perc in zip(mean_values, percentile_axis)]
        return NumericDistribution(cdf=mean_cdf)

    # ------------------------------------------------------------------ #
    # Reasoning aggregation
    # ------------------------------------------------------------------ #
    async def create_unified_explanation(
        self,
        question: MetaculusQuestion,
        research_prediction_collections: list,
        aggregated_prediction: Any,
        end_to_end_predictions: list[ReasonedPrediction] | None = None,
        summarizer_model: str | None = None,
    ) -> str:
        """
        Aggregate multiple forecaster reasonings into a single unified report.

        research_prediction_collections is expected to contain objects with a `predictions`
        attribute (iterable of ReasonedPrediction) and optional `research` or `name`.
        """
        end_to_end_predictions = end_to_end_predictions or []

        logger.info(f"Collecting reasonings from {len(research_prediction_collections)} research collection(s) and {len(end_to_end_predictions)} end-to-end prediction(s)")

        all_reasonings: list[str] = []
        for idx, collection in enumerate(research_prediction_collections):
            label = getattr(collection, "name", f"Research Report {idx + 1}")
            preds = getattr(collection, "predictions", []) or []
            for pred_idx, pred in enumerate(preds):
                forecaster_label = f"{label} / Forecaster {pred_idx + 1}"
                all_reasonings.append(f"{forecaster_label}:\n{pred.reasoning}")

        for idx, pred in enumerate(end_to_end_predictions):
            all_reasonings.append(f"End-to-end forecaster {idx + 1}:\n{pred.reasoning}")

        if not all_reasonings:
            logger.warning("No forecaster reasonings available")
            return "No forecaster reasonings available."

        logger.info(f"Collected {len(all_reasonings)} total reasoning(s)")

        summary_prompt = self._build_summary_prompt(question, aggregated_prediction, all_reasonings)
        llm = self._get_llm_for_role("summarizer", fallback_model=summarizer_model)
        logger.info(f"Synthesizing with model: {llm.model}")
        try:
            summary = await llm.invoke(summary_prompt)
            logger.info("Synthesis completed successfully")
        except Exception as e:
            logger.error(f"Failed to summarize reasonings: {e}")
            summary = "\n\n".join(all_reasonings)
        return summary

    def _build_summary_prompt(self, question: MetaculusQuestion, aggregated_prediction: Any, reasonings: list[str]) -> str:
        joined_reasonings = "\n\n".join(reasonings)
        return f"""
Question: {question.question_text}
Resolution criteria: {question.resolution_criteria}
Fine print: {question.fine_print}

Aggregated prediction: {aggregated_prediction}

Below are multiple forecaster reasonings. Synthesize them into a single ~500 word report that:
- Highlights consensus and disagreements
- Notes key evidence and uncertainties
- States the aggregated prediction clearly

Reasonings:
{joined_reasonings}
"""

    def _get_llm_for_role(self, role: str, fallback_model: str | None = None) -> GeneralLlm:
        """
        Select an LLM for the given role from parent_bot.llms if available, else default.
        """
        default_model = fallback_model or "openrouter/google/gemini-2.5-flash"
        preferred = default_model
        if self.parent_bot and getattr(self.parent_bot, "llms", None):
            preferred = self.parent_bot.llms.get(role) or self.parent_bot.llms.get("default") or default_model

        # Try preferred first; if the necessary key is missing, fall back to default and finally gemini-2.5-flash.
        candidates = []
        for model in (preferred, default_model, "openrouter/google/gemini-2.5-flash"):
            if model not in candidates:
                candidates.append(model)

        chosen = candidates[0]
        for model in candidates:
            valid, reason = validate_model_credentials(model)
            if valid:
                chosen = model
                break
            logger.warning(f"Model '{model}' for role '{role}' is unavailable: {reason}. Trying fallback.")

        return GeneralLlm(model=chosen)
