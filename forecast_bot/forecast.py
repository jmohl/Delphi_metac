"""
Forecast generation pipeline for forecast_bot_v2.

Implements binary, multiple choice, numeric, and end-to-end forecasting flows
using the v2 data models and LLM wrappers.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Callable, Awaitable

from forecast_bot_v2.configs import ForecasterConfig, EndToEndForecasterConfig
from forecast_bot_v2.llm_wrappers import GeneralLlm
from forecast_bot_v2.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecast_bot_v2.notepad import Notepad
from forecast_bot_v2.prompts import get_prompt
from forecast_bot_v2.utils import clean_indents

logger = logging.getLogger(__name__)


class ForecastModule:
    """
    Handles prediction generation for binary, multiple choice, numeric, and end-to-end questions.
    """

    def __init__(self, parent_bot=None) -> None:
        self.parent_bot = parent_bot

    # ------------------------------------------------------------------ #
    # Public forecast entrypoints
    # ------------------------------------------------------------------ #
    async def run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        return await self._run_standard_forecast(
            question=question,
            research=research,
            forecast_label="binary",
            prompt_builder=lambda cfg: self._build_binary_prompt(question, research, cfg.binary_prompt_key),
            prediction_parser=self._parse_binary_prediction,
        )

    async def run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        return await self._run_standard_forecast(
            question=question,
            research=research,
            forecast_label="multiple choice",
            prompt_builder=lambda cfg: self._build_multiple_choice_prompt(question, research, cfg.multiple_choice_prompt_key),
            prediction_parser=lambda reasoning: self._parse_multiple_choice_prediction_async(reasoning, question.options or []),
        )

    async def run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        return await self._run_standard_forecast(
            question=question,
            research=research,
            forecast_label="numeric",
            prompt_builder=lambda cfg: self._build_numeric_prompt(question, research, cfg.numeric_prompt_key),
            prediction_parser=self._parse_numeric_prediction_async,
        )

    async def run_end_to_end_forecast_on_binary(
        self, question: BinaryQuestion, config: EndToEndForecasterConfig
    ) -> ReasonedPrediction[float]:
        prompt = self._build_end_to_end_binary_prompt(question, config.binary_prompt_key)
        return await self._invoke_end_to_end(question, config, prompt, self._parse_binary_prediction, "binary")

    async def run_end_to_end_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, config: EndToEndForecasterConfig
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = self._build_end_to_end_multiple_choice_prompt(question, config.multiple_choice_prompt_key)
        return await self._invoke_end_to_end(
            question, config, prompt, lambda reasoning: self._parse_multiple_choice_prediction_async(reasoning, question.options or []), "multiple choice"
        )

    async def run_end_to_end_forecast_on_numeric(
        self, question: NumericQuestion, config: EndToEndForecasterConfig
    ) -> ReasonedPrediction[NumericDistribution]:
        prompt = self._build_end_to_end_numeric_prompt(question, config.numeric_prompt_key)
        return await self._invoke_end_to_end(question, config, prompt, self._parse_numeric_prediction_async, "numeric")

    # ------------------------------------------------------------------ #
    # Core forecast runner
    # ------------------------------------------------------------------ #
    async def _run_standard_forecast(
        self,
        question: MetaculusQuestion,
        research: str,
        forecast_label: str,
        prompt_builder: Callable[[ForecasterConfig], str],
        prediction_parser: Callable[[str], Awaitable[tuple[Any, Any]]] | Callable[[str], tuple[Any, Any]] | Callable[[str], Any],
    ) -> ReasonedPrediction:
        """
        Shared flow for binary, multiple choice, and numeric forecasts.
        """
        notepad = await self._get_notepad(question)
        forecaster_config: ForecasterConfig = notepad.get_next_forecaster_config()

        logger.info(
            f"Generating {forecast_label} forecast for URL {question.page_url} "
            f"using config: {forecaster_config.name} "
            f"(model: {forecaster_config.model})"
        )

        prompt = clean_indents(prompt_builder(forecaster_config))

        model = GeneralLlm(model=forecaster_config.model)

        reasoning = await model.invoke(prompt)

        if asyncio.iscoroutinefunction(prediction_parser):  # type: ignore[arg-type]
            parsed = await prediction_parser(reasoning)  # type: ignore
        else:
            parsed = prediction_parser(reasoning)  # type: ignore
            # Handle case where parser is a lambda wrapping an async function
            if asyncio.iscoroutine(parsed):
                parsed = await parsed

        if isinstance(parsed, tuple) and len(parsed) == 2:
            prediction_value, log_value = parsed
        else:
            prediction_value, log_value = parsed, parsed

        reasoning_with_label = f"[Forecaster: {forecaster_config.name}]\n\n{reasoning}"

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {log_value}"
        )
        return ReasonedPrediction(prediction_value=prediction_value, reasoning=reasoning_with_label)

    # ------------------------------------------------------------------ #
    # Prompt builders
    # ------------------------------------------------------------------ #
    def _build_binary_prompt(self, question: BinaryQuestion, research: str, prompt_key: str) -> str:
        prompt_template = get_prompt("binary", prompt_key)
        return prompt_template.format(
            question_text=question.question_text,
            background_info=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            research=research,
            current_date=datetime.now().strftime("%Y-%m-%d"),
        )

    def _build_multiple_choice_prompt(self, question: MultipleChoiceQuestion, research: str, prompt_key: str) -> str:
        prompt_template = get_prompt("multiple_choice", prompt_key)
        options_text = ", ".join(question.options) if question.options else "N/A"
        return prompt_template.format(
            question_text=question.question_text,
            options=options_text,
            background_info=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            research=research,
            current_date=datetime.now().strftime("%Y-%m-%d"),
        )

    def _build_numeric_prompt(self, question: NumericQuestion, research: str, prompt_key: str) -> str:
        prompt_template = get_prompt("numeric", prompt_key)

        # Construct bound messages based on whether bounds are open or closed
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = f"The outcome can not be lower than {question.lower_bound}."

        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = f"The outcome can not be higher than {question.upper_bound}."

        return prompt_template.format(
            question_text=question.question_text,
            background_info=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            research=research,
            units=question.unit or "",
            current_date=datetime.now().strftime("%Y-%m-%d"),
            lower_bound_message=lower_bound_message,
            upper_bound_message=upper_bound_message,
        )

    def _build_end_to_end_binary_prompt(self, question: BinaryQuestion, prompt_key: str) -> str:
        prompt_template = get_prompt("end_to_end_binary", prompt_key)
        return prompt_template.format(
            question_text=question.question_text,
            background_info=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            current_date=datetime.now().strftime("%Y-%m-%d"),
        )

    def _build_end_to_end_multiple_choice_prompt(self, question: MultipleChoiceQuestion, prompt_key: str) -> str:
        prompt_template = get_prompt("end_to_end_multiple_choice", prompt_key)
        options_text = ", ".join(question.options) if question.options else "N/A"
        return prompt_template.format(
            question_text=question.question_text,
            options=options_text,
            background_info=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            current_date=datetime.now().strftime("%Y-%m-%d"),
        )

    def _build_end_to_end_numeric_prompt(self, question: NumericQuestion, prompt_key: str) -> str:
        prompt_template = get_prompt("end_to_end_numeric", prompt_key)

        # Construct bound messages based on whether bounds are open or closed
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = f"The outcome can not be lower than {question.lower_bound}."

        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = f"The outcome can not be higher than {question.upper_bound}."

        return prompt_template.format(
            question_text=question.question_text,
            background_info=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            units=question.unit or "",
            current_date=datetime.now().strftime("%Y-%m-%d"),
            lower_bound_message=lower_bound_message,
            upper_bound_message=upper_bound_message,
        )

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #
    def _parse_binary_prediction(self, reasoning: str) -> tuple[float, float]:
        # Pattern 1: "Probability: XX%"
        match = re.search(r"Probability:\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%", reasoning, flags=re.IGNORECASE)
        if match:
            prob = float(match.group(1)) / 100.0
            prob = max(0.01, min(0.99, prob))
            return prob, prob

        # Pattern 2: Any decimal between 0 and 1
        match = re.search(r"\b0\.[0-9]+\b", reasoning)
        if match:
            prob = float(match.group(0))
            prob = max(0.01, min(0.99, prob))
            return prob, prob

        # Pattern 3: Percentage without "Probability:" label (e.g., "75%")
        match = re.search(r"\b([0-9]{1,3}(?:\.[0-9]+)?)\s*%", reasoning)
        if match:
            prob = float(match.group(1)) / 100.0
            prob = max(0.01, min(0.99, prob))
            return prob, prob

        # Final fallback: raise error instead of guessing
        logger.error("All parsing methods failed for binary prediction")
        raise ValueError("Could not parse binary prediction from reasoning. No valid probability found.")

    async def _parse_multiple_choice_prediction_async(self, reasoning: str, options: list[str]) -> tuple[PredictedOptionList, PredictedOptionList]:
        # Try LLM-based extraction first (most robust)
        try:
            probabilities = await self._llm_parse_multiple_choice(reasoning, options)
            logger.info("LLM-based extraction succeeded")

            # Normalize
            total = sum(probabilities.values()) or 1.0
            normalized = {opt: val / total for opt, val in probabilities.items()}
            return normalized, normalized
        except Exception as e:
            logger.warning(f"LLM-based extraction failed: {e}, falling back to regex parsing")

        # Fallback to regex parsing
        probabilities: PredictedOptionList = {}
        for i, opt in enumerate(options):
            # Pattern 1: Exact option name (e.g., "0 or 1: 0.25")
            pattern1 = rf"{re.escape(opt)}\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?"
            match = re.search(pattern1, reasoning, flags=re.IGNORECASE)
            if match:
                probabilities[opt] = float(match.group(1)) / 100.0 if float(match.group(1)) > 1 else float(match.group(1))
                continue

            # Pattern 2: Option with underscore (e.g., "Option_A: 0.25")
            pattern2 = rf"Option[_\s]?{chr(65 + i)}\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?"
            match = re.search(pattern2, reasoning, flags=re.IGNORECASE)
            if match:
                probabilities[opt] = float(match.group(1)) / 100.0 if float(match.group(1)) > 1 else float(match.group(1))
                continue

            # Pattern 3: Option with number (e.g., "Option 1: 0.25")
            pattern3 = rf"Option[_\s]?{i + 1}\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?"
            match = re.search(pattern3, reasoning, flags=re.IGNORECASE)
            if match:
                probabilities[opt] = float(match.group(1)) / 100.0 if float(match.group(1)) > 1 else float(match.group(1))
                continue

        # Final fallback: raise error instead of uniform distribution
        if not probabilities and options:
            logger.error("All parsing methods failed for multiple choice prediction")
            raise ValueError("Could not parse multiple choice prediction from reasoning. No valid probabilities found.")

        # Normalize
        total = sum(probabilities.values()) or 1.0
        normalized = {opt: val / total for opt, val in probabilities.items()}
        return normalized, normalized

    async def _llm_parse_multiple_choice(self, reasoning: str, options: list[str]) -> PredictedOptionList:
        """
        Use an LLM to extract multiple choice predictions when regex parsing fails.
        """
        from forecast_bot_v2.llm_wrappers import GeneralLlm
        from forecast_bot_v2.prompts import MULTIPLE_CHOICE_PARSING_INSTRUCTIONS

        parsing_prompt = f"""Extract the probability predictions from the following forecaster reasoning.

{MULTIPLE_CHOICE_PARSING_INSTRUCTIONS.format(options=', '.join(options))}

The forecaster's output may use formats like:
- "Option_A: 0.95" or "Option_B: 0.04"
- "Option 1: 0.95" or "Option 2: 0.04"
- "{options[0]}: 0.95" or "{options[1]}: 0.04"

Return ONLY a JSON object mapping each option name to its probability as a decimal between 0 and 1.
Example format: {{"option1": 0.25, "option2": 0.50, "option3": 0.25}}

Forecaster reasoning:
{reasoning}

JSON output:"""

        parser_model = GeneralLlm(model="gpt-4o-mini")  # Use a cheap, fast model for parsing
        response = await parser_model.invoke(parsing_prompt)

        # Extract JSON from response
        from forecast_bot_v2.llm_wrappers import _extract_json
        json_obj = _extract_json(response)

        if not json_obj or not isinstance(json_obj, dict):
            raise ValueError("LLM parser did not return valid JSON object")

        # Map the extracted probabilities to the correct option names
        result: PredictedOptionList = {}
        for opt in options:
            # Try to find this option in the parsed JSON (case-insensitive)
            for key, val in json_obj.items():
                if key.lower() == opt.lower():
                    result[opt] = float(val)
                    break

        if len(result) != len(options):
            raise ValueError(f"LLM parser did not extract all options: got {len(result)}, expected {len(options)}")

        return result

    async def _parse_numeric_prediction_async(self, reasoning: str) -> tuple[NumericDistribution, dict[str, float]]:
        # Try LLM-based extraction first (most robust)
        try:
            percentiles_map = await self._llm_parse_numeric(reasoning)
            logger.info("LLM-based extraction succeeded for numeric prediction")
            points = [(v, p / 100.0) for p, v in sorted(percentiles_map.items())]
            distribution = NumericDistribution.from_percentiles(points)
            return distribution, percentiles_map
        except Exception as e:
            logger.warning(f"LLM-based extraction failed for numeric: {e}, falling back to regex parsing")

        # Fallback to enhanced regex parsing
        percentiles_map = {}
        for p in (10, 20, 40, 60, 80, 90):
            # Pattern 1: Standard format "Percentile 10: XX"
            pattern1 = rf"Percentile\s*{p}\s*:\s*([\-0-9,\.]+)"
            match = re.search(pattern1, reasoning, flags=re.IGNORECASE)
            if match:
                value_str = match.group(1).replace(",", "")  # Remove commas
                percentiles_map[p] = float(value_str)
                continue

            # Pattern 2: Alternative formats "P10: XX" or "p10: XX"
            pattern2 = rf"P\s*{p}\s*:\s*([\-0-9,\.]+)"
            match = re.search(pattern2, reasoning, flags=re.IGNORECASE)
            if match:
                value_str = match.group(1).replace(",", "")
                percentiles_map[p] = float(value_str)
                continue

            # Pattern 3: Ordinal format "10th percentile: XX"
            ordinal = {10: "10th", 20: "20th", 40: "40th", 60: "60th", 80: "80th", 90: "90th"}
            pattern3 = rf"{ordinal[p]}\s+percentile\s*:\s*([\-0-9,\.]+)"
            match = re.search(pattern3, reasoning, flags=re.IGNORECASE)
            if match:
                value_str = match.group(1).replace(",", "")
                percentiles_map[p] = float(value_str)
                continue

        # If we got at least some percentiles, use them
        if percentiles_map:
            # If we're missing some percentiles, try to interpolate
            if len(percentiles_map) < 6:
                logger.warning(f"Only found {len(percentiles_map)}/6 percentiles, using available data")

            points = [(v, p / 100.0) for p, v in sorted(percentiles_map.items())]
            distribution = NumericDistribution.from_percentiles(points)
            return distribution, percentiles_map

        # Final fallback: raise error rather than creating a degenerate distribution
        logger.error("All parsing methods failed for numeric prediction")
        raise ValueError("Could not parse numeric prediction from reasoning. No valid percentiles found.")

    async def _llm_parse_numeric(self, reasoning: str) -> dict[int, float]:
        """
        Use an LLM to extract numeric percentile predictions when regex parsing fails.
        """
        from forecast_bot_v2.llm_wrappers import GeneralLlm

        parsing_prompt = f"""Extract the numeric percentile predictions from the following forecaster reasoning.

The forecaster should have provided 6 percentile values (10th, 20th, 40th, 60th, 80th, 90th).
They may be formatted as:
- "Percentile 10: 123.45"
- "P10: 123.45"
- "10th percentile: 123.45"

Return ONLY a JSON object mapping each percentile number (as an integer) to its value (as a number).
Example format: {{"10": 100.5, "20": 150.2, "40": 200.0, "60": 250.8, "80": 300.5, "90": 350.9}}

Important:
- The keys should be the percentile numbers: 10, 20, 40, 60, 80, 90
- The values should be the predicted values at those percentiles
- Values can be negative, decimals, or integers
- Do not include commas in numbers

Forecaster reasoning:
{reasoning}

JSON output:"""

        parser_model = GeneralLlm(model="gpt-4o-mini")  # Use a cheap, fast model for parsing
        response = await parser_model.invoke(parsing_prompt)

        # Extract JSON from response
        from forecast_bot_v2.llm_wrappers import _extract_json
        json_obj = _extract_json(response)

        if not json_obj or not isinstance(json_obj, dict):
            raise ValueError("LLM parser did not return valid JSON object")

        # Convert string keys to integers and validate
        percentiles_map: dict[int, float] = {}
        expected_percentiles = {10, 20, 40, 60, 80, 90}

        for key, val in json_obj.items():
            try:
                percentile = int(key)
                if percentile in expected_percentiles:
                    percentiles_map[percentile] = float(val)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid percentile key or value: {key}={val}") from e

        if len(percentiles_map) < 3:
            raise ValueError(f"LLM parser found too few percentiles: got {len(percentiles_map)}, expected 6")

        return percentiles_map

    # ------------------------------------------------------------------ #
    # End-to-end invocation
    # ------------------------------------------------------------------ #
    async def _invoke_end_to_end(
        self,
        question: MetaculusQuestion,
        config: EndToEndForecasterConfig,
        prompt: str,
        prediction_parser: Callable[[str], Any],
        label: str,
    ) -> ReasonedPrediction:
        llm = GeneralLlm(model=config.model)
        reasoning = await llm.invoke(
            prompt,
            tools=[{"type": "web_search"}],
            reasoning_effort=config.reasoning_effort,
        )
        parsed = prediction_parser(reasoning)
        # Handle case where parser is a lambda wrapping an async function
        if asyncio.iscoroutine(parsed):
            parsed = await parsed

        if isinstance(parsed, tuple) and len(parsed) == 2:
            prediction_value, log_value = parsed
        else:
            prediction_value, log_value = parsed, parsed

        reasoning_with_label = f"[End-to-End Forecaster: {config.name}]\n\n{reasoning}"
        logger.info(f"End-to-end forecasted URL {question.page_url} ({label}) with prediction: {log_value}")
        return ReasonedPrediction(prediction_value=prediction_value, reasoning=reasoning_with_label)

    # ------------------------------------------------------------------ #
    # Notepad lookup
    # ------------------------------------------------------------------ #
    async def _get_notepad(self, question: MetaculusQuestion) -> Notepad:
        """
        Retrieve the Notepad for this question from the parent bot.
        """
        if self.parent_bot and hasattr(self.parent_bot, "_get_notepad"):
            return await self.parent_bot._get_notepad(question)  # type: ignore[attr-defined]
        if self.parent_bot and hasattr(self.parent_bot, "get_notepad"):
            maybe = self.parent_bot.get_notepad(question)
            if asyncio.iscoroutine(maybe):
                return await maybe
            return maybe  # type: ignore[return-value]
        raise ValueError("Parent bot does not provide a notepad retrieval method.")
