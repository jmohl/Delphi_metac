"""
Research pipeline for forecast_bot_v2.

Ports the v1 research flow without forecasting_tools. Supports multiple web-search
strategies and configurable prompts.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional

from forecast_bot_v2.configs import ResearchBotConfig
from forecast_bot_v2.llm_wrappers import GeneralLlm, validate_model_credentials
from forecast_bot_v2.prompts import get_prompt
from forecast_bot_v2 import prompts
from forecast_bot_v2.questions import MetaculusQuestion
from forecast_bot_v2.utils import clean_indents

logger = logging.getLogger(__name__)


class ResearchModule:
    """
    Handles web search and research generation for forecasting questions.
    """

    def __init__(self, parent_bot=None, default_news_model: str = "openrouter/google/gemini-2.5-flash") -> None:
        self.parent_bot = parent_bot
        self.default_news_model = default_news_model

    async def run_research(self, question: MetaculusQuestion, config: ResearchBotConfig) -> str:
        """
        Generate research for a question using the configured web-search mode and prompt.
        """
        logger.info(f"  Web search mode: {config.web_search_mode}")
        news_summary = await self._generate_news(question, config)
        logger.info(f"  News generation completed")

        options = ", ".join(question.options) if getattr(question, "options", None) else "N/A"
        prompt_template = get_prompt("research", config.research_prompt_key)
        prompt_text = clean_indents(
            prompt_template.format(
                question_text=question.question_text,
                resolution_criteria=question.resolution_criteria,
                fine_print=question.fine_print,
                options=options,
                news_summary=news_summary,
            )
        )

        researcher_llm = self._get_llm_for_role("researcher")
        logger.info(f"  Generating research report with model: {researcher_llm.model}")
        reasoning = await researcher_llm.invoke(prompt_text)
        logger.info(f"  Research report generated")
        return reasoning

    # ------------------------------------------------------------------ #
    # News generation helpers
    # ------------------------------------------------------------------ #
    async def _generate_news(self, question: MetaculusQuestion, config: ResearchBotConfig) -> str:
        mode = config.web_search_mode
        if mode == "none":
            logger.info(f"    Web search disabled")
            return "Web search disabled for this run."

        if mode == "multi" and config.multi_search_models:
            logger.info(f"    Running multi-model web search with {len(config.multi_search_models)} model(s): {', '.join(config.multi_search_models)}")
            return await self._generate_multi_web_search_news(question, config.multi_search_models)

        # Default: single news search
        logger.info(f"    Running web search with model: {self.default_news_model}")
        return await self._generate_web_search_news(question, model=self.default_news_model)

    async def _generate_web_search_news(self, question: MetaculusQuestion, model: str) -> str:
        """
        Generate news summary using an LLM with web search tools.
        """
        prompt = prompts.NEWS_REPORTER_PROMPT.format(
            question_text=question.question_text,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
        )

        try:
            llm = GeneralLlm(model=model)
            response = await llm.invoke(prompt, tools=[{"type": "web_search"}])
            return response
        except Exception as e:
            logger.error(f"Web search news generation failed for model {model}: {e}")
            return f"Web search error: {e}"

    async def _generate_multi_web_search_news(self, question: MetaculusQuestion, models: list[str]) -> str:
        """
        Run multiple web searches in parallel using different models and combine results.
        """
        async def _single(model: str, idx: int) -> str:
            result = await self._generate_web_search_news(question, model)
            return f"Web search #{idx + 1} ({model}) found: {result}"

        tasks = [_single(m, i) for i, m in enumerate(models)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        combined = []
        for res in results:
            if isinstance(res, Exception):
                combined.append(f"Error: {res}")
            else:
                combined.append(res)
        return "\n\n".join(combined)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _get_llm_for_role(self, role: str) -> GeneralLlm:
        """
        Select an LLM for the given role from parent_bot.llms if available, else default.
        """
        default_model = "openrouter/google/gemini-2.5-flash"
        preferred = default_model
        if self.parent_bot and getattr(self.parent_bot, "llms", None):
            preferred = self.parent_bot.llms.get(role) or self.parent_bot.llms.get("default") or default_model

        for model in (preferred, default_model):
            valid, reason = validate_model_credentials(model)
            if valid:
                return GeneralLlm(model=model)
            logger.warning(f"Model '{model}' for role '{role}' is unavailable: {reason}. Trying fallback.")

        return GeneralLlm(model=default_model)
