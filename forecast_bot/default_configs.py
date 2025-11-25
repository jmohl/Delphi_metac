"""
Default configurations for research and forecaster bots (v2).

This file mirrors the v1 defaults but targets the forecast_bot classes and wrappers.
You can copy this file and modify it to create custom configurations, then point the
v2 main script at the custom file if you want to override these defaults.
"""

from forecast_bot.configs import BotConfig, EndToEndForecasterConfig, ForecasterConfig, ResearchBotConfig


def get_default_configs() -> tuple[BotConfig, list[ResearchBotConfig], list[ForecasterConfig], list[EndToEndForecasterConfig]]:
    """
    Returns the default bot, research, forecaster, and end-to-end forecaster configurations.

    Returns:
        tuple: (bot_config, research_configs, forecaster_configs, end_to_end_configs)
    """
    # Bot initialization configuration. ------------------------
    # These settings control the overall bot behavior, report generation,
    # and which LLMs are used for different tasks.
    # Set both research_reports_per_question and predictions_per_research_report to 0
    # to run in end-to-end-only mode (requires end-to-end forecasters configured below).
    bot_config = BotConfig(
        research_reports_per_question=0,
        predictions_per_research_report=0,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to="forecast_bot/reports",
        skip_previously_forecasted_questions=False,
        enable_end_to_end_forecaster=True,
        end_to_end_forecasters_per_question=1,
        llms={
            # Default LLM (not actually used when forecaster configs are specified)
            "default": "openrouter/google/gemini-2.5-flash",
            # LLM for summarizing research reports
            "summarizer": "openrouter/google/gemini-2.5-flash",
            # LLM for aggregating research (not the individual researchers - those are in research configs)
            "researcher": "openrouter/google/gemini-2.5-flash",
            # LLM for parsing and validating predictions
            "parser": "openrouter/google/gemini-2.5-flash",
        },
    )

    # Research bot configurations. ------------------------
    # Research bots will be selected in order based on the number of research reports needed.
    # If more research bots are needed than provided here, the cycle will repeat from the start of the list.
    # This config file modifies the prompt and model used for web searches, as well as the prompt for the research aggregation bot.
    # The research aggregator model is selected in main.py.
    # Note that there are 3 options for web_search_mode:
    # - "lite": single model web search with faster but less comprehensive results using hardcoded model gemini-2.5-flash with openrouter search
    # - "multi": multiple parallel web searches using specified models in multi_search_models
    # - "heavy": single model web search with more comprehensive results using hardcoded model gpt-5-mini and native web search
    research_configs = [
        # single search bots
        ResearchBotConfig(
            name="Lite Search (flash x1)",
            web_search_mode="lite", #when using lite or heavy, do not need to specify multi_search_models
            research_prompt_key="default"
        ),
        # ResearchBotConfig(
        #     name="Multi Search (5-mini x1)",
        #     web_search_mode="multi", #multi can be used with a single bot as well, as done here
        #     research_prompt_key="default", #use default for research prompt if using only 1 model
        #     multi_search_models=[
        #         "gpt-5-mini",  # OpenAI native web search
        #     ]
        # ),
        # Multi-search with parallel web searches using different models
        # ResearchBotConfig(
        #     name="Multi Search (5-mini + Haiku + Flash)",
        #     web_search_mode="multi",
        #     research_prompt_key="multi", #use multi prompt for research when using multiple searches
        #     multi_search_models=[
        #         "gpt-5-mini",  # OpenAI native web search
        #         "claude-haiku-4-5",  # Anthropic native web search
        #         "openrouter/google/gemini-2.5-flash:online",  # Gemini online mode
        #     ]
        # ),
    ]

    # Configurations for forecaster bots. -------------------
    # Forecaster bots will be selected in order based on the number of forecasts per reserach report.
    # If more forecaster bots are needed than provided here, the cycle will repeat from the start of the list.
    forecaster_configs = [
        ForecasterConfig(
            name="Gemini flash",
            model="openrouter/google/gemini-2.5-flash",
            binary_prompt_key="default",
            numeric_prompt_key="default",
            multiple_choice_prompt_key="default"
        ),
        # ForecasterConfig(
        #     name="GPT-5",
        #     model="openrouter/openai/gpt-5",
        #     binary_prompt_key="default",
        #     numeric_prompt_key="default",
        #     multiple_choice_prompt_key="default"
        # )
    ]

    # End-to-end forecaster configurations. -------------------
    # End-to-end forecasters combine research and forecasting in a single web-enabled call.
    # They will be selected in order based on the number of end-to-end forecasts per question.
    # If more end-to-end forecasters are needed than provided here, the cycle will repeat from the start of the list.
    # Note: These forecasters use OpenAI's native web search, so they must use models that support it (e.g., gpt-5-mini)
    end_to_end_configs = [
        # EndToEndForecasterConfig(
        #     name="GPT-5-mini End-to-End",
        #     model="gpt-5-mini",
        #     binary_prompt_key="default",
        #     numeric_prompt_key="default",
        #     multiple_choice_prompt_key="default"
        # ),
        EndToEndForecasterConfig(
            name="GPT-5.1 End-to-End",
            model="gpt-5.1",
            reasoning_effort="medium", #medium is good
            binary_prompt_key="default",
            numeric_prompt_key="default",
            multiple_choice_prompt_key="default"
        ),
    ]

    return bot_config, research_configs, forecaster_configs, end_to_end_configs
