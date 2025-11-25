# Delphi forecast bot
This is an updated version of the metac-bot-delphi project which has been fully refactored to split up the various components and no longer relies on forecasting_tools.

There are two broad forecasting architectures in this bot, which can be run in isolation or in parallel with aggregation across forecasts.
All model configurations are specified in config files which can be passed in to the function. The 'default_config.py' file contains the currently active setup.

### Architecture 1: Functionality split across multiple llm calls
This was the original architecture and most closely mirrors the original metac template bots. The forecasting process is split into research -> forecast-> aggregate steps
The goal here was to achieve better forecasting accuracy through aggregation of multiple research runs and forecaster bots.
Research: research can be configured to utilize any number of independent bot calls with or without distinct prompts, which are then aggregated by a separate LLM call. These are specified in the 'reasearcher' configs.
Forecast: Any number of forecast bots (with or without distinct prompts) are fed the research results from step 1 in addition to the question and generate independent reasoning and forecasts. These are specified in 'forecaster' configs.

### Architecture 2: End-to-end forecasting with agentic web search
This architecture uses a single end-to-end approach to leverage more agentic reasoning models with web search enabled. The program flow is similar to the above, except that the research and forecast pieces are combined into a single step.
Multiple independent models, with or without distinct prompts, can be run with end-to-end forecasting, though this is designed to work with agentic models with web search enabled.
In experiments, I found that a single end to end search was more accurate than any configuration of architecture 1. However, this can be fairly expensive with a single GPT-5.1 run with medium thinking costs ~$0.30 per question.

### Common to both architectures:
Aggregate: Predictions from all forecast bots are aggregated using mean across predictions. An independent llm call using 'summarizer' llm is used to provide a unified explanation that considers all independent forecasts.


This rest of this README covers the runtime flow, configuration surface, and how to execute the script.

## Flow
- Boot: `.env` variables are loaded (API keys, Metaculus token). CLI args are parsed; configs are loaded from a Python module exposing `get_default_configs()`.
- Question intake (`forecast_bot/main.py`):
  - `--mode test_questions` (default) loads `TEST_QUESTION_URLS`.
  - `--mode urls --urls <url...>` pulls specific Metaculus question URLs.
  - `--mode tournament --tournament-id <id_or_slug>` fetches open tournament questions via `MetaculusClient`.
- Per‑question orchestration (`DelphiV2Bot.forecast_question`):
  - `Notepad` builds queues of research bots, forecasters, and end‑to‑end forecasters based on counts in `BotConfig`.
  - Research path: each `ResearchBotConfig` runs `_generate_news` (single or multi web search) then fills a research prompt; results feed the forecasters.
  - Forecast path: for each research report, `ForecasterConfig` instances render prompts per type (binary/multiple_choice/numeric) and parse the model outputs into structured predictions.
  - End‑to‑end path (optional): `EndToEndForecasterConfig` models call web‑enabled prompts directly, bypassing separate research.
  - Aggregation (`AggregationModule`): combines predictions (mean for binary, averaged CDF for numeric, normalized sum for multiple choice) and synthesizes a unified explanation with the summarizer model.
  - Reporting (`Reporter`): saves a JSON report to `BotConfig.folder_to_save_reports_to`; if publishing is enabled, posts the forecast and a private comment to Metaculus.

## Configuration Surface
Configs live in `forecast_bot/default_configs.py`. Copy that file, adjust values, and point `--config` at your copy. `get_default_configs()` must return `(bot_config, research_configs, forecaster_configs, end_to_end_configs)`.

- `BotConfig`
  - `research_reports_per_question`, `predictions_per_research_report`: set to 0 to skip traditional research/forecasting.
  - `enable_end_to_end_forecaster`, `end_to_end_forecasters_per_question`: control web‑enabled one‑shot forecasts.
  - `publish_reports_to_metaculus`: if `True`, forecasts/comments are posted (requires `--publish` or config).
  - `folder_to_save_reports_to`: where JSON reports are written.
  - `llms`: model map for roles (`default`, `summarizer`, `researcher`, `parser`).
- `ResearchBotConfig`
  - `web_search_mode`: `lite`, `multi`, `heavy`, or `none`; `multi_search_models` lists models when `multi` is used.
  - `research_prompt_key`: selects the research prompt template in `prompts.py`.
- `ForecasterConfig`
  - `model`, `temperature`, and prompt keys per question type (`binary_prompt_key`, etc.).
- `EndToEndForecasterConfig`
  - Similar to `ForecasterConfig` but used for single‑call web‑enabled forecasts; supports `reasoning_effort` for models that accept it.

`prepare_configs` cycles or truncates config lists so requested counts are met.

## Running the Script
Always use Poetry’s environment from the repo root:

```bash
poetry run python -m forecast_bot.main --mode test_questions
```

Common variations:
- Specific URLs: `poetry run python -m forecast_bot.main --mode urls --urls https://www.metaculus.com/questions/123/ ...`
- Tournament: `poetry run python -m forecast_bot.main --mode tournament --tournament-id <id_or_slug>`
- Custom config: `poetry run python -m forecast_bot.main --config path/to/my_configs.py`
- Publish forecasts/comments: add `--publish` (requires valid `METACULUS_TOKEN`).

If you run `python forecast_bot/main.py` directly, set `PYTHONPATH=.` so imports resolve.

## Environment & Keys
- Metaculus: `METACULUS_TOKEN` (required to post forecasts/comments).
- Models:
  - OpenAI: `OPENAI_API_KEY`
  - OpenRouter: `OPENROUTER_API_KEY` (+ optional `OPENROUTER_BASE_URL`, `OPENROUTER_REFERRER`, `OPENROUTER_TITLE`) — required for the default Gemini models
  - Anthropic: `ANTHROPIC_API_KEY` (if using Claude models)
- Place them in `.env`; `load_dotenv()` in `main.py` reads them automatically.

## Outputs
- Reports: JSON files in `forecast_bot/reports` by default (e.g., `report_<question_id>_<timestamp>.json`) containing the question snapshot, aggregated prediction, unified explanation, research texts, and forecaster reasonings.
- Metaculus posting: when enabled, `Reporter` sends forecasts via `MetaculusClient` and posts a private comment with the synthesized explanation.
