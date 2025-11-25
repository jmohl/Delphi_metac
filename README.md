# Forecast Bot v2

Lightweight, dependency‑minimal Metaculus forecaster that mirrors the v1 pipeline without `forecasting_tools`. This README covers the runtime flow, configuration surface, and how to execute the script.

## End‑to‑End Flow
- Boot: `.env` variables are loaded (API keys, Metaculus token). CLI args are parsed; configs are loaded from a Python module exposing `get_default_configs()`.
- Question intake (`forecast_bot_v2/main.py`):
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
Configs live in `forecast_bot_v2/default_configs.py`. Copy that file, adjust values, and point `--config` at your copy. `get_default_configs()` must return `(bot_config, research_configs, forecaster_configs, end_to_end_configs)`.

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
poetry run python -m forecast_bot_v2.main --mode test_questions
```

Common variations:
- Specific URLs: `poetry run python -m forecast_bot_v2.main --mode urls --urls https://www.metaculus.com/questions/123/ ...`
- Tournament: `poetry run python -m forecast_bot_v2.main --mode tournament --tournament-id <id_or_slug>`
- Custom config: `poetry run python -m forecast_bot_v2.main --config path/to/my_configs.py`
- Publish forecasts/comments: add `--publish` (requires valid `METACULUS_TOKEN`).

If you run `python forecast_bot_v2/main.py` directly, set `PYTHONPATH=.` so imports resolve.

## Environment & Keys
- Metaculus: `METACULUS_TOKEN` (required to post forecasts/comments).
- Models:
  - OpenAI: `OPENAI_API_KEY`
  - OpenRouter: `OPENROUTER_API_KEY` (+ optional `OPENROUTER_BASE_URL`, `OPENROUTER_REFERRER`, `OPENROUTER_TITLE`) — required for the default Gemini models
  - Anthropic: `ANTHROPIC_API_KEY` (if using Claude models)
- Place them in `.env`; `load_dotenv()` in `main.py` reads them automatically.

## Outputs
- Reports: JSON files in `forecast_bot_v2/reports` by default (e.g., `report_<question_id>_<timestamp>.json`) containing the question snapshot, aggregated prediction, unified explanation, research texts, and forecaster reasonings.
- Metaculus posting: when enabled, `Reporter` sends forecasts via `MetaculusClient` and posts a private comment with the synthesized explanation.
