"""
Script to emulate the tournament question pulling logic from forecast_bot/main.py
and save the questions as a JSON file.

Usage:
    PYTHONPATH=. python pull_tournament_questions.py
"""

import json
import logging
from dataclasses import asdict
from typing import Any

from dotenv import load_dotenv
from forecasting_tools import MetaculusClient as FTMetaculusClient

from forecast_bot.questions import convert_forecasting_tools_question

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main() -> None:
    logger.info("Starting tournament question pull...")

    # Instantiate the forecasting_tools client
    ft_client = FTMetaculusClient()

    minibench_questions = []
    ai_competition_questions = []

    # Fetch Minibench questions
    try:
        logger.info(f"Fetching questions from Minibench tournament (ID: {FTMetaculusClient.CURRENT_MINIBENCH_ID})...")
        ft_minibench = ft_client.get_all_open_questions_from_tournament(
            FTMetaculusClient.CURRENT_MINIBENCH_ID
        )
        minibench_questions = [convert_forecasting_tools_question(q) for q in ft_minibench]
        logger.info(f"Retrieved {len(minibench_questions)} questions from Minibench")
    except Exception as e:
        logger.error(f"Failed to fetch Minibench questions: {e}")

    # Fetch AI Competition questions
    try:
        logger.info(f"Fetching questions from AI Competition tournament (ID: {FTMetaculusClient.CURRENT_AI_COMPETITION_ID})...")
        ft_ai_comp = ft_client.get_all_open_questions_from_tournament(
            FTMetaculusClient.CURRENT_AI_COMPETITION_ID
        )
        ai_competition_questions = [convert_forecasting_tools_question(q) for q in ft_ai_comp]
        logger.info(f"Retrieved {len(ai_competition_questions)} questions from AI Competition")
    except Exception as e:
        logger.error(f"Failed to fetch AI Competition questions: {e}")

    if not minibench_questions and not ai_competition_questions:
        logger.warning("No open questions found in either tournament.")

    # Combine and deduplicate questions by ID
    all_questions = minibench_questions + ai_competition_questions
    seen_ids = set()
    unique_questions = []
    
    for q in all_questions:
        if q.id not in seen_ids:
            unique_questions.append(q)
            seen_ids.add(q.id)
        else:
            logger.debug(f"Skipping duplicate question ID {q.id}")

    logger.info(f"Total unique questions from both tournaments: {len(unique_questions)}")

    # Convert to list of dicts for JSON serialization
    serialized_questions = [asdict(q) for q in unique_questions]

    # Save to JSON file
    output_file = "tournament_questions.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serialized_questions, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved {len(serialized_questions)} questions to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save questions to JSON: {e}")

if __name__ == "__main__":
    main()

