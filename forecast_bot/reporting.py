"""
Reporting and posting utilities for forecast_bot.

Handles saving reports locally and posting forecasts/comments to Metaculus.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from forecast_bot.metaculus_client import MetaculusClient
from forecast_bot.questions import MetaculusQuestion, create_forecast_payload

logger = logging.getLogger(__name__)


@dataclass
class ForecastReport:
    """
    Container for report data. Not persisted in a specific schema yet.
    """

    question: MetaculusQuestion
    prediction: Any
    explanation: str
    price_estimate: float = 0.0
    minutes_taken: float = 0.0
    errors: list[str] = field(default_factory=list)
    research: list[str] = field(default_factory=list)
    forecaster_reasonings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        from forecast_bot.questions import NumericDistribution, NumericQuestion

        payload = asdict(self)
        payload["question"] = asdict(self.question)

        # If prediction is a NumericDistribution, expand it to full 201-point CDF
        if isinstance(self.prediction, NumericDistribution) and isinstance(self.question, NumericQuestion):
            # Convert to the full 201-point CDF that Metaculus expects
            if self.question.lower_bound is not None and self.question.upper_bound is not None:
                full_cdf_values = self.prediction.to_metaculus_cdf(
                    lower_bound=self.question.lower_bound,
                    upper_bound=self.question.upper_bound,
                    open_lower_bound=self.question.open_lower_bound,
                    open_upper_bound=self.question.open_upper_bound,
                    continuous_range=self.question.continuous_range,
                )
                # Store as list of percentile objects matching Metaculus format
                payload["prediction"] = {
                    "cdf": [
                        {"value": self.question.continuous_range[i] if self.question.continuous_range else
                                 self.question.lower_bound + (self.question.upper_bound - self.question.lower_bound) * i / 200,
                         "percentile": full_cdf_values[i]}
                        for i in range(len(full_cdf_values))
                    ]
                }

        return json.dumps(payload, indent=2, default=str)


class Reporter:
    """
    Save reports to disk and optionally post forecasts/comments to Metaculus.
    """

    def __init__(self, folder: str = "forecast_bot/reports", metaculus_client: Optional[MetaculusClient] = None) -> None:
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.client = metaculus_client or MetaculusClient()

    def save_report(self, report: ForecastReport) -> str:
        """
        Save report as JSON. Returns filepath.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"report_{report.question.id}_{timestamp}.json"
        path = os.path.join(self.folder, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report.to_json())
        logger.info(f"Saved report to {path}")
        return path

    def post_forecast(self, report: ForecastReport, dry_run: bool = True) -> None:
        """
        Post forecast to Metaculus using the MetaculusClient. Respects dry_run.
        """
        payload = create_forecast_payload(
            report.prediction,
            report.question.question_type,
            question=report.question,
        )
        if dry_run:
            logger.info(f"[DRY RUN] Would post forecast for question {report.question.id}: {payload}")
            return

        self.client.post_forecast(
            report.question.id,
            report.prediction,
            report.question.question_type,
            question=report.question,
        )
        logger.info(f"Posted forecast for question {report.question.id}")

    def post_comment(self, report: ForecastReport, dry_run: bool = True) -> None:
        """
        Post a private comment with the unified explanation. Respects dry_run.
        """
        if dry_run:
            logger.info(f"[DRY RUN] Would post comment for post {report.question.post_id}")
            return
        try:
            reasoning_block = "\n\n---\nIndividual forecaster notes:\n" + "\n\n".join(report.forecaster_reasonings)
            comment_text = report.explanation + reasoning_block
            self.client.post_comment(report.question.post_id, comment_text, include_forecast=True, is_private=True)
            logger.info(f"Posted comment for post {report.question.post_id}")
        except Exception as e:
            logger.error(f"Failed to post comment: {e}")
