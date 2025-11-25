"""
Metaculus API client for forecast_bot.

Provides lightweight wrappers around the Metaculus REST API without relying on
forecasting_tools. Supports fetching questions/posts and posting forecasts/comments.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable

import requests

from forecast_bot.questions import (
    MetaculusQuestion,
    create_forecast_payload,
)


API_BASE_URL = "https://www.metaculus.com/api"
DEFAULT_TIMEOUT = 30


class MetaculusClient:
    """
    Lightweight Metaculus API client.
    """

    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.getenv("METACULUS_TOKEN")
        self._session = requests.Session()

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _auth_headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Token {self.token}"}

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
    ) -> Any:
        url = f"{API_BASE_URL}{path}"
        response = self._session.request(
            method=method,
            url=url,
            params=params,
            json=json_body,
            headers=self._auth_headers(),
            timeout=DEFAULT_TIMEOUT,
        )
        if not response.ok:
            raise RuntimeError(f"Metaculus API error {response.status_code}: {response.text}")
        if not response.content:
            return None
        return response.json()

    # ------------------------------------------------------------------ #
    # Fetch operations
    # ------------------------------------------------------------------ #
    def list_posts_from_tournament(
        self,
        tournament_id: int | str,
        offset: int = 0,
        count: int = 50,
        statuses: str = "open",
        forecast_types: Iterable[str] = ("binary", "multiple_choice", "numeric", "discrete"),
    ) -> dict:
        """
        List posts for a given tournament.
        """
        params = {
            "limit": count,
            "offset": offset,
            "order_by": "-hotness",
            "forecast_type": ",".join(forecast_types),
            "tournaments": [tournament_id],
            "statuses": statuses,
            "include_description": "true",
        }
        return self._request("GET", "/posts/", params=params)

    def get_post(self, post_id: int) -> dict:
        """
        Fetch a single post by ID.
        """
        return self._request("GET", f"/posts/{post_id}/", params={"include_description": "true"})

    def get_question(self, question_id: int) -> MetaculusQuestion:
        """
        Fetch a question by ID.
        """
        data = self._request("GET", f"/questions/{question_id}/")
        return MetaculusQuestion.from_api(data)

    def get_question_by_url(self, url: str) -> MetaculusQuestion:
        """
        Fetch a question using its Metaculus URL. Supports both question and post URLs.
        """
        # Try numeric extraction first (handles standard and community URLs reliably)
        numeric_id = _extract_id_from_url(url)
        if numeric_id is not None:
            # Try treating the id as a post first to cover community/discrete posts
            try:
                post = self.get_post(numeric_id)
                if question_data := post.get("question"):
                    q = MetaculusQuestion.from_api(question_data, post_id=numeric_id)
                    if not q.page_url:
                        q.page_url = url
                    return q
            except Exception:
                # Fall through to attempt question fetch
                pass

            # Fallback to fetching as a question id
            q = self.get_question(numeric_id)
            if not q.page_url:
                q.page_url = url
            return q

        # As a last resort, resolve via posts lookup by URL
        resolved = self._resolve_question_by_url(url)
        if resolved:
            if not resolved.page_url:
                resolved.page_url = url
            return resolved

        raise ValueError(f"Could not extract or resolve id from URL: {url}")

    def _resolve_question_by_url(self, url: str) -> MetaculusQuestion | None:
        """
        Attempt to resolve a question using the full URL via the posts endpoint.
        """
        for param_name in ("page_url", "url"):
            try:
                data = self._request(
                    "GET",
                    "/posts/",
                    params={param_name: url, "include_description": "true"},
                )
                results = data.get("results") if isinstance(data, dict) else None
                if not results:
                    continue

                # Ensure the returned post actually matches the requested URL to avoid random results
                normalized_target = url.rstrip("/")
                for post in results:
                    post_url = str(post.get("page_url") or post.get("url") or "").rstrip("/")
                    if post_url and post_url == normalized_target:
                        question_data = post.get("question")
                        if question_data:
                            q = MetaculusQuestion.from_api(question_data, post_id=post.get("id"))
                            if not q.page_url:
                                q.page_url = post_url
                            return q
            except Exception:
                continue
        return None

    def get_questions_from_tournament(
        self,
        tournament_id: int | str,
        max_questions: int | None = None,
        statuses: str = "open",
    ) -> list[MetaculusQuestion]:
        """
        Fetch questions for a tournament, respecting optional max_questions.
        """
        collected: list[MetaculusQuestion] = []
        offset = 0
        page_size = 50
        while True:
            posts = self.list_posts_from_tournament(
                tournament_id=tournament_id,
                offset=offset,
                count=page_size,
                statuses=statuses,
            )

            for post in posts.get("results", []):
                if question := post.get("question"):
                    question_obj = MetaculusQuestion.from_api(question, post_id=post.get("id"))
                    collected.append(question_obj)
                    if max_questions and len(collected) >= max_questions:
                        return collected

            if not posts.get("next"):
                break

            offset += page_size
        return collected

    # ------------------------------------------------------------------ #
    # Post operations
    # ------------------------------------------------------------------ #
    def post_comment(self, post_id: int, text: str, include_forecast: bool = True, is_private: bool = True) -> None:
        """
        Post a comment to a Metaculus post.
        """
        payload = {
            "text": text,
            "parent": None,
            "included_forecast": include_forecast,
            "is_private": is_private,
            "on_post": post_id,
        }
        self._request("POST", "/comments/create/", json_body=payload)

    def post_forecast(self, question_id: int, forecast: Any, question_type: str, question: Any = None) -> None:
        """
        Post a forecast to a Metaculus question.

        Args:
            question_id: ID of the question
            forecast: The prediction value
            question_type: Type of question (binary, numeric, multiple_choice, discrete)
            question: Optional question object (required for numeric questions)
        """
        payload = create_forecast_payload(forecast, question_type, question=question)
        body = [{"question": question_id, **payload}]
        self._request("POST", "/questions/forecast/", json_body=body)


def _extract_id_from_url(url: str) -> int | None:
    """
    Extract the trailing numeric id from a Metaculus URL.
    """
    matches = re.findall(r"/(\d+)/", url)
    if not matches:
        return None
    # Prefer the largest id in the path (captures community slug patterns)
    return int(max(matches, key=int))
