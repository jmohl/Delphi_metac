from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

import numpy as np

ForecastType = Literal["binary", "numeric", "multiple_choice", "discrete"]
PredictedOptionList = dict[str, float]


def convert_forecasting_tools_question(ft_question: Any) -> "MetaculusQuestion":
    """
    Convert a forecasting_tools MetaculusQuestion to our custom MetaculusQuestion format.

    The forecasting_tools library uses different attribute names:
    - id_of_question -> id
    - id_of_post -> post_id
    """
    # Build the data dict for our MetaculusQuestion.from_api() method
    data = {
        "id": ft_question.id_of_question,
        "post_id": ft_question.id_of_post,
        "type": ft_question.question_type,
        "page_url": ft_question.page_url,
        "title": ft_question.question_text,
        "resolution_criteria": ft_question.resolution_criteria or "",
        "fine_print": ft_question.fine_print or "",
        "description": ft_question.background_info or "",
    }

    # Preserve already_forecasted field from forecasting_tools question
    if hasattr(ft_question, "already_forecasted"):
        data["my_forecasts"] = {
            "latest": {
                "forecast_values": {} if ft_question.already_forecasted else None
            }
        }

    # Add type-specific fields
    if ft_question.question_type == "numeric":
        # Get bounds, checking both the question attributes and api_json
        lower_bound = ft_question.lower_bound
        upper_bound = ft_question.upper_bound

        # If bounds are None, try to get them from api_json
        if (lower_bound is None or upper_bound is None) and hasattr(ft_question, "api_json") and ft_question.api_json:
            question_data = ft_question.api_json.get("question", {})
            scaling = question_data.get("scaling", {})
            if "range_min" in scaling:
                lower_bound = scaling["range_min"]
            if "range_max" in scaling:
                upper_bound = scaling["range_max"]

        data["possibilities"] = {
            "scale": {
                "min": lower_bound,
                "max": upper_bound,
                "open_lower_bound": ft_question.open_lower_bound,
                "open_upper_bound": ft_question.open_upper_bound,
                "unit": getattr(ft_question, "unit_of_measure", None),
            }
        }

        # Extract continuous_range from api_json if available (for logarithmic scale questions)
        if (lower_bound is not None and upper_bound is not None and
            hasattr(ft_question, "api_json") and ft_question.api_json):
            question_data = ft_question.api_json.get("question", {})
            scaling = question_data.get("scaling", {})
            if "continuous_range" in scaling:
                data["continuous_range"] = scaling["continuous_range"]
    elif ft_question.question_type in ("multiple_choice", "discrete"):
        # For multiple choice, we'd need to extract options from the API JSON
        # The forecasting_tools question might have this in api_json
        if hasattr(ft_question, "api_json") and ft_question.api_json:
            data["options"] = ft_question.api_json.get("options")
            if ft_question.question_type == "discrete":
                data["possibilities"] = ft_question.api_json.get("possibilities", {})

    return MetaculusQuestion.from_api(data, post_id=ft_question.id_of_post)


@dataclass
class MetaculusQuestion:
    """
    Minimal question representation independent of forecasting_tools.
    """

    id: int
    post_id: int
    question_type: ForecastType
    page_url: str
    question_text: str
    resolution_criteria: str = ""
    fine_print: str = ""
    background_info: str = ""
    options: Optional[list[str]] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    open_lower_bound: bool = False
    open_upper_bound: bool = False
    unit: Optional[str] = None
    already_forecasted: Optional[bool] = None

    @classmethod
    def from_api(cls, data: dict[str, Any], post_id: int | None = None) -> "MetaculusQuestion":
        """
        Build a question object from Metaculus API payloads.
        """
        question_id = data.get("id") or data.get("question", {}).get("id")
        if question_id is None:
            raise ValueError("Question payload missing 'id'.")

        question_type = _extract_question_type(data)
        post_identifier = post_id or data.get("post_id") or data.get("post") or data.get("id")

        page_url = data.get("page_url") or data.get("url") or ""
        question_text = data.get("title") or data.get("name") or data.get("question_text") or ""
        resolution_criteria = data.get("resolution_criteria") or data.get("resolutionCriteria") or ""
        fine_print = data.get("fine_print") or data.get("finePrint") or ""
        background_info = data.get("description") or data.get("details") or ""

        # Check if question has already been forecasted by the authenticated user
        already_forecasted = False
        try:
            forecast_values = data.get("my_forecasts", {}).get("latest", {}).get("forecast_values")
            already_forecasted = forecast_values is not None
        except Exception:
            already_forecasted = False

        options = _extract_options(data)
        bounds = _extract_numeric_bounds(data)

        if question_type == "binary":
            return BinaryQuestion(
                id=question_id,
                post_id=post_identifier,
                question_type=question_type,
                page_url=page_url,
                question_text=question_text,
                resolution_criteria=resolution_criteria,
                fine_print=fine_print,
                background_info=background_info,
                already_forecasted=already_forecasted,
            )

        if question_type in ("multiple_choice", "discrete"):
            lower_bound, upper_bound, open_lower_bound, open_upper_bound, unit, continuous_range = bounds
            return MultipleChoiceQuestion(
                id=question_id,
                post_id=post_identifier,
                question_type=question_type,
                page_url=page_url,
                question_text=question_text,
                resolution_criteria=resolution_criteria,
                fine_print=fine_print,
                background_info=background_info,
                options=options or [],
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                open_lower_bound=open_lower_bound,
                open_upper_bound=open_upper_bound,
                already_forecasted=already_forecasted,
            )

        lower_bound, upper_bound, open_lower_bound, open_upper_bound, unit, continuous_range = bounds
        return NumericQuestion(
            id=question_id,
            post_id=post_identifier,
            question_type="numeric",
            page_url=page_url,
            question_text=question_text,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
            background_info=background_info,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            open_lower_bound=open_lower_bound,
            open_upper_bound=open_upper_bound,
            unit=unit,
            continuous_range=continuous_range,
            already_forecasted=already_forecasted,
        )


@dataclass
class BinaryQuestion(MetaculusQuestion):
    """
    Binary forecasting question.
    """


@dataclass
class MultipleChoiceQuestion(MetaculusQuestion):
    """
    Multiple choice or discrete forecasting question.
    """

    options: list[str] | None = None


@dataclass
class NumericQuestion(MetaculusQuestion):
    """
    Numeric forecasting question with optional bounds.
    """

    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    open_lower_bound: bool = False
    open_upper_bound: bool = False
    unit: Optional[str] = None
    continuous_range: Optional[list[float]] = None  # For logarithmic scale questions


@dataclass
class Percentile:
    """
    Value-percentile pair for numeric distributions.
    """

    value: float
    percentile: float  # expressed as 0-1


@dataclass
class NumericDistribution:
    """
    Lightweight numeric distribution representation.
    """

    cdf: list[Percentile]

    @classmethod
    def from_percentiles(cls, points: Iterable[tuple[float, float]]) -> "NumericDistribution":
        """
        Build a NumericDistribution from (value, percentile) pairs.
        Percentiles are expected on a 0-1 scale.
        """
        percentiles = [Percentile(value=val, percentile=p) for val, p in points]
        sorted_cdf = sorted(percentiles, key=lambda item: item.percentile)
        return cls(cdf=sorted_cdf)

    def get_cdf(self) -> list[Percentile]:
        """
        Return the CDF as a percentile-sorted list.
        """
        return sorted(self.cdf, key=lambda item: item.percentile)

    def to_metaculus_cdf(
        self,
        lower_bound: float,
        upper_bound: float,
        open_lower_bound: bool = False,
        open_upper_bound: bool = False,
        continuous_range: list[float] | None = None,
        num_points: int = 201,
    ) -> list[float]:
        """
        Convert the distribution into a 201-length CDF payload expected by Metaculus.

        The CDF must comply with Metaculus API constraints:
        - CDF must be monotonically increasing by at least 5e-05 at every step
        - CDF at lower bound must be 0.0 (if lower bound is closed)
        - CDF at upper bound must be at most 0.999 (if upper bound is open)

        Args:
            lower_bound: Lower bound of the question
            upper_bound: Upper bound of the question
            open_lower_bound: Whether the lower bound is open (exclusive)
            open_upper_bound: Whether the upper bound is open (exclusive)
            continuous_range: Optional list of bin values for logarithmic scale questions
            num_points: Number of points in the CDF (default 201)
        """
        sorted_cdf = self.get_cdf()

        # Extract our percentile data (value, percentile pairs)
        our_values = [point.value for point in sorted_cdf]
        our_percentiles = [point.percentile for point in sorted_cdf]

        # Add anchor points at the bounds to ensure smooth interpolation
        # This prevents flat extrapolation outside the declared percentile range
        percentile_min = min(our_percentiles)
        percentile_max = max(our_percentiles)
        value_min = our_values[our_percentiles.index(percentile_min)]
        value_max = our_values[our_percentiles.index(percentile_max)]

        # Add lower bound anchor
        if open_lower_bound:
            # For open bounds, ensure at least 0.5% probability mass below the bound
            if lower_bound < value_min:
                # Add anchor halfway between 0 and minimum percentile, but at least 0.005
                anchor_percentile = max(0.005, 0.5 * percentile_min)
                our_values.insert(0, lower_bound)
                our_percentiles.insert(0, anchor_percentile)
        else:
            # For closed bounds, CDF at bound should be 0.0
            if lower_bound < value_min:
                our_values.insert(0, lower_bound)
                our_percentiles.insert(0, 0.0)

        # Add upper bound anchor
        if open_upper_bound:
            # For open bounds, ensure at least 0.5% probability mass above the bound
            # Set anchor conservatively to avoid exceeding 0.999 after monotonicity enforcement
            if upper_bound > value_max:
                # Use 0.98 to leave room for monotonicity constraints to add steps
                # This still ensures significant probability mass in the tail
                anchor_percentile = min(0.98, percentile_max + 0.5 * (1.0 - percentile_max))
                our_values.append(upper_bound)
                our_percentiles.append(anchor_percentile)
        else:
            # For closed bounds, CDF at bound should be 1.0
            if upper_bound > value_max:
                our_values.append(upper_bound)
                our_percentiles.append(1.0)

        # Create value grid - use continuous_range if available (for log scale),
        # otherwise fall back to linear spacing
        if continuous_range is not None and len(continuous_range) == num_points:
            # Use the provided continuous_range for logarithmic scale questions
            value_grid = np.array(continuous_range)
        else:
            # Fall back to linear spacing
            value_grid = np.linspace(lower_bound, upper_bound, num=num_points)

        # Interpolate to get CDF values (percentiles) at each grid point
        # np.interp requires x-coordinates (our_values) to be increasing
        cdf_values = np.interp(value_grid, our_values, our_percentiles)

        # Apply Metaculus constraints
        min_step = 5e-05

        # 1. If lower bound is closed, CDF at lower bound must be 0.0
        if not open_lower_bound:
            cdf_values[0] = 0.0

        # 2. Determine upper bound for CDF
        max_cdf_value = 0.999 if open_upper_bound else 1.0
        cdf_values[-1] = min(cdf_values[-1], max_cdf_value)

        # 3. Ensure monotonicity with minimum step of 5e-05
        # Forward pass: enforce monotonic increase
        for i in range(1, len(cdf_values)):
            if cdf_values[i] < cdf_values[i - 1] + min_step:
                cdf_values[i] = cdf_values[i - 1] + min_step

        # 4. Backward pass: if we exceeded the upper bound, compress the distribution
        if cdf_values[-1] > max_cdf_value:
            # We need to rescale to fit within bounds
            # Find the first point that would cause us to exceed bounds
            overshoot = cdf_values[-1] - max_cdf_value
            cdf_values[-1] = max_cdf_value

            # Propagate backwards to maintain monotonicity
            for i in range(len(cdf_values) - 2, -1, -1):
                max_allowed = cdf_values[i + 1] - min_step
                if cdf_values[i] > max_allowed:
                    cdf_values[i] = max_allowed

        # 5. Final validation: ensure bounds are respected
        cdf_values = np.clip(cdf_values, 0.0, max_cdf_value)

        # Extra safety: explicitly enforce the upper bound constraint
        # This catches any floating point precision issues
        if cdf_values[-1] > max_cdf_value:
            cdf_values[-1] = max_cdf_value

        return cdf_values.tolist()


@dataclass
class ReasonedPrediction:
    """
    Coupled prediction value with reasoning text.
    """

    prediction_value: Any
    reasoning: str


def create_forecast_payload(
    forecast: float | PredictedOptionList | list[float] | NumericDistribution | dict[str, float],
    question_type: ForecastType,
    question: MetaculusQuestion | None = None,
) -> dict[str, Any]:
    """
    Generate the Metaculus payload in the correct format.

    Args:
        forecast: The prediction to format
        question_type: Type of question (binary, numeric, multiple_choice, discrete)
        question: Optional question object (required for numeric questions)
    """
    if question_type == "binary":
        probability = float(forecast)  # type: ignore[arg-type]
        return {
            "probability_yes": probability,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }

    if question_type in ("multiple_choice", "discrete"):
        if not isinstance(forecast, dict):
            raise TypeError("Multiple choice forecast must be a mapping of option -> probability.")
        normalized = normalize_multiple_choice_probs(forecast)

        if question_type == "discrete":
            # Metaculus expects discrete forecasts as a CDF (length inbound_outcome_count + 1)
            discrete_cdf = _discrete_probs_to_cdf(normalized, question)
            return {
                "probability_yes": None,
                "probability_yes_per_category": None,
                "continuous_cdf": discrete_cdf,
            }

        return {
            "probability_yes": None,
            "probability_yes_per_category": normalized,
            "continuous_cdf": None,
        }

    # For numeric questions, we need the question bounds
    if not isinstance(question, NumericQuestion):
        raise ValueError("Numeric forecasts require a NumericQuestion object with bounds")

    cdf_payload = _normalize_numeric_forecast(forecast, question)
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": cdf_payload,
    }


def normalize_multiple_choice_probs(options: PredictedOptionList) -> PredictedOptionList:
    """
    Normalize multiple choice probabilities to sum to 1.0.
    """
    total = sum(options.values())
    if total <= 0:
        raise ValueError("Multiple choice probabilities must sum to a positive value.")
    return {label: prob / total for label, prob in options.items()}


def _normalize_numeric_forecast(
    forecast: float | list[float] | NumericDistribution | dict[str, float],
    question: NumericQuestion,
) -> list[float] | dict[str, float]:
    """
    Normalize numeric forecast into Metaculus-compatible payloads.

    Args:
        forecast: The prediction distribution
        question: NumericQuestion with bounds information
    """
    if isinstance(forecast, NumericDistribution):
        if question.lower_bound is None or question.upper_bound is None:
            raise ValueError(
                f"NumericQuestion must have lower_bound and upper_bound set. "
                f"Question ID: {question.id}, bounds: [{question.lower_bound}, {question.upper_bound}]"
            )

        return forecast.to_metaculus_cdf(
            lower_bound=question.lower_bound,
            upper_bound=question.upper_bound,
            open_lower_bound=question.open_lower_bound,
            open_upper_bound=question.open_upper_bound,
            continuous_range=question.continuous_range,
        )

    if isinstance(forecast, list):
        return [float(val) for val in forecast]

    if isinstance(forecast, dict):
        return {k: float(v) for k, v in forecast.items()}

    raise TypeError("Numeric forecast must be a NumericDistribution, list, or dict.")


def _discrete_probs_to_cdf(probabilities: PredictedOptionList, question: MetaculusQuestion | None) -> list[float]:
    """
    Convert a discrete probability mass function into the CDF format Metaculus expects.

    The discrete API uses `continuous_cdf` with length inbound_outcome_count + 1.

    """
    option_order = question.options if question and question.options else list(probabilities.keys())
    seen = set()
    ordered_labels = []
    for label in option_order:
        if label in seen:
            continue
        seen.add(label)
        ordered_labels.append(label)

    # Include any probability labels we didn't see in the question options to preserve mass
    for label in probabilities.keys():
        if label in seen:
            continue
        seen.add(label)
        ordered_labels.append(label)

    cdf: list[float] = []
    running_total = 0.0
    for label in ordered_labels:
        running_total += probabilities.get(label, 0.0)
        cdf.append(running_total)

    final_mass = cdf[-1]
    if final_mass <= 0:
        raise ValueError("Discrete probabilities must sum to a positive value.")

    max_cdf_value = 0.999 if question and question.open_upper_bound else 1.0
    scale = max_cdf_value / final_mass

    scaled_cdf = [0.0]
    for val in cdf[1:]:
        scaled_val = min(max_cdf_value, val * scale)
        scaled_cdf.append(scaled_val)

    # Guard against floating point drift
    scaled_cdf[-1] = max_cdf_value
    return scaled_cdf


def _extract_question_type(data: dict[str, Any]) -> ForecastType:
    qtype = data.get("type") or data.get("forecast_type") or data.get("question_type")
    if qtype in ("binary", "probability"):
        return "binary"
    if qtype in ("multiple_choice", "categorical"):
        return "multiple_choice"
    if qtype == "discrete":
        return "discrete"
    return "numeric"


def _extract_options(data: dict[str, Any]) -> list[str] | None:
    options: list[str] = []
    if isinstance(data.get("options"), list):
        options = [str(opt.get("name", opt)) if isinstance(opt, dict) else str(opt) for opt in data["options"]]

    possibilities = data.get("possibilities", {})
    if isinstance(possibilities, dict):
        answers = possibilities.get("answers")
        if isinstance(answers, list):
            for ans in answers:
                if isinstance(ans, dict):
                    label = ans.get("name") or ans.get("value") or ans.get("label") or str(ans.get("id", ""))
                    options.append(str(label))
                else:
                    options.append(str(ans))

    # For discrete questions, extract options from the scaling field
    if not options and data.get("type") == "discrete":
        scaling = data.get("scaling", {})
        if isinstance(scaling, dict):
            inbound_count = scaling.get("inbound_outcome_count")
            nominal_min = scaling.get("nominal_min")
            nominal_max = scaling.get("nominal_max")
            open_upper = scaling.get("open_upper_bound", False)

            if inbound_count is not None and nominal_min is not None:
                start = int(nominal_min)
                count = int(inbound_count)
                # Base bins (Metaculus defines inbound_outcome_count bins starting at nominal_min)
                for i in range(start, start + count):
                    options.append(str(i))
                # If open upper bound, add an overflow bin above the last nominal value
                if open_upper and options:
                    options.append(f"more than {options[-1]}")
            elif nominal_min is not None and nominal_max is not None:
                # Fallback to old behavior if inbound_count missing
                for i in range(int(nominal_min), int(nominal_max) + 1):
                    if i == int(nominal_max) and scaling.get("open_upper_bound", False):
                        options.append(f"more than {i}")
                    else:
                        options.append(str(i))

    return options or None


def _extract_numeric_bounds(data: dict[str, Any]) -> tuple[Optional[float], Optional[float], bool, bool, Optional[str], Optional[list[float]]]:
    lower_bound = None
    upper_bound = None
    # Start with top-level flags if present; some API payloads expose them outside of possibilities/scale
    open_lower = bool(data.get("open_lower_bound", False))
    open_upper = bool(data.get("open_upper_bound", False))
    unit = None
    continuous_range = None

    possibilities = data.get("possibilities", {})
    if isinstance(possibilities, dict):
        # Possibilities may also expose open bound flags directly
        open_lower = bool(possibilities.get("open_lower_bound", open_lower))
        open_upper = bool(possibilities.get("open_upper_bound", open_upper))

        scale = possibilities.get("scale") or {}
        if isinstance(scale, dict):
            # Prefer scale values when present, falling back to previously seen bounds
            lower_bound = scale.get("min", lower_bound)
            upper_bound = scale.get("max", upper_bound)

            # API responses have used both open_min/open_max and open_*_bound; respect whichever is present
            open_lower = bool(scale.get("open_min", scale.get("open_lower_bound", open_lower)))
            open_upper = bool(scale.get("open_max", scale.get("open_upper_bound", open_upper)))
            unit = scale.get("unit")

    # Check top-level scaling field (used when possibilities is None)
    scaling = data.get("scaling")
    if isinstance(scaling, dict):
        # Extract bounds from scaling if not already set
        if lower_bound is None:
            lower_bound = scaling.get("range_min")
        if upper_bound is None:
            upper_bound = scaling.get("range_max")

        # Extract open bound flags from scaling
        open_lower = bool(scaling.get("open_lower_bound", open_lower))
        open_upper = bool(scaling.get("open_upper_bound", open_upper))

        # Extract continuous_range from scaling
        if continuous_range is None:
            continuous_range = scaling.get("continuous_range")

    # Extract continuous_range from top-level or from data
    if continuous_range is None:
        continuous_range = data.get("continuous_range")

    return lower_bound, upper_bound, open_lower, open_upper, unit, continuous_range


__all__ = [
    "BinaryQuestion",
    "MetaculusQuestion",
    "MultipleChoiceQuestion",
    "NumericDistribution",
    "NumericQuestion",
    "Percentile",
    "PredictedOptionList",
    "ReasonedPrediction",
    "create_forecast_payload",
    "normalize_multiple_choice_probs",
]
