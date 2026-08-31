from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import entropy as shannon_entropy

try:
    from state.dialogue_state import no_preference_attributes as _no_preference
except ImportError:
    def _no_preference(session_profile: "Mapping[str, Any]") -> List[str]:
        prefix = "no_preference:"
        return [
            str(v)[len(prefix):]
            for v in (session_profile.get("rejected") or [])
            if str(v).startswith(prefix)
        ]

__all__ = [
    "ALLOWED_ATTRIBUTES",
    "ASKABLE_ATTRIBUTES",
    "IDEAL_VALUE_COUNT",
    "AskState",
    "ask_state_from_profile",
    "AttributeScore",
    "AttributeTable",
    "WeightedEntropy",
    "MAX_TAG_WEIGHT",
    "FEATURE_SCORE",
    "HARD_REFUSAL_LIMIT",
    "MIN_SCORE",
    "TAG_AFFINITY",
    "TAG_STRENGTH",
    "explain_selection",
    "preference_weight",
    "rank_attributes",
    "rank_weights",
    "select_attribute",
]

ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)

ASKABLE_ATTRIBUTES = ("material", "color", "size", "style", "budget", "use_case", "brand")

_MISSING = -1

_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    "material": re.compile(
        r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I
    ),
    "color": re.compile(
        r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I
    ),
    "size": re.compile(r"\b(xs|xxl|xl|small|medium|large|petite|plus size|one size)\b", re.I),
    "style": re.compile(
        r"\b(long sleeve|short sleeve|sleeveless|v-neck|crew neck|turtleneck|"
        r"slim fit|relaxed fit|loose fit|fitted)\b",
        re.I,
    ),
    "use_case": re.compile(r"\b(hiking|running|gym|winter|outdoor|work)\b", re.I),
}

_BUDGET_BANDS: Tuple[Tuple[float, str], ...] = (
    (15.0, "under 15"),
    (30.0, "15 to 30"),
    (60.0, "30 to 60"),
    (120.0, "60 to 120"),
)

TAG_AFFINITY: Dict[str, Dict[str, float]] = {
    "fit": {"size": 1.6, "style": 1.3},
    "material": {"material": 1.8},
    "comfort": {"material": 1.4, "size": 1.3, "style": 1.1},
    "style": {"style": 1.6, "color": 1.3},
    "durability": {"material": 1.5},
    "performance": {"use_case": 1.5, "material": 1.2},
    "warmth": {"material": 1.5, "use_case": 1.3},
    "weather": {"use_case": 1.6, "material": 1.3},
    "general shopping": {},
}

TAG_STRENGTH = 1.0
MAX_TAG_WEIGHT = 2.5

ANSWERABILITY: Dict[str, float] = {
    "material": 0.77,
    "color": 0.26,
    "style": 0.09,
    "size": 0.04,
    "use_case": 0.02,
    "budget": 0.00,
    "brand": 0.00,
}

IDEAL_VALUE_COUNT = 12.0
FEATURE_SCORE = 0.30
MIN_SCORE = 0.0
_REFUSAL_DECAY = 0.55
HARD_REFUSAL_LIMIT = 2


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def _price_band(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        price = float(value)
    except (TypeError, ValueError):
        cleaned = re.sub(r"[^0-9.]", "", str(value))
        if not cleaned or cleaned.count(".") > 1:
            return ""
        price = float(cleaned)
    if price < 0:
        return ""
    for ceiling, label in _BUDGET_BANDS:
        if price <= ceiling:
            return label
    return "over 120"


def _extract(product: dict) -> Dict[str, str]:
    corpus = " ".join(
        (
            _text(product.get("title")),
            _text(product.get("features")),
            _text(product.get("details")),
            _text(product.get("categories")),
        )
    )
    values = {name: "" for name in ASKABLE_ATTRIBUTES}
    for name, pattern in _PATTERNS.items():
        found = pattern.search(corpus)
        if found:
            values[name] = found.group(1).lower()
    values["budget"] = _price_band(product.get("price"))
    values["brand"] = _text(product.get("store")).strip().lower()
    return values


@dataclass
class AttributeTable:
    codes: Dict[str, np.ndarray]
    vocabs: Dict[str, List[str]]
    n_products: int

    @classmethod
    def load(
        cls,
        catalog_path: "str | Path" = "data/catalog.jsonl",
        attributes: Sequence[str] = ASKABLE_ATTRIBUTES,
    ) -> "AttributeTable":
        raw: Dict[str, List[str]] = {name: [] for name in attributes}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                values = _extract(json.loads(line))
                for name in attributes:
                    raw[name].append(values.get(name, ""))

        codes: Dict[str, np.ndarray] = {}
        vocabs: Dict[str, List[str]] = {}
        n_products = len(next(iter(raw.values()))) if raw else 0

        for name, column in raw.items():
            values = np.asarray(column, dtype=object)
            vocab_array, inverse = np.unique(values, return_inverse=True)
            coded = inverse.astype(np.int32)
            if len(vocab_array) and vocab_array[0] == "":
                coded = np.where(coded == 0, _MISSING, coded - 1)
                vocab_array = vocab_array[1:]
            codes[name] = coded
            vocabs[name] = [str(v) for v in vocab_array]

        return cls(codes=codes, vocabs=vocabs, n_products=n_products)

    def attributes(self) -> Tuple[str, ...]:
        return tuple(self.codes)


@dataclass
class AskState:
    confirmed: set = field(default_factory=set)
    banned: set = field(default_factory=set)
    refusals: Dict[str, int] = field(default_factory=dict)

    def confirm(self, attribute: str) -> None:
        self.confirmed.add(attribute)
        self.refusals.pop(attribute, None)

    def ban(self, attribute: str) -> None:
        self.banned.add(attribute)
        self.refusals.pop(attribute, None)

    def refute(self, attribute: str) -> None:
        if attribute not in self.confirmed and attribute not in self.banned:
            self.refusals[attribute] = self.refusals.get(attribute, 0) + 1

    def is_blocked(self, attribute: str) -> bool:
        return attribute in self.confirmed or attribute in self.banned

    def penalty(self, attribute: str) -> float:
        return _REFUSAL_DECAY ** self.refusals.get(attribute, 0)


_TRACKABLE_ATTRIBUTES = set(ASKABLE_ATTRIBUTES) | {"feature"}


def ask_state_from_profile(
    session_profile: Mapping[str, Any],
    previous_ask_attribute: str = "",
    refusals: Optional[Mapping[str, int]] = None,
) -> "AskState":
    profile = dict(session_profile or {})
    confirmed = {key for key in ASKABLE_ATTRIBUTES if profile.get(key)}
    banned = set(_no_preference(profile)) & _TRACKABLE_ATTRIBUTES

    counts = dict(refusals or {})
    asked = (previous_ask_attribute or "").strip()
    if asked and asked in _TRACKABLE_ATTRIBUTES and asked not in confirmed and asked not in banned:
        counts[asked] = counts.get(asked, 0) + 1

    fatigued = {attr for attr, n in counts.items() if n >= HARD_REFUSAL_LIMIT}
    banned = banned | (fatigued & _TRACKABLE_ATTRIBUTES)

    return AskState(confirmed=confirmed, banned=banned, refusals=counts)


@dataclass
class AttributeScore:
    attribute: str
    score: float
    entropy: float
    normalised_entropy: float
    head_split: float
    coverage: float
    distinct_values: int
    tag_weight: float = 1.0
    refusal_penalty: float = 1.0
    blocked: bool = False
    top_values: List[Tuple[str, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "attribute": self.attribute,
            "score": None if self.score == float("-inf") else round(self.score, 4),
            "entropy_bits": round(self.entropy, 4),
            "normalised_entropy": round(self.normalised_entropy, 4),
            "head_split": round(self.head_split, 4),
            "coverage": round(self.coverage, 4),
            "distinct_values": self.distinct_values,
            "tag_weight": round(self.tag_weight, 4),
            "refusal_penalty": round(self.refusal_penalty, 4),
            "blocked": self.blocked,
            "top_values": [(value, round(mass, 4)) for value, mass in self.top_values],
        }

def preference_weight(
    attribute: str,
    preference_tags: Optional[Sequence[str]],
    strength: Optional[float] = None,
) -> float:
    if not preference_tags:
        return 1.0
    scale = TAG_STRENGTH if strength is None else strength
    if scale == 0.0:
        return 1.0
    weight = 1.0
    for tag in preference_tags:
        affinity = TAG_AFFINITY.get(str(tag).strip().lower())
        if affinity and attribute in affinity:
            weight *= 1.0 + scale * (affinity[attribute] - 1.0)
    return min(weight, MAX_TAG_WEIGHT)


def rank_weights(size: int, half_life: float = 20.0) -> np.ndarray:
    if size <= 0:
        return np.empty(0, dtype=np.float64)
    ranks = np.arange(1, size + 1, dtype=np.float64)
    return np.power(0.5, ranks / float(half_life))


def _pool_indices(pool: Iterable[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[int] = []
    positions: List[int] = []
    for position, item in enumerate(pool):
        index = item.get("catalog_index")
        if index is None:
            continue
        rows.append(int(index))
        positions.append(position)
    return np.asarray(rows, dtype=np.int64), np.asarray(positions, dtype=np.int64)


def _weighted_entropy(
    codes: np.ndarray, weights: np.ndarray, n_values: int
) -> Tuple[float, float, float, np.ndarray]:
    known = codes != _MISSING
    total_mass = float(weights.sum())
    if total_mass <= 0.0 or not known.any() or n_values <= 0:
        return 0.0, 0.0, 1.0, np.zeros(0, dtype=np.float64)

    mass = np.bincount(codes[known], weights=weights[known], minlength=n_values).astype(np.float64)
    known_mass = float(mass.sum())
    if known_mass <= 0.0:
        return 0.0, 0.0, 1.0, mass

    entropy = float(shannon_entropy(mass, base=2))
    coverage = known_mass / total_mass
    largest_share = float(mass.max() / known_mass)
    return entropy, coverage, largest_share, mass


def rank_attributes(
    pool: Sequence[Dict[str, Any]],
    table: AttributeTable,
    state: Optional[AskState] = None,
    preference_tags: Optional[Sequence[str]] = None,
    half_life: float = 20.0,
    strength: Optional[float] = None,
    include_blocked: bool = False,
) -> List[AttributeScore]:
    state = state or AskState()
    skip_feature = {"feature"} if "feature" in state.banned else set()
    rows, positions = _pool_indices(pool)
    if rows.size == 0:
        return []

    weights = rank_weights(len(pool), half_life=half_life)[positions]

    scored: List[AttributeScore] = []
    for attribute in table.attributes():
        blocked = state.is_blocked(attribute)
        if blocked and not include_blocked:
            continue

        vocab = table.vocabs[attribute]
        codes = table.codes[attribute][rows]
        entropy, coverage, largest_share, mass = _weighted_entropy(codes, weights, len(vocab))
        distinct = int((mass > 0.0).sum()) if mass.size else 0

        if blocked or distinct < 2:
            scored.append(
                AttributeScore(
                    attribute=attribute,
                    score=float("-inf"),
                    entropy=entropy,
                    normalised_entropy=0.0,
                    head_split=0.0,
                    coverage=coverage,
                    distinct_values=distinct,
                    blocked=blocked,
                )
            )
            continue

        normalised = float(entropy / np.log2(distinct))
        head_split = 1.0 - largest_share

        tag_weight = preference_weight(attribute, preference_tags, strength)

        refusal_penalty = state.penalty(attribute)
        cardinality = min(1.0, IDEAL_VALUE_COUNT / distinct)
        base = 0.6 * normalised + 0.4 * head_split
        score = (
            base
            * coverage
            * cardinality
            * ANSWERABILITY.get(attribute, 0.7)
            * tag_weight
            * refusal_penalty
        )

        top = sorted(
            ((vocab[i], float(mass[i])) for i in np.nonzero(mass)[0]),
            key=lambda pair: -pair[1],
        )[:5]

        scored.append(
            AttributeScore(
                attribute=attribute,
                score=float(score),
                entropy=entropy,
                normalised_entropy=normalised,
                head_split=head_split,
                coverage=coverage,
                distinct_values=distinct,
                tag_weight=tag_weight,
                refusal_penalty=refusal_penalty,
                top_values=top,
            )
        )
    if "feature" not in skip_feature:
        feature_score = FEATURE_SCORE * state.penalty("feature")
        scored.append(
            AttributeScore(
                attribute="feature",
                score=float(feature_score),
                entropy=0.0,
                normalised_entropy=0.0,
                head_split=0.0,
                coverage=1.0,
                distinct_values=0,
                refusal_penalty=state.penalty("feature"),
                blocked=False,
            )
        )
    elif include_blocked:
        scored.append(
            AttributeScore(
                attribute="feature", score=float("-inf"), entropy=0.0,
                normalised_entropy=0.0, head_split=0.0, coverage=1.0,
                distinct_values=0, blocked=True,
            )
        )

    scored.sort(key=lambda item: (-item.score, item.attribute))
    return scored


class WeightedEntropy:
    def __init__(
        self,
        catalog_path: "str | Path" = "data/catalog.jsonl",
        table: Optional[AttributeTable] = None,
        half_life: float = 20.0,
        strength: Optional[float] = None,
        min_score: float = MIN_SCORE,
    ) -> None:
        self.table = table if table is not None else AttributeTable.load(catalog_path)
        self.half_life = half_life
        self.strength = strength
        self.min_score = min_score

    @staticmethod
    def _pool(candidate_indices: Sequence[int]) -> List[Dict[str, Any]]:
        return [{"catalog_index": int(index)} for index in candidate_indices]

    def _ask_state(self, state: Mapping[str, Any]) -> AskState:
        return ask_state_from_profile(
            state.get("session_profile") or {},
            str(state.get("previous_ask_attribute") or ""),
            state.get("attribute_refusals") or None,
        )

    @staticmethod
    def _tags(state: Mapping[str, Any]) -> List[str]:
        profile = state.get("user_profile") or {}
        return [str(tag) for tag in (profile.get("preference_tags") or [])]

    def explain_selection(
        self,
        state: Mapping[str, Any],
        top_500_candidate_indices: Sequence[int],
        include_blocked: bool = False,
    ) -> List[List[Any]]:
        ranked = rank_attributes(
            self._pool(top_500_candidate_indices),
            self.table,
            self._ask_state(state),
            self._tags(state),
            half_life=self.half_life,
            strength=self.strength,
            include_blocked=include_blocked,
        )
        if not include_blocked:
            ranked = [item for item in ranked if np.isfinite(item.score)]
        return [
            [
                item.attribute,
                round(item.entropy, 4),
                None if not np.isfinite(item.score) else round(item.score, 4),
            ]
            for item in ranked
        ]

    def select(
        self,
        state: Mapping[str, Any],
        top_500_candidate_indices: Sequence[int],
    ) -> Tuple[Optional[str], List[List[Any]]]:
        scored = self.explain_selection(state, top_500_candidate_indices)
        if not scored:
            return None, scored
        best_attribute, _entropy, best_score = scored[0]
        if best_score is None or best_score < self.min_score:
            return None, scored
        return str(best_attribute), scored


def explain_selection(
    pool: Sequence[Dict[str, Any]],
    table: AttributeTable,
    state: Optional[AskState] = None,
    preference_tags: Optional[Sequence[str]] = None,
    half_life: float = 20.0,
    strength: Optional[float] = None,
    min_score: float = MIN_SCORE,
    include_blocked: bool = True,
) -> Dict[str, Any]:
    ranked = rank_attributes(
        pool, table, state, preference_tags,
        half_life=half_life, strength=strength, include_blocked=include_blocked,
    )
    eligible = [item for item in ranked if not item.blocked and np.isfinite(item.score)]
    best = eligible[0] if eligible else None
    asked = bool(best is not None and best.score >= min_score)

    return {
        "selected": best.attribute if asked else None,
        "asked": asked,
        "reason": (
            "argmax above gate" if asked
            else "no eligible attribute" if best is None
            else f"best score {best.score:.4f} below min_score {min_score}"
        ),
        "pool_size": len(pool),
        "min_score": min_score,
        "scored": [item.as_dict() for item in ranked],
    }

def select_attribute(
    pool: Sequence[Dict[str, Any]],
    table: AttributeTable,
    state: Optional[AskState] = None,
    preference_tags: Optional[Sequence[str]] = None,
    half_life: float = 20.0,
    strength: Optional[float] = None,
    min_score: float = MIN_SCORE,
) -> Tuple[Optional[str], List[AttributeScore]]:
    ranked = rank_attributes(pool, table, state, preference_tags, half_life=half_life, strength=strength)
    if not ranked:
        return None, ranked
    best = ranked[0]
    if not np.isfinite(best.score) or best.score < min_score:
        return None, ranked
    return best.attribute, ranked