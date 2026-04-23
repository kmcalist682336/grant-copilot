"""PeerRetriever — axis-specific peer ranking for a target geography.

Phase 3 node. Given an anchor (MSA / county / neighborhood as an
aggregated tract set) and an axis name, returns the top-K most similar
peer entities drawn from the ``peer_features`` table.

Peers are ranked via z-score Mahalanobis-ish distance within an
axis-specific subset of the catalog's 104 features. Candidate pools
are size-bucketed so a small place doesn't get compared against a
giant city that happens to have similar income.

No LLM. All arithmetic + SQL reads. Deterministic for a given
(anchor, axis, pool) tuple.

Public API:
    PeerRef                 — one ranked peer
    PeerRetrievalError      — raised on bad inputs or missing data
    PeerRetriever.peers()   — entry point
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from scripts.chatbot.peer_features_catalog import AXES, FEATURES

logger = logging.getLogger(__name__)

GeoLevel = Literal["place", "county", "msa", "neighborhood"]


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class PeerRef(BaseModel):
    """One ranked peer."""

    model_config = ConfigDict(extra="ignore")

    geo_level: GeoLevel
    geo_id: str
    geo_name: str
    state_fips: Optional[str] = None
    population: Optional[int] = None

    # Axis this peer came from (one peer may appear in multiple axes
    # independently — each call returns one axis at a time).
    axis: str
    rank: int = Field(..., description="1-indexed rank within the axis.")
    distance: float = Field(
        ...,
        description=(
            "Mean z-score distance (lower = more similar). Includes "
            "proximity + size modifiers."
        ),
    )
    raw_distance: float = Field(
        default=0.0,
        description=(
            "Pre-modifier z-score distance. Useful for debugging which "
            "peers were promoted by the proximity/size bonuses."
        ),
    )
    features_compared: int = Field(
        ..., description="How many axis features contributed to the score.",
    )
    # Concrete feature values for the peer, so the synthesizer can say
    # "Chatham County's poverty rate is 15.1%, comparable to DeKalb's
    # 13.5%" instead of just naming peers. Populated by peer_context
    # from the peer's full feature vector when relevant.
    feature_values: dict[str, float] = Field(default_factory=dict)
    match_explanation: str = Field(
        default="",
        description=(
            "One-sentence human-readable justification: why this "
            "particular geography was picked (proximity, population "
            "scale, top 1-2 matching features)."
        ),
    )


class PeerRetrievalError(RuntimeError):
    """Raised when inputs are invalid or the DB lacks coverage."""


# ---------------------------------------------------------------------------
# Geographic proximity — Census Bureau divisions (finer-grained) + 4
# regions (coarse). Mapping by state FIPS. Used to bias ranking toward
# peers that are physically closer: SW Atlanta should prefer Macon
# (same state), Birmingham/Nashville/NOLA (same region, different
# division), and Charlotte/Miami (same division) over, say, Boise or
# Burlington. We don't have centroids in peer_features.sqlite, so
# division+region is the cheap proxy; it captures "close by" well
# enough for grant-narrative peer picks.
# ---------------------------------------------------------------------------

_DIVISION_BY_STATE: dict[str, str] = {
    # New England
    "09": "new_england", "23": "new_england", "25": "new_england",
    "33": "new_england", "44": "new_england", "50": "new_england",
    # Middle Atlantic
    "34": "mid_atlantic", "36": "mid_atlantic", "42": "mid_atlantic",
    # East North Central
    "17": "east_north_central", "18": "east_north_central",
    "26": "east_north_central", "39": "east_north_central",
    "55": "east_north_central",
    # West North Central
    "19": "west_north_central", "20": "west_north_central",
    "27": "west_north_central", "29": "west_north_central",
    "31": "west_north_central", "38": "west_north_central",
    "46": "west_north_central",
    # South Atlantic
    "10": "south_atlantic", "11": "south_atlantic", "12": "south_atlantic",
    "13": "south_atlantic", "24": "south_atlantic", "37": "south_atlantic",
    "45": "south_atlantic", "51": "south_atlantic", "54": "south_atlantic",
    # East South Central
    "01": "east_south_central", "21": "east_south_central",
    "28": "east_south_central", "47": "east_south_central",
    # West South Central
    "05": "west_south_central", "22": "west_south_central",
    "40": "west_south_central", "48": "west_south_central",
    # Mountain
    "04": "mountain", "08": "mountain", "16": "mountain", "30": "mountain",
    "32": "mountain", "35": "mountain", "49": "mountain", "56": "mountain",
    # Pacific
    "02": "pacific", "06": "pacific", "15": "pacific",
    "41": "pacific", "53": "pacific",
}

_REGION_BY_DIVISION: dict[str, str] = {
    "new_england": "northeast", "mid_atlantic": "northeast",
    "east_north_central": "midwest", "west_north_central": "midwest",
    "south_atlantic": "south", "east_south_central": "south",
    "west_south_central": "south",
    "mountain": "west", "pacific": "west",
}


def _proximity_tier(
    anchor_state: Optional[str], peer_state: Optional[str],
) -> str:
    """Return one of 'same_state' / 'same_division' / 'same_region' /
    'nationwide' describing how close two states are."""
    if not anchor_state or not peer_state:
        return "nationwide"
    if anchor_state == peer_state:
        return "same_state"
    a_div = _DIVISION_BY_STATE.get(anchor_state)
    p_div = _DIVISION_BY_STATE.get(peer_state)
    if a_div and p_div and a_div == p_div:
        return "same_division"
    a_reg = _REGION_BY_DIVISION.get(a_div) if a_div else None
    p_reg = _REGION_BY_DIVISION.get(p_div) if p_div else None
    if a_reg and p_reg and a_reg == p_reg:
        return "same_region"
    return "nationwide"


# Discount on raw distance for each tier. Higher discount = stronger
# preference for nearby peers. Tuned so same-state wins over a
# marginally-closer-in-feature-space peer on a different continent,
# but doesn't completely override feature-space similarity.
_PROXIMITY_DISCOUNT: dict[str, float] = {
    "same_state": 0.75,        # 25% distance discount
    "same_division": 0.88,     # 12% discount (e.g. GA → FL/NC/SC)
    "same_region": 0.94,       # 6% discount (GA → AL/TN/LA)
    "nationwide": 1.00,
}


def _size_modifier(
    anchor_pop: Optional[int], peer_pop: Optional[int],
) -> float:
    """Weight peers toward larger cities — a bigger comparator makes
    grant narratives more persuasive. The anchor's own size bucket
    already caps how different peers can be; within that bucket we
    give a small nudge to the upper half."""
    if not anchor_pop or not peer_pop or anchor_pop <= 0:
        return 1.0
    ratio = peer_pop / anchor_pop
    if ratio >= 1.25:
        return 0.95           # 5% discount for "stronger comparator"
    if ratio >= 0.8:
        return 1.00           # roughly the same size
    return 1.03               # mild penalty for distinctly smaller peers


# ---------------------------------------------------------------------------
# Per-peer explanation strings — surface why THIS peer was picked
# ---------------------------------------------------------------------------

_PROXIMITY_PHRASE: dict[str, str] = {
    "same_state": "same state",
    "same_division": "same Census division",
    "same_region": "same Census region",
    "nationwide": "nationwide peer",
}


def _humanize_feature(name: str) -> str:
    """Translate a snake_case feature key into a reader-friendly
    phrase for the explanation string. Falls back to underscores→
    spaces when the feature isn't explicitly mapped."""
    overrides = {
        "median_household_income": "median household income",
        "poverty_rate": "poverty rate",
        "child_poverty_rate": "child poverty rate",
        "snap_participation_rate": "SNAP participation rate",
        "pct_without_vehicle": "share without a vehicle",
        "median_gross_rent": "median gross rent",
        "median_home_value": "median home value",
        "pct_rent_burdened": "share of rent-burdened households",
        "unemployment_rate": "unemployment rate",
        "pct_children_under_18": "share of children under 18",
        "pct_seniors_65_plus": "share of seniors 65+",
        "pct_hispanic": "Hispanic share",
        "pct_black": "Black share",
        "pct_white": "White share",
        "pct_asian": "Asian share",
        "pct_foreign_born": "foreign-born share",
        "pct_limited_english": "limited-English share",
        "pct_bachelors_or_higher": "share with a bachelor's+",
    }
    if name in overrides:
        return overrides[name]
    return name.replace("_", " ")


def _format_population(p: Optional[int]) -> str:
    if not p:
        return ""
    if p >= 1_000_000:
        return f"{p/1_000_000:.1f}M"
    if p >= 1_000:
        return f"{p/1_000:.0f}k"
    return str(p)


def _compose_explanation(
    *,
    anchor: "AnchorFeatures",
    peer: "AnchorFeatures",
    tier: str,
    matched_deltas: list[tuple[str, float]],
) -> str:
    """Build a short human-readable justification for why ``peer``
    was picked for ``anchor`` on this axis. Three bits:

      1. Proximity tier (same state / division / region / nationwide)
      2. Relative population (≈ same / larger / smaller)
      3. The 1–2 features the peer matches the anchor most tightly on

    Composed as a semicolon-joined phrase so it slots cleanly into
    either prose or a REPL listing.
    """
    bits: list[str] = [_PROXIMITY_PHRASE.get(tier, "peer")]

    if peer.population and anchor.population:
        pop_str = _format_population(peer.population)
        ratio = peer.population / anchor.population
        if ratio >= 1.25:
            bits.append(f"larger population ({pop_str})")
        elif ratio >= 0.8:
            bits.append(f"similar population ({pop_str})")
        else:
            bits.append(f"smaller population ({pop_str})")
    elif peer.population:
        bits.append(f"population {_format_population(peer.population)}")

    # Top 1–2 features where the peer is closest to the anchor.
    close_feats = [f for f, d in matched_deltas[:2] if d < 1.0]
    if close_feats:
        phrase = " and ".join(_humanize_feature(f) for f in close_feats)
        bits.append(f"closely matched on {phrase}")

    return "; ".join(bits)


# ---------------------------------------------------------------------------
# Size bucketing — prevents cross-scale nonsense peers
# ---------------------------------------------------------------------------

_SIZE_BUCKETS = [
    (0, 5_000),
    (2_000, 10_000),
    (5_000, 25_000),
    (15_000, 80_000),
    (50_000, 250_000),
    (150_000, 1_000_000),
    (500_000, 10_000_000),
]


def _bucket_for(population: int) -> tuple[int, int]:
    """Pick the bucket that best matches the anchor. Buckets overlap so
    the anchor isn't stuck at an edge."""
    for lo, hi in _SIZE_BUCKETS:
        if lo <= population < hi:
            return lo, hi
    # Above the top bucket — keep the widest.
    return _SIZE_BUCKETS[-1]


# ---------------------------------------------------------------------------
# Anchor resolution
# ---------------------------------------------------------------------------

@dataclass
class AnchorFeatures:
    """A resolved anchor's features + metadata."""

    geo_level: GeoLevel
    geo_id: str
    geo_name: str
    population: Optional[int]
    state_fips: Optional[str]
    features: dict[str, float]


def _row_to_anchor(row: sqlite3.Row) -> AnchorFeatures:
    fj = json.loads(row["features_json"]) if row["features_json"] else {}
    return AnchorFeatures(
        geo_level=row["geo_level"],
        geo_id=row["geo_id"],
        geo_name=row["geo_name"],
        population=row["population"],
        state_fips=row["state_fips"],
        features={k: float(v) for k, v in fj.items() if v is not None},
    )


# ---------------------------------------------------------------------------
# Core retriever
# ---------------------------------------------------------------------------

class PeerRetriever:
    """Thread-safe wrapper over `peer_features.sqlite`."""

    def __init__(
        self, db_path: Path | str,
        *, vintage: Optional[int] = None,
    ):
        self._db_path = Path(db_path)
        if not self._db_path.exists():
            raise PeerRetrievalError(
                f"peer_features DB missing: {self._db_path}"
            )
        # We open a new connection per call — sqlite3 connections are
        # not thread-safe when shared. Lightweight enough for our scale.
        self._vintage = vintage

    # ------------------------------------------------------------------
    # Anchor lookups
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db_path)
        con.row_factory = sqlite3.Row
        return con

    def _resolve_vintage(self, con: sqlite3.Connection) -> int:
        if self._vintage is not None:
            return self._vintage
        r = con.execute(
            "SELECT MAX(vintage) AS v FROM peer_features"
        ).fetchone()
        if r is None or r["v"] is None:
            raise PeerRetrievalError("peer_features table is empty")
        return int(r["v"])

    def lookup_anchor(
        self, *, geo_level: GeoLevel, geo_id: str,
    ) -> Optional[AnchorFeatures]:
        """Fetch the feature vector for a specific geo_id, or None when
        it isn't in the DB."""
        with self._connect() as con:
            vintage = self._resolve_vintage(con)
            row = con.execute(
                """
                SELECT geo_level, geo_id, geo_name, state_fips,
                       population, features_json
                  FROM peer_features
                 WHERE geo_level = ? AND geo_id = ? AND vintage = ?
                """,
                (geo_level, geo_id, vintage),
            ).fetchone()
            return _row_to_anchor(row) if row else None

    def anchor_from_aggregated(
        self, *, features: dict[str, float], population: int,
        label: str, state_fips: Optional[str] = None,
    ) -> AnchorFeatures:
        """Build a synthetic anchor from externally-aggregated features
        (e.g. an ATL neighborhood summed over tracts, a region like
        'southwest Atlanta', etc.). No DB lookup.

        Supplying ``state_fips`` lets proximity weighting rank in-state
        peers above distant ones; leave None when the anchor spans
        multiple states or the home state is unknown.

        The caller is responsible for ensuring the feature names match
        the catalog keys.
        """
        unknown = [k for k in features if k not in FEATURES]
        if unknown:
            raise PeerRetrievalError(
                f"unknown feature names in synthetic anchor: {unknown[:5]}"
            )
        return AnchorFeatures(
            geo_level="place",       # treat as place-equivalent
            geo_id=f"synthetic::{label}",
            geo_name=label,
            population=population,
            state_fips=state_fips,
            features=dict(features),
        )

    # ------------------------------------------------------------------
    # Candidate pool
    # ------------------------------------------------------------------

    def _candidate_pool(
        self,
        con: sqlite3.Connection,
        *, vintage: int, geo_level: GeoLevel,
        anchor: AnchorFeatures,
        restrict_state: Optional[str] = None,
    ) -> list[AnchorFeatures]:
        """Return candidate peers: same geo_level, population within
        the anchor's size bucket, excluding the anchor itself.

        If ``restrict_state`` is given, limits to entities in that state
        FIPS — use this for within-MSA or within-state peer queries.
        """
        params: list = [geo_level, vintage]
        where = ["geo_level = ?", "vintage = ?", "geo_id != ?"]
        params.append(anchor.geo_id)

        if anchor.population is None:
            # Without a population we can't size-bucket; return the full
            # level so the caller at least sees something.
            logger.debug("anchor has no population; skipping size bucket")
        else:
            lo, hi = _bucket_for(anchor.population)
            where.append("population BETWEEN ? AND ?")
            params.extend([lo, hi])

        if restrict_state:
            where.append("state_fips = ?")
            params.append(restrict_state)

        sql = (
            "SELECT geo_level, geo_id, geo_name, state_fips, population, "
            "       features_json "
            "FROM peer_features WHERE " + " AND ".join(where)
        )
        return [_row_to_anchor(r) for r in con.execute(sql, params)]

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _axis_features(self, axis: str) -> list[str]:
        if axis not in AXES:
            raise PeerRetrievalError(
                f"unknown axis: {axis!r}. Available: {sorted(AXES)}"
            )
        return list(AXES[axis]["features"])

    @staticmethod
    def _z_normalize(
        pool: list[AnchorFeatures], feature_names: list[str],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Compute (mean, stddev) per feature across the pool, ignoring
        NULL/missing values. Returns two dicts keyed on feature name.

        Zero-stddev features (constant across the pool) are given a
        pseudo-stddev of 1.0 so they contribute 0 distance without
        dividing by zero — effectively disabling them for this axis.
        """
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for fname in feature_names:
            vals = [
                p.features[fname] for p in pool
                if fname in p.features
            ]
            if not vals:
                means[fname] = 0.0
                stds[fname] = 1.0
                continue
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            s = math.sqrt(var) if var > 0 else 1.0
            means[fname] = m
            stds[fname] = s
        return means, stds

    @staticmethod
    def _distance(
        anchor_vec: dict[str, float],
        peer_vec: dict[str, float],
        feature_names: list[str],
        means: dict[str, float],
        stds: dict[str, float],
    ) -> tuple[float, int, list[tuple[str, float]]]:
        """Mean squared z-score distance across features that are
        non-null on BOTH the anchor and the peer. Returns
        (mean_dist, n_features_compared, per_feature_deltas).

        ``per_feature_deltas`` is the list of ``(feature_name,
        |z_anchor - z_peer|)`` pairs, sorted ascending — the first
        few entries identify the features this peer matches the
        anchor on most tightly, which becomes the "closely matched
        on X, Y" phrase in the peer explanation.

        NULL on either side means "can't compare that axis" — we drop
        it rather than impute, so small places with sparse features
        just get ranked on fewer dimensions.
        """
        squares: list[float] = []
        deltas: list[tuple[str, float]] = []
        for fname in feature_names:
            if fname not in anchor_vec or fname not in peer_vec:
                continue
            a = (anchor_vec[fname] - means[fname]) / stds[fname]
            b = (peer_vec[fname] - means[fname]) / stds[fname]
            diff = abs(a - b)
            squares.append((a - b) ** 2)
            deltas.append((fname, diff))
        if not squares:
            return float("inf"), 0, []
        deltas.sort(key=lambda kv: kv[1])
        return math.sqrt(sum(squares) / len(squares)), len(squares), deltas

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def peers(
        self,
        anchor: AnchorFeatures,
        *,
        axis: str,
        geo_level: GeoLevel = "place",
        top_k: int = 10,
        restrict_state: Optional[str] = None,
        min_features: int = 3,
    ) -> list[PeerRef]:
        """Return top-K peers ranked by axis-specific z-score distance.

        Parameters
        ----------
        anchor
            Either a ``lookup_anchor`` result or a synthetic anchor from
            ``anchor_from_aggregated``.
        axis
            One of the keys in ``AXES`` (e.g. ``"economic"``,
            ``"housing"``, ``"race_ethnicity"``).
        geo_level
            The candidate pool's geo_level (``place`` / ``county`` / ``msa``).
        top_k
            How many peers to return.
        restrict_state
            Limit candidates to one state FIPS (e.g. ``"13"`` for Georgia).
            Use this for within-state or within-metro peer sets.
        min_features
            Drop candidates with fewer than this many comparable
            features. Prevents degenerate "peer" hits where only one
            feature was available on both sides.
        """
        feature_names = self._axis_features(axis)
        with self._connect() as con:
            vintage = self._resolve_vintage(con)
            pool = self._candidate_pool(
                con, vintage=vintage, geo_level=geo_level,
                anchor=anchor, restrict_state=restrict_state,
            )
        if not pool:
            return []
        means, stds = self._z_normalize(pool, feature_names)

        # Score every candidate: raw z-distance, then multiplied by
        # proximity + size modifiers so the rank order biases toward
        # nearby and larger comparators without overriding outright
        # feature-space dissimilarity.
        scored: list[
            tuple[float, float, int, list[tuple[str, float]],
                  str, AnchorFeatures]
        ] = []
        for p in pool:
            raw_dist, n, deltas = self._distance(
                anchor.features, p.features, feature_names, means, stds,
            )
            if n < min_features or math.isinf(raw_dist):
                continue
            tier = _proximity_tier(anchor.state_fips, p.state_fips)
            adj_dist = (
                raw_dist
                * _PROXIMITY_DISCOUNT.get(tier, 1.0)
                * _size_modifier(anchor.population, p.population)
            )
            scored.append((adj_dist, raw_dist, n, deltas, tier, p))
        scored.sort(key=lambda t: t[0])

        out: list[PeerRef] = []
        for rank, (adj_dist, raw_dist, n, deltas, tier, p) in enumerate(
            scored[:top_k], start=1,
        ):
            axis_vals = {
                fname: p.features[fname]
                for fname in feature_names
                if fname in p.features
            }
            explanation = _compose_explanation(
                anchor=anchor, peer=p, tier=tier,
                matched_deltas=deltas,
            )
            out.append(PeerRef(
                geo_level=p.geo_level,
                geo_id=p.geo_id,
                geo_name=p.geo_name,
                state_fips=p.state_fips,
                population=p.population,
                axis=axis,
                rank=rank,
                distance=adj_dist,
                raw_distance=raw_dist,
                features_compared=n,
                feature_values=axis_vals,
                match_explanation=explanation,
            ))
        return out

    # ------------------------------------------------------------------
    # Convenience: compute all axes at once
    # ------------------------------------------------------------------

    def peers_by_axis(
        self,
        anchor: AnchorFeatures,
        *,
        axes: Optional[list[str]] = None,
        geo_level: GeoLevel = "place",
        top_k: int = 5,
        restrict_state: Optional[str] = None,
    ) -> dict[str, list[PeerRef]]:
        """Return ``{axis_name: [top-K peers, …], …}`` for many axes in
        one shot. Uses default ``min_features`` per axis."""
        if axes is None:
            axes = list(AXES)
        return {
            axis: self.peers(
                anchor, axis=axis, geo_level=geo_level,
                top_k=top_k, restrict_state=restrict_state,
            )
            for axis in axes
        }
