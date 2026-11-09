from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ============================================================
# Paths
# ============================================================

ROOT = (
    Path.home()
    / "scratch"
    / "chemscreen_data"
)

PREDICTIONS = (
    ROOT
    / "predictions"
    / "candidate_predictions.jsonl"
)

TRAINING = (
    ROOT
    / "training"
    / "phase_training.jsonl"
)

OUTPUT_DIR = (
    ROOT
    / "predictions"
    / "pco_selection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SHORTLIST_JSON = (
    OUTPUT_DIR
    / "pco_shortlist.json"
)

SHORTLIST_JSONL = (
    OUTPUT_DIR
    / "pco_shortlist.jsonl"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "pco_selection_summary.json"
)


# ============================================================
# Manual-mode defaults
#
# These are only used when --auto-thresholds is NOT supplied.
# ============================================================

DEFAULT_MAX_EFA = 7.009303

DEFAULT_MIN_ROCK = 0.40

DEFAULT_MIN_SPINEL = 0.40


# ============================================================
# Automatic-search defaults
#
# Search order:
#
#     phase >= 0.50
#         EFA p10 -> p90
#
#     phase >= 0.45
#         EFA p10 -> p90
#
#     phase >= 0.40
#         EFA p10 -> p90
#
# The selector prioritizes the strictest configured phase
# threshold first. Within that phase threshold, it chooses
# the lowest EFA percentile that yields at least --top unique
# candidate compositions.
#
# p100 is intentionally excluded by default because the EFA
# distribution contains extreme upper-tail outliers.
# ============================================================

DEFAULT_PHASE_THRESHOLDS = (
    0.50,
    0.45,
    0.40,
)

DEFAULT_EFA_PERCENTILE_START = 10

DEFAULT_EFA_PERCENTILE_STOP = 90

DEFAULT_EFA_PERCENTILE_STEP = 1


# ============================================================
# IO helpers
# ============================================================

def read_jsonl(
    path,
):
    with path.open() as f:

        for line in f:

            if line.strip():

                yield json.loads(
                    line
                )


def write_jsonl(
    handle,
    obj,
):
    handle.write(
        json.dumps(
            obj,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )


# ============================================================
# Numeric helpers
# ============================================================

def finite(
    value,
):
    return (
        isinstance(
            value,
            (int, float),
        )
        and not isinstance(
            value,
            bool,
        )
        and math.isfinite(
            value
        )
    )


# ============================================================
# Evidence helpers
# ============================================================

def get_efa(
    record,
):
    dft = (
        record.get(
            "dft_evidence"
        )
        or {}
    )

    value = dft.get(
        "efa"
    )

    if finite(
        value
    ):

        return float(
            value
        )

    return None


def get_phase_values(
    record,
):
    rock = record.get(
        "best_rock_salt_score"
    )

    spinel = record.get(
        "best_spinel_score"
    )

    dual = record.get(
        "best_dual_phase_score"
    )

    robust = record.get(
        "route_robust_dual_score"
    )

    if not finite(
        rock
    ):
        rock = None

    if not finite(
        spinel
    ):
        spinel = None

    if not finite(
        dual
    ):
        dual = None

    if not finite(
        robust
    ):
        robust = None

    return (
        rock,
        spinel,
        dual,
        robust,
    )


def get_literature_counts(
    record,
):
    literature = (
        record.get(
            "literature_evidence"
        )
        or {}
    )

    reports = literature.get(
        "report_count",
        0,
    )

    successful = literature.get(
        "successful_count",
        0,
    )

    if not finite(
        reports
    ):
        reports = 0.0

    if not finite(
        successful
    ):
        successful = 0.0

    return (
        float(
            reports
        ),
        float(
            successful
        ),
    )


# ============================================================
# Candidate identity
# ============================================================

def selection_identity(
    record,
):
    composition = record.get(
        "composition_key"
    )

    if composition:

        return (
            "composition",
            str(
                composition
            ),
        )

    return (
        "material",
        str(
            record.get(
                "material_auid"
            )
        ),
    )


def unique_composition_count(
    records,
):
    return len(
        {
            selection_identity(
                record
            )

            for record
            in records
        }
    )


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Select final PCO-like candidates from "
        "phase-outcome predictions."
    )
)


parser.add_argument(
    "--top",
    type=int,
    default=20,
    help=(
        "Number of unique candidate compositions "
        "to select."
    ),
)


parser.add_argument(
    "--include-tested",
    action="store_true",
    help=(
        "Allow materials or normalized compositions "
        "already represented in OSES."
    ),
)


parser.add_argument(
    "--allow-missing-efa",
    action="store_true",
    help=(
        "Allow candidates without EFA in manual mode. "
        "Automatic threshold search requires EFA."
    ),
)


parser.add_argument(
    "--allow-out-of-domain",
    action="store_true",
    help=(
        "Allow candidates containing cations not seen "
        "in the OSES training element domain."
    ),
)


parser.add_argument(
    "--allow-duplicate-compositions",
    action="store_true",
    help=(
        "Allow multiple material records with the same "
        "normalized cation composition."
    ),
)


parser.add_argument(
    "--route",
    choices=[
        "any",
        "air",
        "ar",
    ],
    default="any",
    help=(
        "Optionally require the candidate's best predicted "
        "route to be Air or Ar."
    ),
)


# ============================================================
# Automatic threshold search
# ============================================================

parser.add_argument(
    "--auto-thresholds",
    action="store_true",
    help=(
        "Automatically choose the strictest configured "
        "phase threshold and strictest feasible EFA "
        "percentile that yield at least --top unique "
        "candidate compositions."
    ),
)


parser.add_argument(
    "--phase-thresholds",
    type=str,
    default="0.50,0.45,0.40",
    help=(
        "Comma-separated phase thresholds searched in order "
        "when --auto-thresholds is enabled. "
        "The same threshold is applied to rock salt "
        "and spinel."
    ),
)


parser.add_argument(
    "--efa-percentile-start",
    type=int,
    default=
        DEFAULT_EFA_PERCENTILE_START,
    help=(
        "Starting EFA percentile for automatic search."
    ),
)


parser.add_argument(
    "--efa-percentile-stop",
    type=int,
    default=
        DEFAULT_EFA_PERCENTILE_STOP,
    help=(
        "Final EFA percentile for automatic search."
    ),
)


parser.add_argument(
    "--efa-percentile-step",
    type=int,
    default=
        DEFAULT_EFA_PERCENTILE_STEP,
    help=(
        "EFA percentile increment for automatic search."
    ),
)


# ============================================================
# Manual thresholds
# ============================================================

parser.add_argument(
    "--max-efa",
    type=float,
    default=DEFAULT_MAX_EFA,
    help=(
        "Maximum EFA in manual mode. "
        f"Default: {DEFAULT_MAX_EFA:.6f}."
    ),
)


parser.add_argument(
    "--min-rock",
    type=float,
    default=DEFAULT_MIN_ROCK,
    help=(
        "Minimum rock-salt score in manual mode. "
        f"Default: {DEFAULT_MIN_ROCK:.2f}."
    ),
)


parser.add_argument(
    "--min-spinel",
    type=float,
    default=DEFAULT_MIN_SPINEL,
    help=(
        "Minimum spinel score in manual mode. "
        f"Default: {DEFAULT_MIN_SPINEL:.2f}."
    ),
)


# ============================================================
# Additional fixed filters
# ============================================================

parser.add_argument(
    "--min-dual",
    type=float,
    default=None,
    help=(
        "Optional minimum dual-phase product score."
    ),
)


parser.add_argument(
    "--min-robust-dual",
    type=float,
    default=None,
    help=(
        "Optional minimum route-robust dual score."
    ),
)


parser.add_argument(
    "--min-literature-reports",
    type=float,
    default=None,
    help=(
        "Optional minimum literature report count."
    ),
)


args = parser.parse_args()


# ============================================================
# Argument validation
# ============================================================

if args.top <= 0:

    raise RuntimeError(
        "--top must be greater than zero."
    )


if args.efa_percentile_step <= 0:

    raise RuntimeError(
        "--efa-percentile-step must be positive."
    )


if not (
    0
    <= args.efa_percentile_start
    <= 100
):

    raise RuntimeError(
        "--efa-percentile-start must be between 0 and 100."
    )


if not (
    0
    <= args.efa_percentile_stop
    <= 100
):

    raise RuntimeError(
        "--efa-percentile-stop must be between 0 and 100."
    )


if (
    args.efa_percentile_start
    > args.efa_percentile_stop
):

    raise RuntimeError(
        "--efa-percentile-start cannot exceed "
        "--efa-percentile-stop."
    )


if (
    args.auto_thresholds
    and args.allow_missing_efa
):

    raise RuntimeError(
        "--auto-thresholds requires EFA values and cannot "
        "be combined with --allow-missing-efa."
    )


# ============================================================
# Parse phase threshold search grid
# ============================================================

try:

    phase_thresholds = [
        float(
            value.strip()
        )

        for value
        in args.phase_thresholds.split(",")

        if value.strip()
    ]

except ValueError as exc:

    raise RuntimeError(
        "Could not parse --phase-thresholds."
    ) from exc


if not phase_thresholds:

    raise RuntimeError(
        "--phase-thresholds contains no values."
    )


for threshold in phase_thresholds:

    if not (
        0.0
        <= threshold
        <= 1.0
    ):

        raise RuntimeError(
            "Phase thresholds must lie between 0 and 1."
        )


# ============================================================
# Validate inputs
# ============================================================

for path in [
    PREDICTIONS,
    TRAINING,
]:

    if not path.exists():

        raise RuntimeError(
            f"Missing required file: {path}"
        )


# ============================================================
# Load predictions
# ============================================================

records = list(
    read_jsonl(
        PREDICTIONS
    )
)


if not records:

    raise RuntimeError(
        "Prediction file contains no candidates."
    )


print(
    "Input prediction rows:",
    len(
        records
    ),
)


# ============================================================
# Build normalized OSES composition index
#
# This is distinct from material-level exclusion.
#
# Multiple LOOP material AUIDs may represent the same
# normalized cation composition. For prospective validation,
# the composition itself should be novel relative to OSES.
# ============================================================

oses_compositions = set()


for row in read_jsonl(
    TRAINING
):

    composition = row.get(
        "composition_key"
    )

    if composition:

        oses_compositions.add(
            str(
                composition
            )
        )


print(
    "OSES composition groups:",
    len(
        oses_compositions
    ),
)


# ============================================================
# Non-adaptive eligibility
#
# These criteria do NOT change during automatic threshold
# search.
#
# The adaptive dimensions are only:
#
#     minimum rock-salt score
#     minimum spinel score
#     maximum EFA
# ============================================================

def passes_nonadaptive_filters(
    record,
):

    # --------------------------------------------------------
    # Previously tested material
    # --------------------------------------------------------

    if (
        record.get(
            "has_oses_experiment"
        )
        and not args.include_tested
    ):

        return False


    # --------------------------------------------------------
    # Previously tested normalized composition
    # --------------------------------------------------------

    composition = record.get(
        "composition_key"
    )


    if (
        not args.include_tested
        and composition
        and str(
            composition
        )
        in oses_compositions
    ):

        return False


    # --------------------------------------------------------
    # Applicability domain
    # --------------------------------------------------------

    in_domain = record.get(
        "in_training_element_domain"
    )


    if not args.allow_out_of_domain:

        if in_domain is not True:

            return False


    # --------------------------------------------------------
    # Best synthesis route
    # --------------------------------------------------------

    if (
        args.route != "any"
        and record.get(
            "best_route"
        )
        != args.route
    ):

        return False


    # --------------------------------------------------------
    # Phase predictions must exist
    # --------------------------------------------------------

    (
        rock,
        spinel,
        dual,
        robust,
    ) = get_phase_values(
        record
    )


    if (
        rock is None
        or spinel is None
        or dual is None
    ):

        return False


    # --------------------------------------------------------
    # Optional dual score
    # --------------------------------------------------------

    if (
        args.min_dual
        is not None
        and dual
        < args.min_dual
    ):

        return False


    # --------------------------------------------------------
    # Optional robust dual score
    # --------------------------------------------------------

    if (
        args.min_robust_dual
        is not None
        and (
            robust is None
            or robust
            < args.min_robust_dual
        )
    ):

        return False


    # --------------------------------------------------------
    # Optional literature requirement
    # --------------------------------------------------------

    (
        literature_reports,
        _,
    ) = get_literature_counts(
        record
    )


    if (
        args.min_literature_reports
        is not None
        and literature_reports
        < args.min_literature_reports
    ):

        return False


    return True


# ============================================================
# Automatic threshold search
# ============================================================

active_max_efa = (
    args.max_efa
)

active_min_rock = (
    args.min_rock
)

active_min_spinel = (
    args.min_spinel
)

auto_selection = None

auto_search_trace = []


if args.auto_thresholds:

    # --------------------------------------------------------
    # EFA reference population:
    #
    # candidates that are:
    #
    #   not previously tested
    #   compositionally novel relative to OSES
    #   inside the selected applicability domain
    #   valid under any fixed route/dual/literature filters
    #   EFA-covered
    #
    # This prevents OSES-overlap compositions from changing
    # the dynamic percentile thresholds.
    # --------------------------------------------------------

    efa_reference_values = []


    for record in records:

        if not passes_nonadaptive_filters(
            record
        ):

            continue


        efa = get_efa(
            record
        )


        if efa is not None:

            efa_reference_values.append(
                efa
            )


    if not efa_reference_values:

        raise RuntimeError(
            "No EFA-covered candidates remain after "
            "non-adaptive filtering."
        )


    efa_reference_array = np.asarray(
        efa_reference_values,
        dtype=float,
    )


    found = False


    # --------------------------------------------------------
    # Search order:
    #
    # 1. phase threshold order supplied by user
    # 2. EFA percentile low -> high
    #
    # This prioritizes stronger predicted phase behavior.
    # Within a given phase threshold, it finds the lowest EFA
    # percentile capable of filling the desired unique-
    # composition batch.
    # --------------------------------------------------------

    for phase_threshold in (
        phase_thresholds
    ):

        for percentile in range(
            args.efa_percentile_start,
            args.efa_percentile_stop
            + 1,
            args.efa_percentile_step,
        ):

            efa_cutoff = float(
                np.percentile(
                    efa_reference_array,
                    percentile,
                )
            )


            qualifying = []


            for record in records:

                if not passes_nonadaptive_filters(
                    record
                ):

                    continue


                efa = get_efa(
                    record
                )


                (
                    rock,
                    spinel,
                    _,
                    _,
                ) = get_phase_values(
                    record
                )


                if (
                    efa is None
                    or rock is None
                    or spinel is None
                ):

                    continue


                if efa > efa_cutoff:

                    continue


                if rock < phase_threshold:

                    continue


                if spinel < phase_threshold:

                    continue


                qualifying.append(
                    record
                )


            unique_count = (
                unique_composition_count(
                    qualifying
                )
            )


            search_result = {
                "phase_threshold":
                    phase_threshold,

                "efa_percentile":
                    percentile,

                "efa_cutoff":
                    efa_cutoff,

                "qualifying_records":
                    len(
                        qualifying
                    ),

                "qualifying_unique_compositions":
                    unique_count,
            }


            auto_search_trace.append(
                search_result
            )


            if (
                unique_count
                >= args.top
            ):

                active_min_rock = (
                    phase_threshold
                )

                active_min_spinel = (
                    phase_threshold
                )

                active_max_efa = (
                    efa_cutoff
                )


                auto_selection = {
                    **search_result,

                    "requested_unique_compositions":
                        args.top,

                    "efa_reference_population":
                        len(
                            efa_reference_values
                        ),

                    "oses_composition_groups_excluded":
                        len(
                            oses_compositions
                        ),

                    "search_priority":
                        (
                            "phase_threshold_order_first; "
                            "efa_percentile_ascending_second"
                        ),
                }


                found = True

                break


        if found:

            break


    if not found:

        maximum_unique = 0

        best_attempt = None


        for result in (
            auto_search_trace
        ):

            count = result[
                "qualifying_unique_compositions"
            ]


            if count > maximum_unique:

                maximum_unique = (
                    count
                )

                best_attempt = (
                    result
                )


        raise RuntimeError(
            "Automatic threshold search could not find "
            f"{args.top} unique candidate compositions "
            "within the configured search grid. "
            f"Best result: {best_attempt!r}"
        )


# ============================================================
# Report automatic operating point
# ============================================================

if auto_selection:

    print()

    print(
        "=== AUTOMATIC THRESHOLD SELECTION ==="
    )

    print(
        "Requested unique candidates:",
        args.top,
    )

    print(
        "Phase threshold:",
        auto_selection[
            "phase_threshold"
        ],
    )

    print(
        "EFA percentile:",
        f"p{auto_selection['efa_percentile']}",
    )

    print(
        "EFA cutoff:",
        auto_selection[
            "efa_cutoff"
        ],
    )

    print(
        "EFA reference population:",
        auto_selection[
            "efa_reference_population"
        ],
    )

    print(
        "Qualifying material records:",
        auto_selection[
            "qualifying_records"
        ],
    )

    print(
        "Qualifying unique compositions:",
        auto_selection[
            "qualifying_unique_compositions"
        ],
    )


# ============================================================
# Final eligibility filtering
#
# This pass records explicit exclusion counts using the active
# thresholds selected above.
# ============================================================

eligible = []


filter_counts = {
    "input":
        len(
            records
        ),

    "already_oses_tested_material":
        0,

    "already_oses_composition":
        0,

    "out_of_training_element_domain":
        0,

    "missing_domain_metadata":
        0,

    "missing_efa":
        0,

    "efa_above_threshold":
        0,

    "wrong_best_route":
        0,

    "missing_phase_prediction":
        0,

    "rock_below_threshold":
        0,

    "spinel_below_threshold":
        0,

    "dual_below_threshold":
        0,

    "robust_dual_below_threshold":
        0,

    "literature_below_threshold":
        0,
}


for record in records:

    # --------------------------------------------------------
    # Previously tested material
    # --------------------------------------------------------

    if (
        record.get(
            "has_oses_experiment"
        )
        and not args.include_tested
    ):

        filter_counts[
            "already_oses_tested_material"
        ] += 1

        continue


    # --------------------------------------------------------
    # Previously tested normalized composition
    # --------------------------------------------------------

    composition = record.get(
        "composition_key"
    )


    if (
        not args.include_tested
        and composition
        and str(
            composition
        )
        in oses_compositions
    ):

        filter_counts[
            "already_oses_composition"
        ] += 1

        continue


    # --------------------------------------------------------
    # Applicability domain
    # --------------------------------------------------------

    in_domain = record.get(
        "in_training_element_domain"
    )


    if (
        in_domain is None
        and not args.allow_out_of_domain
    ):

        filter_counts[
            "missing_domain_metadata"
        ] += 1

        continue


    if (
        in_domain is False
        and not args.allow_out_of_domain
    ):

        filter_counts[
            "out_of_training_element_domain"
        ] += 1

        continue


    # --------------------------------------------------------
    # EFA
    # --------------------------------------------------------

    efa = get_efa(
        record
    )


    if (
        efa is None
        and not args.allow_missing_efa
    ):

        filter_counts[
            "missing_efa"
        ] += 1

        continue


    if (
        active_max_efa
        is not None
        and efa is not None
        and efa
        > active_max_efa
    ):

        filter_counts[
            "efa_above_threshold"
        ] += 1

        continue


    # --------------------------------------------------------
    # Route
    # --------------------------------------------------------

    best_route = record.get(
        "best_route"
    )


    if (
        args.route != "any"
        and best_route
        != args.route
    ):

        filter_counts[
            "wrong_best_route"
        ] += 1

        continue


    # --------------------------------------------------------
    # Phase prediction
    # --------------------------------------------------------

    (
        rock,
        spinel,
        dual,
        robust,
    ) = get_phase_values(
        record
    )


    if (
        rock is None
        or spinel is None
        or dual is None
    ):

        filter_counts[
            "missing_phase_prediction"
        ] += 1

        continue


    if (
        active_min_rock
        is not None
        and rock
        < active_min_rock
    ):

        filter_counts[
            "rock_below_threshold"
        ] += 1

        continue


    if (
        active_min_spinel
        is not None
        and spinel
        < active_min_spinel
    ):

        filter_counts[
            "spinel_below_threshold"
        ] += 1

        continue


    if (
        args.min_dual
        is not None
        and dual
        < args.min_dual
    ):

        filter_counts[
            "dual_below_threshold"
        ] += 1

        continue


    if (
        args.min_robust_dual
        is not None
        and (
            robust is None
            or robust
            < args.min_robust_dual
        )
    ):

        filter_counts[
            "robust_dual_below_threshold"
        ] += 1

        continue


    # --------------------------------------------------------
    # Literature
    # --------------------------------------------------------

    (
        literature_reports,
        literature_successful,
    ) = get_literature_counts(
        record
    )


    if (
        args.min_literature_reports
        is not None
        and literature_reports
        < args.min_literature_reports
    ):

        filter_counts[
            "literature_below_threshold"
        ] += 1

        continue


    # --------------------------------------------------------
    # Retain selection metadata
    # --------------------------------------------------------

    retained = dict(
        record
    )


    retained[
        "_selection"
    ] = {
        "efa":
            efa,

        "rock_salt_score":
            rock,

        "spinel_score":
            spinel,

        "dual_phase_score":
            dual,

        "route_robust_dual_score":
            robust,

        "literature_report_count":
            literature_reports,

        "literature_successful_count":
            literature_successful,
    }


    eligible.append(
        retained
    )


filter_counts[
    "eligible_before_composition_dedup"
] = len(
    eligible
)


# ============================================================
# Ranking
#
# Thresholds define eligibility.
#
# Ranking among eligible candidates:
#
#     1. highest dual-phase heuristic
#     2. lower EFA
#     3. successful literature support
#     4. literature report count
#     5. route robustness
#
# No weighted aggregate score is used.
# ============================================================

def ranking_key(
    record,
):
    selection = (
        record[
            "_selection"
        ]
    )

    efa = selection[
        "efa"
    ]

    robust = selection[
        "route_robust_dual_score"
    ]

    return (
        -selection[
            "dual_phase_score"
        ],

        (
            efa
            if efa is not None
            else float(
                "inf"
            )
        ),

        -selection[
            "literature_successful_count"
        ],

        -selection[
            "literature_report_count"
        ],

        -(
            robust
            if robust is not None
            else 0.0
        ),

        str(
            record.get(
                "material_auid"
            )
        ),
    )


eligible.sort(
    key=
        ranking_key
)


# ============================================================
# Composition deduplication
# ============================================================

if args.allow_duplicate_compositions:

    deduplicated = (
        eligible
    )

    duplicate_compositions_removed = 0


else:

    deduplicated = []

    seen_identities = set()

    duplicate_compositions_removed = 0


    for record in eligible:

        identity = selection_identity(
            record
        )


        if (
            identity
            in seen_identities
        ):

            duplicate_compositions_removed += 1

            continue


        seen_identities.add(
            identity
        )


        deduplicated.append(
            record
        )


filter_counts[
    "duplicate_compositions_removed"
] = (
    duplicate_compositions_removed
)


filter_counts[
    "eligible_after_composition_dedup"
] = len(
    deduplicated
)


# ============================================================
# Final shortlist
# ============================================================

selected = deduplicated[
    :args.top
]


if (
    len(
        selected
    )
    < args.top
):

    print(
        "WARNING:",
        f"requested {args.top} candidates, "
        f"but only {len(selected)} passed selection."
    )


# ============================================================
# Build shortlist records
# ============================================================

shortlist = []


for rank, record in enumerate(
    selected,
    start=1,
):

    selection = (
        record[
            "_selection"
        ]
    )

    dft = (
        record.get(
            "dft_evidence"
        )
        or {}
    )

    literature = (
        record.get(
            "literature_evidence"
        )
        or {}
    )

    route_predictions = (
        record.get(
            "route_predictions"
        )
        or {}
    )


    reasons = []


    if not record.get(
        "has_oses_experiment"
    ):

        reasons.append(
            "not_previously_oses_tested_material"
        )


    composition = record.get(
        "composition_key"
    )


    if (
        not composition
        or str(
            composition
        )
        not in oses_compositions
    ):

        reasons.append(
            "composition_novel_relative_to_oses"
        )


    if (
        record.get(
            "in_training_element_domain"
        )
        is True
    ):

        reasons.append(
            "inside_training_element_domain"
        )

    else:

        reasons.append(
            "outside_training_element_domain_allowed"
        )


    if (
        selection[
            "efa"
        ]
        is not None
    ):

        reasons.append(
            "efa_within_selected_threshold"
        )


    reasons.extend(
        [
            "rock_salt_score_above_selected_threshold",
            "spinel_score_above_selected_threshold",
            "dual_rock_salt_spinel_prediction",
        ]
    )


    if (
        selection[
            "literature_report_count"
        ] > 0
    ):

        reasons.append(
            "literature_support_available"
        )


    if (
        selection[
            "route_robust_dual_score"
        ]
        is not None
    ):

        reasons.append(
            "air_ar_route_comparison_available"
        )


    shortlist_record = {
        "selection_rank":
            rank,

        "material_auid":
            record.get(
                "material_auid"
            ),

        "composition_key":
            record.get(
                "composition_key"
            ),

        "model_cation_elements":
            record.get(
                "model_cation_elements"
            )
            or {},

        "full_material_elements":
            record.get(
                "full_material_elements"
            )
            or {},

        "in_training_element_domain":
            record.get(
                "in_training_element_domain"
            ),

        "unseen_elements":
            record.get(
                "unseen_elements"
            )
            or [],

        "already_in_oses_composition_space":
            (
                str(
                    composition
                )
                in oses_compositions
                if composition
                else False
            ),

        "best_synthesis_route":
            record.get(
                "best_route"
            ),

        "phase_prediction": {
            "rock_salt_score":
                selection[
                    "rock_salt_score"
                ],

            "spinel_score":
                selection[
                    "spinel_score"
                ],

            "dual_phase_score":
                selection[
                    "dual_phase_score"
                ],

            "route_robust_dual_score":
                selection[
                    "route_robust_dual_score"
                ],

            "route_dual_score_difference":
                record.get(
                    "route_dual_score_difference"
                ),
        },

        "route_predictions":
            route_predictions,

        "dft_evidence": {
            "efa":
                dft.get(
                    "efa"
                ),

            "efa_source":
                dft.get(
                    "efa_source"
                ),

            "formation_energy_mean":
                dft.get(
                    "formation_energy_mean"
                ),

            "xint_hf_mean":
                dft.get(
                    "xint_hf_mean"
                ),

            "crystal_system_mode":
                dft.get(
                    "crystal_system_mode"
                ),

            "spacegroup_mode":
                dft.get(
                    "spacegroup_mode"
                ),

            "exact_record_count":
                dft.get(
                    "exact_record_count"
                ),

            "equivalent_record_count":
                dft.get(
                    "equivalent_record_count"
                ),
        },

        "literature_evidence": {
            "report_count":
                literature.get(
                    "report_count",
                    0,
                ),

            "successful_count":
                literature.get(
                    "successful_count",
                    0,
                ),

            "source":
                literature.get(
                    "source"
                ),
        },

        "selection_thresholds": {
            "mode":
                (
                    "automatic"
                    if args.auto_thresholds
                    else "manual"
                ),

            "max_efa":
                active_max_efa,

            "min_rock_salt":
                active_min_rock,

            "min_spinel":
                active_min_spinel,

            "min_dual":
                args.min_dual,

            "min_robust_dual":
                args.min_robust_dual,

            "efa_percentile":
                (
                    auto_selection[
                        "efa_percentile"
                    ]
                    if auto_selection
                    else None
                ),
        },

        "selection_reasons":
            reasons,

        "selection_policy": {
            "oses_overlap":
                (
                    "excluded"
                    if not args.include_tested
                    else "allowed"
                ),

            "applicability_domain":
                (
                    "strict"
                    if not args.allow_out_of_domain
                    else "out_of_domain_allowed"
                ),

            "threshold_selection":
                (
                    "automatic"
                    if args.auto_thresholds
                    else "manual"
                ),

            "ranking_primary":
                "highest dual-phase score",

            "ranking_secondary":
                "lower EFA",

            "ranking_tertiary":
                (
                    "successful literature evidence, "
                    "then literature report count"
                ),

            "additional_tiebreak":
                "route robustness",

            "weighted_score_used":
                False,
        },
    }


    shortlist.append(
        shortlist_record
    )


# ============================================================
# Write shortlist
# ============================================================

with SHORTLIST_JSON.open(
    "w"
) as out:

    json.dump(
        shortlist,
        out,
        indent=2,
        ensure_ascii=False,
    )


with SHORTLIST_JSONL.open(
    "w"
) as out:

    for record in shortlist:

        write_jsonl(
            out,
            record,
        )


# ============================================================
# Selection summary
# ============================================================

air_selected = sum(
    1

    for record in shortlist

    if record[
        "best_synthesis_route"
    ] == "air"
)


ar_selected = sum(
    1

    for record in shortlist

    if record[
        "best_synthesis_route"
    ] == "ar"
)


with_literature = sum(
    1

    for record in shortlist

    if record[
        "literature_evidence"
    ][
        "report_count"
    ] > 0
)


with_efa = sum(
    1

    for record in shortlist

    if record[
        "dft_evidence"
    ][
        "efa"
    ]
    is not None
)


oses_overlap_selected = sum(
    1

    for record in shortlist

    if record[
        "already_in_oses_composition_space"
    ]
)


summary = {
    "requested_top":
        args.top,

    "selected":
        len(
            shortlist
        ),

    "selection_scope":
        "PCO-like candidate selection",

    "scientific_objective":
        (
            "Identify compositionally novel, relatively "
            "low-EFA candidates within the OSES experimental "
            "applicability domain that are predicted to "
            "support both rock-salt and spinel formation."
        ),

    "oses_composition_count":
        len(
            oses_compositions
        ),

    "threshold_mode":
        (
            "automatic"
            if args.auto_thresholds
            else "manual"
        ),

    "active_thresholds": {
        "max_efa":
            active_max_efa,

        "min_rock_salt":
            active_min_rock,

        "min_spinel":
            active_min_spinel,

        "min_dual":
            args.min_dual,

        "min_robust_dual":
            args.min_robust_dual,
    },

    "automatic_threshold_selection":
        auto_selection,

    "automatic_search_configuration": {
        "phase_thresholds":
            phase_thresholds,

        "efa_percentile_start":
            args.efa_percentile_start,

        "efa_percentile_stop":
            args.efa_percentile_stop,

        "efa_percentile_step":
            args.efa_percentile_step,

        "search_priority":
            (
                "phase threshold order first; "
                "EFA percentile ascending second"
            ),
    },

    "automatic_search_trace":
        (
            auto_search_trace
            if args.auto_thresholds
            else []
        ),

    "selection_policy": {
        "weighted_score_used":
            False,

        "oses_material_overlap_allowed":
            args.include_tested,

        "oses_composition_overlap_allowed":
            args.include_tested,

        "novel_composition_required":
            not args.include_tested,

        "applicability_domain":
            (
                "strict"
                if not args.allow_out_of_domain
                else "out_of_domain_allowed"
            ),

        "unique_composition_required":
            not args.allow_duplicate_compositions,

        "ranking_primary":
            "dual_phase_score_desc",

        "ranking_secondary":
            "efa_asc",

        "ranking_tertiary":
            "literature_success_desc",

        "ranking_quaternary":
            "literature_report_count_desc",

        "additional_tiebreak":
            "route_robust_dual_score_desc",
    },

    "active_filters": {
        "include_tested":
            args.include_tested,

        "allow_missing_efa":
            args.allow_missing_efa,

        "allow_out_of_domain":
            args.allow_out_of_domain,

        "allow_duplicate_compositions":
            args.allow_duplicate_compositions,

        "route":
            args.route,

        "min_literature_reports":
            args.min_literature_reports,
    },

    "filter_counts":
        filter_counts,

    "selected_evidence": {
        "with_efa":
            with_efa,

        "with_literature":
            with_literature,

        "best_route_air":
            air_selected,

        "best_route_ar":
            ar_selected,

        "oses_composition_overlap":
            oses_overlap_selected,

        "compositionally_novel":
            (
                len(
                    shortlist
                )
                - oses_overlap_selected
            ),
    },

    "outputs": {
        "shortlist_json":
            str(
                SHORTLIST_JSON
            ),

        "shortlist_jsonl":
            str(
                SHORTLIST_JSONL
            ),
    },
}


with SUMMARY_JSON.open(
    "w"
) as out:

    json.dump(
        summary,
        out,
        indent=2,
    )


# ============================================================
# Console report
# ============================================================

print()

print(
    "=== PCO SELECTION COMPLETE ==="
)

print(
    "Input predictions:",
    len(
        records
    ),
)

print(
    "OSES composition groups:",
    len(
        oses_compositions
    ),
)

print(
    "Threshold mode:",
    (
        "automatic"
        if args.auto_thresholds
        else "manual"
    ),
)

print(
    "Applicability domain:",
    (
        "strict"
        if not args.allow_out_of_domain
        else "out-of-domain allowed"
    ),
)

print(
    "Prior OSES compositions:",
    (
        "excluded"
        if not args.include_tested
        else "allowed"
    ),
)

print()

print(
    "Active thresholds:"
)

print(
    "  max EFA:",
    active_max_efa,
)

print(
    "  min rock:",
    active_min_rock,
)

print(
    "  min spinel:",
    active_min_spinel,
)

print()

print(
    "Previously tested materials removed:",
    filter_counts[
        "already_oses_tested_material"
    ],
)

print(
    "Previously tested compositions removed:",
    filter_counts[
        "already_oses_composition"
    ],
)

print(
    "Out-of-domain removed:",
    filter_counts[
        "out_of_training_element_domain"
    ],
)

print(
    "Missing domain metadata:",
    filter_counts[
        "missing_domain_metadata"
    ],
)

print(
    "Eligible before composition dedup:",
    filter_counts[
        "eligible_before_composition_dedup"
    ],
)

print(
    "Duplicate compositions removed:",
    duplicate_compositions_removed,
)

print(
    "Eligible after composition dedup:",
    filter_counts[
        "eligible_after_composition_dedup"
    ],
)

print(
    "Selected:",
    len(
        shortlist
    ),
)

print(
    "Selected with EFA:",
    with_efa,
)

print(
    "Selected with literature:",
    with_literature,
)

print(
    "Compositionally novel:",
    (
        len(
            shortlist
        )
        - oses_overlap_selected
    ),
)

print(
    "Best route Air:",
    air_selected,
)

print(
    "Best route Ar:",
    ar_selected,
)

print(
    "Shortlist:",
    SHORTLIST_JSON,
)

print()


for record in shortlist:

    phase = (
        record[
            "phase_prediction"
        ]
    )

    dft = (
        record[
            "dft_evidence"
        ]
    )

    literature = (
        record[
            "literature_evidence"
        ]
    )


    print(
        record[
            "selection_rank"
        ],

        record[
            "material_auid"
        ],

        "Composition=",
        record[
            "composition_key"
        ],

        "Route=",
        record[
            "best_synthesis_route"
        ],

        "RS=",
        round(
            phase[
                "rock_salt_score"
            ],
            4,
        ),

        "Spinel=",
        round(
            phase[
                "spinel_score"
            ],
            4,
        ),

        "Dual=",
        round(
            phase[
                "dual_phase_score"
            ],
            4,
        ),

        "EFA=",
        dft[
            "efa"
        ],

        "Lit=",
        literature[
            "report_count"
        ],

        "OSES_overlap=",
        record[
            "already_in_oses_composition_space"
        ],
    )