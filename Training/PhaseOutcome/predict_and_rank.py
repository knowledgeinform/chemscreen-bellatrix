from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib


# ============================================================
# Paths
# ============================================================

ROOT = (
    Path.home()
    / "scratch"
    / "chemscreen_data"
)

CANDIDATES = (
    ROOT
    / "training"
    / "candidate_features.jsonl"
)

OUTPUT_DIR = (
    ROOT
    / "predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR = (
    Path(__file__)
    .resolve()
    .parent
    / "models"
)

VECTORIZER = (
    MODEL_DIR
    / "feature_vectorizer.joblib"
)

ROCK_MODEL = (
    MODEL_DIR
    / "rock_salt_rf.joblib"
)

SPINEL_MODEL = (
    MODEL_DIR
    / "spinel_rf.joblib"
)


# ============================================================
# Experimentally supported OSES synthesis routes
# ============================================================

ROUTES = {
    "air": {
        "synth_route":
            "air",

        "synth_atmosphere":
            "air",

        "synth_max_temp_c":
            1100.0,

        "synth_total_hold_time_hours":
            5.0,

        "synth_quenched":
            1.0,

        "synth_quench_medium":
            "air",

        "synth_step_count":
            1,

        "synth_has_step__other":
            1.0,
    },

    "ar": {
        "synth_route":
            "ar",

        "synth_atmosphere":
            "ar",

        "synth_max_temp_c":
            1100.0,

        "synth_total_hold_time_hours":
            5.0,

        "synth_quenched":
            1.0,

        "synth_quench_medium":
            "rapid_hot_zone_removal",

        "synth_step_count":
            1,

        "synth_has_step__other":
            1.0,
    },
}


# ============================================================
# Helpers
# ============================================================

def read_jsonl(path):
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


def finite(value):
    return (
        isinstance(
            value,
            (int, float),
        )
        and math.isfinite(
            value
        )
    )


def positive_probability(
    model,
    X,
):
    classes = list(
        model.classes_
    )

    if 1 not in classes:

        return [
            0.0
            for _ in range(
                X.shape[0]
            )
        ]

    index = classes.index(
        1
    )

    return model.predict_proba(
        X
    )[:, index]


def first_finite(
    features,
    keys,
):
    for key in keys:

        value = features.get(
            key
        )

        if finite(
            value
        ):

            return float(
                value
            )

    return None


def first_present(
    features,
    keys,
):
    for key in keys:

        value = features.get(
            key
        )

        if value not in (
            None,
            "",
        ):

            return value

    return None


# ============================================================
# DFT evidence
# ============================================================

def get_dft_evidence(
    features,
):
    exact_efa = first_finite(
        features,
        [
            "comp_efa_mean",
        ],
    )

    equivalent_efa = first_finite(
        features,
        [
            "equiv_comp_efa_mean",
        ],
    )

    if exact_efa is not None:

        efa = exact_efa
        efa_source = "exact_material"

    elif equivalent_efa is not None:

        efa = equivalent_efa
        efa_source = (
            "composition_equivalent"
        )

    else:

        efa = None
        efa_source = None


    formation_energy = first_finite(
        features,
        [
            "comp_loop_formation_energy_mean",
            "equiv_comp_loop_formation_energy_mean",
        ],
    )


    xint_hf = first_finite(
        features,
        [
            "comp_xint_hf_mean",
            "equiv_comp_xint_hf_mean",
        ],
    )


    crystal_system = first_present(
        features,
        [
            "comp_crystal_system_mode",
            "equiv_comp_crystal_system_mode",
        ],
    )


    spacegroup = first_present(
        features,
        [
            "comp_spacegroup_mode",
            "equiv_comp_spacegroup_mode",
        ],
    )


    exact_records = first_finite(
        features,
        [
            "comp_record_count",
        ],
    )


    equivalent_records = first_finite(
        features,
        [
            "equiv_comp_record_count",
        ],
    )


    return {
        "efa":
            efa,

        "efa_source":
            efa_source,

        "exact_efa_mean":
            exact_efa,

        "equivalent_efa_mean":
            equivalent_efa,

        "formation_energy_mean":
            formation_energy,

        "xint_hf_mean":
            xint_hf,

        "crystal_system_mode":
            crystal_system,

        "spacegroup_mode":
            spacegroup,

        "exact_record_count":
            exact_records,

        "equivalent_record_count":
            equivalent_records,
    }


# ============================================================
# Literature evidence
# ============================================================

def get_literature_evidence(
    features,
):
    exact_reports = first_finite(
        features,
        [
            "lit_report_count",
        ],
    )

    equivalent_reports = first_finite(
        features,
        [
            "equiv_lit_report_count",
        ],
    )


    exact_success = first_finite(
        features,
        [
            "lit_success_count",
        ],
    )

    equivalent_success = first_finite(
        features,
        [
            "equiv_lit_success_count",
        ],
    )


    if exact_reports is not None:

        reports = exact_reports
        source = "exact_material"

    elif equivalent_reports is not None:

        reports = equivalent_reports
        source = (
            "composition_equivalent"
        )

    else:

        reports = 0.0
        source = None


    if exact_success is not None:

        successful = exact_success

    elif equivalent_success is not None:

        successful = (
            equivalent_success
        )

    else:

        successful = 0.0


    return {
        "report_count":
            reports,

        "successful_count":
            successful,

        "source":
            source,
    }


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--top",
    type=int,
    default=20,
    help=(
        "Number of top candidate materials to write."
    ),
)


parser.add_argument(
    "--route",
    choices=[
        "air",
        "ar",
        "both",
    ],
    default="both",
    help=(
        "Evaluate the Air route, Ar route, "
        "or both experimentally supported OSES routes."
    ),
)


parser.add_argument(
    "--include-tested",
    action="store_true",
    help=(
        "Include materials already represented in OSES."
    ),
)


parser.add_argument(
    "--require-efa",
    action="store_true",
    help=(
        "Require an EFA value for candidate eligibility."
    ),
)


parser.add_argument(
    "--max-efa",
    type=float,
    default=None,
    help=(
        "Optional upper EFA threshold."
    ),
)


parser.add_argument(
    "--min-rock",
    type=float,
    default=None,
    help=(
        "Optional minimum predicted rock-salt probability."
    ),
)


parser.add_argument(
    "--min-spinel",
    type=float,
    default=None,
    help=(
        "Optional minimum predicted spinel probability."
    ),
)


parser.add_argument(
    "--min-dual",
    type=float,
    default=None,
    help=(
        "Optional minimum predicted dual-phase score."
    ),
)


args = parser.parse_args()


# ============================================================
# Validate artifacts
# ============================================================

for path in [
    CANDIDATES,
    VECTORIZER,
    ROCK_MODEL,
    SPINEL_MODEL,
]:

    if not path.exists():

        raise RuntimeError(
            f"Missing required artifact: {path}"
        )


# ============================================================
# Load models
# ============================================================

vectorizer = joblib.load(
    VECTORIZER
)

rock_model = joblib.load(
    ROCK_MODEL
)

spinel_model = joblib.load(
    SPINEL_MODEL
)


# ============================================================
# Routes to evaluate
# ============================================================

if args.route == "both":

    route_names = [
        "air",
        "ar",
    ]

else:

    route_names = [
        args.route
    ]


# ============================================================
# Load candidates
# ============================================================

candidates = []


for candidate in read_jsonl(
    CANDIDATES
):

    has_oses = bool(
        candidate.get(
            "has_oses_experiment"
        )
    )


    if (
        has_oses
        and not args.include_tested
    ):

        continue


    features = dict(
        candidate.get(
            "features"
        )
        or {}
    )


    dft_evidence = (
        get_dft_evidence(
            features
        )
    )


    literature_evidence = (
        get_literature_evidence(
            features
        )
    )


    efa = dft_evidence[
        "efa"
    ]


    if (
        args.require_efa
        and efa is None
    ):

        continue


    if (
        args.max_efa
        is not None
        and efa is not None
        and efa > args.max_efa
    ):

        continue


    candidates.append(
        {
            "material_auid":
                candidate.get(
                    "material_auid"
                ),

            "composition_key":
                candidate.get(
                    "composition_key"
                ),

            "model_cation_elements":
                candidate.get(
                    "model_cation_elements"
                )
                or {},

            "full_material_elements":
                candidate.get(
                    "full_material_elements"
                )
                or {},

            "in_training_element_domain":
                candidate.get(
                    "in_training_element_domain"
                ),

            "unseen_elements":
                candidate.get(
                    "unseen_elements"
                )
                or [],

            "has_oses_experiment":
                has_oses,

            "features":
                features,

            "dft_evidence":
                dft_evidence,

            "literature_evidence":
                literature_evidence,

            "route_predictions":
                {},
        }
    )


if not candidates:

    raise RuntimeError(
        "No candidates remain after eligibility filtering."
    )


print(
    "Eligible candidate materials:",
    len(
        candidates
    ),
)

print(
    "Routes evaluated:",
    route_names,
)


# ============================================================
# Build candidate-route prediction rows
# ============================================================

prediction_rows = []


for candidate_index, candidate in enumerate(
    candidates
):

    for route_name in route_names:

        model_features = dict(
            candidate[
                "features"
            ]
        )


        model_features.update(
            ROUTES[
                route_name
            ]
        )


        prediction_rows.append(
            {
                "candidate_index":
                    candidate_index,

                "route":
                    route_name,

                "features":
                    model_features,
            }
        )


# ============================================================
# Vectorize
# ============================================================

X = vectorizer.transform(
    [
        row[
            "features"
        ]

        for row in prediction_rows
    ]
)


# ============================================================
# Predict phase probabilities
# ============================================================

rock_scores = (
    positive_probability(
        rock_model,
        X,
    )
)


spinel_scores = (
    positive_probability(
        spinel_model,
        X,
    )
)


for (
    prediction_row,
    rock,
    spinel,
) in zip(
    prediction_rows,
    rock_scores,
    spinel_scores,
):

    rock = float(
        rock
    )

    spinel = float(
        spinel
    )


    dual = (
        rock
        * spinel
    )


    candidate = candidates[
        prediction_row[
            "candidate_index"
        ]
    ]


    route_name = (
        prediction_row[
            "route"
        ]
    )


    candidate[
        "route_predictions"
    ][
        route_name
    ] = {
        "rock_salt_score":
            rock,

        "spinel_score":
            spinel,

        "dual_phase_score":
            dual,

        "synthesis_scenario":
            dict(
                ROUTES[
                    route_name
                ]
            ),
    }


# ============================================================
# Preferred route
# ============================================================

for candidate in candidates:

    route_predictions = (
        candidate[
            "route_predictions"
        ]
    )


    best_route = max(
        route_predictions,
        key=lambda route:
            route_predictions[
                route
            ][
                "dual_phase_score"
            ],
    )


    best_prediction = (
        route_predictions[
            best_route
        ]
    )


    candidate[
        "best_route"
    ] = best_route


    candidate[
        "best_rock_salt_score"
    ] = best_prediction[
        "rock_salt_score"
    ]


    candidate[
        "best_spinel_score"
    ] = best_prediction[
        "spinel_score"
    ]


    candidate[
        "best_dual_phase_score"
    ] = best_prediction[
        "dual_phase_score"
    ]


    if (
        "air"
        in route_predictions
        and "ar"
        in route_predictions
    ):

        air_dual = (
            route_predictions[
                "air"
            ][
                "dual_phase_score"
            ]
        )

        ar_dual = (
            route_predictions[
                "ar"
            ][
                "dual_phase_score"
            ]
        )


        candidate[
            "route_robust_dual_score"
        ] = min(
            air_dual,
            ar_dual,
        )


        candidate[
            "route_dual_score_difference"
        ] = abs(
            air_dual
            - ar_dual
        )


    else:

        candidate[
            "route_robust_dual_score"
        ] = candidate[
            "best_dual_phase_score"
        ]


        candidate[
            "route_dual_score_difference"
        ] = None


# ============================================================
# Phase filtering
# ============================================================

filtered = []


for candidate in candidates:

    if (
        args.min_rock
        is not None
        and candidate[
            "best_rock_salt_score"
        ] < args.min_rock
    ):

        continue


    if (
        args.min_spinel
        is not None
        and candidate[
            "best_spinel_score"
        ] < args.min_spinel
    ):

        continue


    if (
        args.min_dual
        is not None
        and candidate[
            "best_dual_phase_score"
        ] < args.min_dual
    ):

        continue


    filtered.append(
        candidate
    )


candidates = filtered


if not candidates:

    raise RuntimeError(
        "No candidates remain after phase filtering."
    )


# ============================================================
# Ranking
#
# Primary:
#     highest dual-phase score
#
# Secondary:
#     EFA available
#
# Third:
#     lower EFA
#
# Fourth:
#     literature support
#
# Fifth:
#     route robustness
#
# Applicability-domain status is preserved but not filtered
# here. Final domain filtering is handled in pco_selection.py.
# ============================================================

def ranking_key(
    candidate
):

    efa = (
        candidate[
            "dft_evidence"
        ][
            "efa"
        ]
    )


    literature = (
        candidate[
            "literature_evidence"
        ]
    )


    return (
        -candidate[
            "best_dual_phase_score"
        ],

        (
            0
            if efa is not None
            else 1
        ),

        (
            efa
            if efa is not None
            else float(
                "inf"
            )
        ),

        -literature[
            "successful_count"
        ],

        -literature[
            "report_count"
        ],

        -candidate[
            "route_robust_dual_score"
        ],

        str(
            candidate[
                "material_auid"
            ]
        ),
    )


candidates.sort(
    key=
        ranking_key
)


# ============================================================
# Ranking metadata
# ============================================================

for rank, candidate in enumerate(
    candidates,
    start=1,
):

    candidate[
        "rank"
    ] = rank


    candidate[
        "ranking_basis"
    ] = {
        "primary":
            "highest_best_route_dual_phase_score",

        "secondary":
            "efa_available_then_lower_efa",

        "tertiary":
            "literature_support",

        "additional_tiebreak":
            "route_robust_dual_score",

        "weighted_score_used":
            False,
    }


# ============================================================
# Output files
# ============================================================

all_out = (
    OUTPUT_DIR
    / "candidate_predictions.jsonl"
)


route_out = (
    OUTPUT_DIR
    / "candidate_route_predictions.jsonl"
)


top_out = (
    OUTPUT_DIR
    / f"top_{args.top}_phase_candidates.json"
)


summary_out = (
    OUTPUT_DIR
    / "prediction_summary.json"
)


# ============================================================
# Write complete predictions
# ============================================================

with all_out.open(
    "w"
) as out:

    for candidate in candidates:

        write_jsonl(
            out,
            {
                "rank":
                    candidate[
                        "rank"
                    ],

                "material_auid":
                    candidate[
                        "material_auid"
                    ],

                "composition_key":
                    candidate[
                        "composition_key"
                    ],

                "model_cation_elements":
                    candidate[
                        "model_cation_elements"
                    ],

                "full_material_elements":
                    candidate[
                        "full_material_elements"
                    ],

                "in_training_element_domain":
                    candidate[
                        "in_training_element_domain"
                    ],

                "unseen_elements":
                    candidate[
                        "unseen_elements"
                    ],

                "has_oses_experiment":
                    candidate[
                        "has_oses_experiment"
                    ],

                "best_route":
                    candidate[
                        "best_route"
                    ],

                "best_rock_salt_score":
                    candidate[
                        "best_rock_salt_score"
                    ],

                "best_spinel_score":
                    candidate[
                        "best_spinel_score"
                    ],

                "best_dual_phase_score":
                    candidate[
                        "best_dual_phase_score"
                    ],

                "route_robust_dual_score":
                    candidate[
                        "route_robust_dual_score"
                    ],

                "route_dual_score_difference":
                    candidate[
                        "route_dual_score_difference"
                    ],

                "route_predictions":
                    candidate[
                        "route_predictions"
                    ],

                "dft_evidence":
                    candidate[
                        "dft_evidence"
                    ],

                "literature_evidence":
                    candidate[
                        "literature_evidence"
                    ],

                "ranking_basis":
                    candidate[
                        "ranking_basis"
                    ],

                "features":
                    candidate[
                        "features"
                    ],
            },
        )


# ============================================================
# Write one row per route
# ============================================================

with route_out.open(
    "w"
) as out:

    for candidate in candidates:

        for (
            route_name,
            prediction,
        ) in candidate[
            "route_predictions"
        ].items():

            write_jsonl(
                out,
                {
                    "rank":
                        candidate[
                            "rank"
                        ],

                    "material_auid":
                        candidate[
                            "material_auid"
                        ],

                    "composition_key":
                        candidate[
                            "composition_key"
                        ],

                    "model_cation_elements":
                        candidate[
                            "model_cation_elements"
                        ],

                    "in_training_element_domain":
                        candidate[
                            "in_training_element_domain"
                        ],

                    "unseen_elements":
                        candidate[
                            "unseen_elements"
                        ],

                    "route":
                        route_name,

                    "rock_salt_score":
                        prediction[
                            "rock_salt_score"
                        ],

                    "spinel_score":
                        prediction[
                            "spinel_score"
                        ],

                    "dual_phase_score":
                        prediction[
                            "dual_phase_score"
                        ],

                    "synthesis_scenario":
                        prediction[
                            "synthesis_scenario"
                        ],

                    "dft_evidence":
                        candidate[
                            "dft_evidence"
                        ],

                    "literature_evidence":
                        candidate[
                            "literature_evidence"
                        ],
                },
            )


# ============================================================
# Top-N output
# ============================================================

top_candidates = []


for candidate in candidates[
    :args.top
]:

    top_candidates.append(
        {
            "rank":
                candidate[
                    "rank"
                ],

            "material_auid":
                candidate[
                    "material_auid"
                ],

            "composition_key":
                candidate[
                    "composition_key"
                ],

            "model_cation_elements":
                candidate[
                    "model_cation_elements"
                ],

            "full_material_elements":
                candidate[
                    "full_material_elements"
                ],

            "in_training_element_domain":
                candidate[
                    "in_training_element_domain"
                ],

            "unseen_elements":
                candidate[
                    "unseen_elements"
                ],

            "best_route":
                candidate[
                    "best_route"
                ],

            "phase_prediction": {
                "rock_salt":
                    candidate[
                        "best_rock_salt_score"
                    ],

                "spinel":
                    candidate[
                        "best_spinel_score"
                    ],

                "dual_phase":
                    candidate[
                        "best_dual_phase_score"
                    ],

                "route_robust_dual":
                    candidate[
                        "route_robust_dual_score"
                    ],

                "route_difference":
                    candidate[
                        "route_dual_score_difference"
                    ],
            },

            "route_predictions":
                candidate[
                    "route_predictions"
                ],

            "dft_evidence":
                candidate[
                    "dft_evidence"
                ],

            "literature_evidence":
                candidate[
                    "literature_evidence"
                ],

            "ranking_basis":
                candidate[
                    "ranking_basis"
                ],
        }
    )


with top_out.open(
    "w"
) as out:

    json.dump(
        top_candidates,
        out,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# Summary
# ============================================================

with_efa = sum(
    1

    for candidate
    in candidates

    if candidate[
        "dft_evidence"
    ][
        "efa"
    ]
    is not None
)


with_literature = sum(
    1

    for candidate
    in candidates

    if candidate[
        "literature_evidence"
    ][
        "report_count"
    ] > 0
)


in_domain_count = sum(
    1

    for candidate
    in candidates

    if candidate[
        "in_training_element_domain"
    ]
    is True
)


out_of_domain_count = sum(
    1

    for candidate
    in candidates

    if candidate[
        "in_training_element_domain"
    ]
    is False
)


best_route_counts = {
    route:
        sum(
            1

            for candidate
            in candidates

            if candidate[
                "best_route"
            ] == route
        )

    for route in route_names
}


summary = {
    "eligible_candidates":
        len(
            candidates
        ),

    "routes_evaluated":
        route_names,

    "candidates_with_efa":
        with_efa,

    "candidates_with_literature":
        with_literature,

    "candidate_domain_coverage": {
        "in_training_element_domain":
            in_domain_count,

        "out_of_training_element_domain":
            out_of_domain_count,
    },

    "best_route_counts":
        best_route_counts,

    "ranking_strategy": {
        "weighted_score":
            False,

        "primary":
            "best_route_dual_phase_score_desc",

        "secondary":
            "efa_available_then_efa_asc",

        "tertiary":
            "literature_support",

        "additional_tiebreak":
            "route_robust_dual_score_desc",
    },

    "filters": {
        "include_tested":
            args.include_tested,

        "require_efa":
            args.require_efa,

        "max_efa":
            args.max_efa,

        "min_rock":
            args.min_rock,

        "min_spinel":
            args.min_spinel,

        "min_dual":
            args.min_dual,
    },

    "outputs": {
        "candidate_predictions":
            str(
                all_out
            ),

        "candidate_route_predictions":
            str(
                route_out
            ),

        "top_candidates":
            str(
                top_out
            ),
    },
}


with summary_out.open(
    "w"
) as out:

    json.dump(
        summary,
        out,
        indent=2,
    )


# ============================================================
# Console
# ============================================================

print()
print(
    "=== PREDICTION COMPLETE ==="
)

print(
    "Eligible candidates:",
    len(
        candidates
    ),
)

print(
    "Candidates with EFA:",
    with_efa,
)

print(
    "Candidates with literature:",
    with_literature,
)

print(
    "In training element domain:",
    in_domain_count,
)

print(
    "Out of training element domain:",
    out_of_domain_count,
)

print(
    "Routes evaluated:",
    route_names,
)

print(
    "Best-route counts:",
    best_route_counts,
)

print(
    "Candidate predictions:",
    all_out,
)

print(
    "Route predictions:",
    route_out,
)

print(
    "Top candidates:",
    top_out,
)

print()


for candidate in candidates[
    :args.top
]:

    dft = (
        candidate[
            "dft_evidence"
        ]
    )

    literature = (
        candidate[
            "literature_evidence"
        ]
    )


    print(
        candidate[
            "rank"
        ],

        candidate[
            "material_auid"
        ],

        "Domain=",
        candidate[
            "in_training_element_domain"
        ],

        "Unseen=",
        candidate[
            "unseen_elements"
        ],

        "Route=",
        candidate[
            "best_route"
        ],

        "RS=",
        round(
            candidate[
                "best_rock_salt_score"
            ],
            4,
        ),

        "Spinel=",
        round(
            candidate[
                "best_spinel_score"
            ],
            4,
        ),

        "Dual=",
        round(
            candidate[
                "best_dual_phase_score"
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
    )