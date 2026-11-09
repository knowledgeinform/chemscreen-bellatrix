from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


# ============================================================
# Paths
# ============================================================

ROOT = (
    Path.home()
    / "scratch"
    / "chemscreen_data"
)

TRAINING_DATA = (
    ROOT
    / "training"
    / "phase_training.jsonl"
)

MODEL_DIR = (
    Path(__file__)
    .resolve()
    .parent
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VECTORIZER_OUT = (
    MODEL_DIR
    / "feature_vectorizer.joblib"
)

ROCK_MODEL_OUT = (
    MODEL_DIR
    / "rock_salt_rf.joblib"
)

SPINEL_MODEL_OUT = (
    MODEL_DIR
    / "spinel_rf.joblib"
)

FEATURES_OUT = (
    MODEL_DIR
    / "phase_model_features.json"
)

METRICS_OUT = (
    MODEL_DIR
    / "phase_model_metrics.json"
)

OOF_OUT = (
    MODEL_DIR
    / "phase_model_oof_predictions.jsonl"
)


# ============================================================
# Model configuration
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 500

MAX_DEPTH = None

MIN_SAMPLES_LEAF = 2

N_JOBS = -1

N_SPLITS = 5


# ============================================================
# Supervised feature policy
#
# The current OSES phase model learns:
#
#     composition
#     +
#     composition-derived chemistry
#     +
#     experimentally supported synthesis route
#
# Current OSES synthesis routes:
#
#     air
#         1100 C / 5 h
#         air atmosphere
#         air quench
#
#     ar
#         1100 C / 5 h
#         Ar atmosphere
#         rapid hot-zone removal
#
# Temperature, hold time, quench status, etc. are not included
# independently because they do not vary independently in the
# current OSES training set.
#
# DFT and literature features are also excluded from this first
# supervised model because their OSES overlap is sparse.
#
# They remain available downstream for candidate ranking.
# ============================================================

ALLOWED_PREFIXES = (
    "element_fraction__",
    "element_present__",
    "chem_",
)

ALLOWED_EXACT_FEATURES = {
    "material_num_elements",
    "synth_route",
}

EXCLUDED_PREFIXES = (
    "comp_",
    "equiv_comp_",
    "lit_",
    "equiv_lit_",
)

EXCLUDED_SYNTHESIS_FEATURES = {
    "synth_atmosphere",
    "synth_max_temp_c",
    "synth_total_hold_time_hours",
    "synth_quenched",
    "synth_quench_medium",
    "synth_step_count",
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


def feature_allowed(
    key,
):
    """
    Return True only for features intentionally included
    in the supervised OSES phase model.
    """

    if (
        key
        in ALLOWED_EXACT_FEATURES
    ):
        return True

    if (
        key
        in EXCLUDED_SYNTHESIS_FEATURES
    ):
        return False

    if key.startswith(
        "synth_has_step__"
    ):
        return False

    if any(
        key.startswith(
            prefix
        )
        for prefix
        in EXCLUDED_PREFIXES
    ):
        return False

    return any(
        key.startswith(
            prefix
        )
        for prefix
        in ALLOWED_PREFIXES
    )


def clean_feature_value(
    value,
):
    """
    Keep only DictVectorizer-compatible feature values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return float(
            value
        )

    if isinstance(
        value,
        (int, float),
    ):

        if not finite(
            value
        ):
            return None

        return float(
            value
        )

    if isinstance(
        value,
        str,
    ):

        value = value.strip()

        if not value:
            return None

        return value

    return None


def filter_features(
    features,
):
    out = {}

    for key, value in (
        features.items()
    ):

        if not feature_allowed(
            key
        ):
            continue

        cleaned = clean_feature_value(
            value
        )

        if cleaned is None:
            continue

        out[
            key
        ] = cleaned

    return out


def positive_probability(
    model,
    X,
):
    """
    Return P(class=1).

    Handles the edge case where a training fold contains
    only class 0.
    """

    classes = list(
        model.classes_
    )

    if 1 not in classes:

        return np.zeros(
            X.shape[0],
            dtype=float,
        )

    index = classes.index(
        1
    )

    return model.predict_proba(
        X
    )[:, index]


def binary_metrics(
    y_true,
    y_pred,
    y_prob,
):
    result = {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "confusion_matrix":
            confusion_matrix(
                y_true,
                y_pred,
                labels=[
                    0,
                    1,
                ],
            ).tolist(),
    }

    if len(
        set(
            y_true
        )
    ) > 1:

        result[
            "roc_auc"
        ] = float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        )

    else:

        result[
            "roc_auc"
        ] = None

    return result


def make_model():
    return RandomForestClassifier(
        n_estimators=
            N_ESTIMATORS,

        max_depth=
            MAX_DEPTH,

        min_samples_leaf=
            MIN_SAMPLES_LEAF,

        class_weight=
            "balanced",

        random_state=
            RANDOM_STATE,

        n_jobs=
            N_JOBS,
    )


def summarize_feature_coverage(
    rows,
):
    presence = Counter()

    for row in rows:

        for key, value in (
            row[
                "model_features"
            ].items()
        ):

            if value not in (
                None,
                "",
            ):

                presence[
                    key
                ] += 1

    return dict(
        sorted(
            presence.items()
        )
    )


def count_routes(
    rows,
):
    routes = Counter()

    for row in rows:

        route = (
            row[
                "model_features"
            ].get(
                "synth_route"
            )
        )

        routes[
            str(
                route
            )
        ] += 1

    return dict(
        routes
    )


# ============================================================
# Load training rows
# ============================================================

rows = []

for record in read_jsonl(
    TRAINING_DATA
):

    targets = (
        record.get(
            "targets"
        )
        or {}
    )

    features = (
        record.get(
            "features"
        )
        or {}
    )

    composition_key = (
        record.get(
            "composition_key"
        )
    )

    if not composition_key:
        continue

    if (
        "rock_salt_present"
        not in targets
        or "spinel_present"
        not in targets
    ):
        continue

    model_features = filter_features(
        features
    )

    if not model_features:
        continue

    synth_route = (
        model_features.get(
            "synth_route"
        )
    )

    if synth_route not in {
        "air",
        "ar",
    }:

        raise RuntimeError(
            "Unexpected or missing synth_route "
            f"for OSES row "
            f"{record.get('oses_id')!r}: "
            f"{synth_route!r}"
        )

    rows.append(
        {
            "oses_id":
                record.get(
                    "oses_id"
                ),

            "material_auid":
                record.get(
                    "material_auid"
                ),

            "recipe_auid":
                record.get(
                    "recipe_auid"
                ),

            "composition_key":
                composition_key,

            "confidence":
                record.get(
                    "confidence"
                ),

            "model_features":
                model_features,

            "rock_target":
                int(
                    bool(
                        targets[
                            "rock_salt_present"
                        ]
                    )
                ),

            "spinel_target":
                int(
                    bool(
                        targets[
                            "spinel_present"
                        ]
                    )
                ),
        }
    )


if not rows:

    raise RuntimeError(
        "No usable training rows found."
    )


# ============================================================
# Dataset validation
# ============================================================

print(
    "Training rows:",
    len(
        rows
    ),
)


unique_groups = sorted(
    {
        row[
            "composition_key"
        ]

        for row in rows
    }
)


print(
    "Unique composition groups:",
    len(
        unique_groups
    ),
)


route_counts = count_routes(
    rows
)


print(
    "Synthesis routes:",
    route_counts,
)


if (
    route_counts.get(
        "air",
        0,
    )
    != 43
    or route_counts.get(
        "ar",
        0,
    )
    != 21
):

    print(
        "WARNING: synthesis-route counts differ "
        "from the previously validated 43 Air / 21 Ar."
    )


if len(
    unique_groups
) < 2:

    raise RuntimeError(
        "Need at least two unique composition groups "
        "for grouped cross-validation."
    )


# ============================================================
# Targets / groups
# ============================================================

y_rock = np.array(
    [
        row[
            "rock_target"
        ]

        for row in rows
    ],
    dtype=int,
)

y_spinel = np.array(
    [
        row[
            "spinel_target"
        ]

        for row in rows
    ],
    dtype=int,
)

groups = np.array(
    [
        row[
            "composition_key"
        ]

        for row in rows
    ],
    dtype=object,
)


print(
    "Rock salt:",
    {
        "positive":
            int(
                y_rock.sum()
            ),

        "negative":
            int(
                len(
                    y_rock
                )
                - y_rock.sum()
            ),
    },
)


print(
    "Spinel:",
    {
        "positive":
            int(
                y_spinel.sum()
            ),

        "negative":
            int(
                len(
                    y_spinel
                )
                - y_spinel.sum()
            ),
    },
)


# ============================================================
# Shared feature vectorizer
# ============================================================

vectorizer = DictVectorizer(
    sparse=True,
)


feature_dicts = [
    row[
        "model_features"
    ]

    for row in rows
]


X = vectorizer.fit_transform(
    feature_dicts
)


feature_names = (
    vectorizer
    .get_feature_names_out()
    .tolist()
)


print(
    "Vectorized features:",
    X.shape[
        1
    ],
)


print(
    "Feature names:"
)

for name in feature_names:
    print(
        " ",
        name,
    )


# ============================================================
# Grouped cross-validation
#
# Group = composition_key
#
# Different synthesis routes for the same composition must
# remain together in the same fold.
# ============================================================

n_splits = min(
    N_SPLITS,
    len(
        unique_groups
    ),
)


if n_splits < 2:

    raise RuntimeError(
        "Not enough groups for GroupKFold."
    )


gkf = GroupKFold(
    n_splits=
        n_splits
)


rock_oof_prob = np.full(
    len(
        rows
    ),
    np.nan,
)

spinel_oof_prob = np.full(
    len(
        rows
    ),
    np.nan,
)

rock_oof_pred = np.full(
    len(
        rows
    ),
    -1,
    dtype=int,
)

spinel_oof_pred = np.full(
    len(
        rows
    ),
    -1,
    dtype=int,
)


fold_summaries = []


for fold_index, (
    train_idx,
    test_idx,
) in enumerate(
    gkf.split(
        X,
        y_rock,
        groups,
    ),
    start=1,
):

    print()
    print(
        f"Fold "
        f"{fold_index}/"
        f"{n_splits}"
    )

    print(
        "  train rows:",
        len(
            train_idx
        ),
    )

    print(
        "  test rows:",
        len(
            test_idx
        ),
    )

    print(
        "  train groups:",
        len(
            set(
                groups[
                    train_idx
                ]
            )
        ),
    )

    print(
        "  test groups:",
        len(
            set(
                groups[
                    test_idx
                ]
            )
        ),
    )

    rock_model = make_model()

    spinel_model = make_model()


    rock_model.fit(
        X[
            train_idx
        ],
        y_rock[
            train_idx
        ],
    )

    spinel_model.fit(
        X[
            train_idx
        ],
        y_spinel[
            train_idx
        ],
    )


    rock_prob = (
        positive_probability(
            rock_model,
            X[
                test_idx
            ],
        )
    )

    spinel_prob = (
        positive_probability(
            spinel_model,
            X[
                test_idx
            ],
        )
    )


    rock_pred = (
        rock_prob
        >= 0.5
    ).astype(
        int
    )

    spinel_pred = (
        spinel_prob
        >= 0.5
    ).astype(
        int
    )


    rock_oof_prob[
        test_idx
    ] = rock_prob

    spinel_oof_prob[
        test_idx
    ] = spinel_prob

    rock_oof_pred[
        test_idx
    ] = rock_pred

    spinel_oof_pred[
        test_idx
    ] = spinel_pred


    fold_summary = {
        "fold":
            fold_index,

        "train_rows":
            len(
                train_idx
            ),

        "test_rows":
            len(
                test_idx
            ),

        "train_groups":
            len(
                set(
                    groups[
                        train_idx
                    ]
                )
            ),

        "test_groups":
            len(
                set(
                    groups[
                        test_idx
                    ]
                )
            ),

        "rock_salt":
            binary_metrics(
                y_rock[
                    test_idx
                ],
                rock_pred,
                rock_prob,
            ),

        "spinel":
            binary_metrics(
                y_spinel[
                    test_idx
                ],
                spinel_pred,
                spinel_prob,
            ),
    }

    fold_summaries.append(
        fold_summary
    )


# ============================================================
# Validate OOF predictions
# ============================================================

if np.isnan(
    rock_oof_prob
).any():

    raise RuntimeError(
        "Missing rock-salt OOF predictions."
    )


if np.isnan(
    spinel_oof_prob
).any():

    raise RuntimeError(
        "Missing spinel OOF predictions."
    )


if (
    rock_oof_pred
    < 0
).any():

    raise RuntimeError(
        "Missing rock-salt OOF classes."
    )


if (
    spinel_oof_pred
    < 0
).any():

    raise RuntimeError(
        "Missing spinel OOF classes."
    )


# ============================================================
# Overall grouped-CV metrics
# ============================================================

rock_metrics = (
    binary_metrics(
        y_rock,
        rock_oof_pred,
        rock_oof_prob,
    )
)

spinel_metrics = (
    binary_metrics(
        y_spinel,
        spinel_oof_pred,
        spinel_oof_prob,
    )
)


print(
    "\n=== ROCK SALT GROUPED CV ==="
)

print(
    json.dumps(
        rock_metrics,
        indent=2,
    )
)


print(
    "\n=== SPINEL GROUPED CV ==="
)

print(
    json.dumps(
        spinel_metrics,
        indent=2,
    )
)


# ============================================================
# Detailed classification reports
# ============================================================

rock_report = classification_report(
    y_rock,
    rock_oof_pred,
    labels=[
        0,
        1,
    ],
    output_dict=True,
    zero_division=0,
)

spinel_report = classification_report(
    y_spinel,
    spinel_oof_pred,
    labels=[
        0,
        1,
    ],
    output_dict=True,
    zero_division=0,
)


# ============================================================
# OOF predictions
# ============================================================

with OOF_OUT.open(
    "w"
) as out:

    for index, row in enumerate(
        rows
    ):

        write_jsonl(
            out,
            {
                "oses_id":
                    row[
                        "oses_id"
                    ],

                "material_auid":
                    row[
                        "material_auid"
                    ],

                "recipe_auid":
                    row[
                        "recipe_auid"
                    ],

                "composition_key":
                    row[
                        "composition_key"
                    ],

                "synth_route":
                    row[
                        "model_features"
                    ].get(
                        "synth_route"
                    ),

                "rock_salt_target":
                    int(
                        y_rock[
                            index
                        ]
                    ),

                "rock_salt_probability":
                    float(
                        rock_oof_prob[
                            index
                        ]
                    ),

                "rock_salt_prediction":
                    int(
                        rock_oof_pred[
                            index
                        ]
                    ),

                "spinel_target":
                    int(
                        y_spinel[
                            index
                        ]
                    ),

                "spinel_probability":
                    float(
                        spinel_oof_prob[
                            index
                        ]
                    ),

                "spinel_prediction":
                    int(
                        spinel_oof_pred[
                            index
                        ]
                    ),

                "dual_phase_probability_product":
                    float(
                        rock_oof_prob[
                            index
                        ]
                        * spinel_oof_prob[
                            index
                        ]
                    ),
            },
        )


# ============================================================
# Fit final models on all OSES data
# ============================================================

rock_model = make_model()

spinel_model = make_model()


rock_model.fit(
    X,
    y_rock,
)

spinel_model.fit(
    X,
    y_spinel,
)


# ============================================================
# Save model artifacts
# ============================================================

joblib.dump(
    vectorizer,
    VECTORIZER_OUT,
)

joblib.dump(
    rock_model,
    ROCK_MODEL_OUT,
)

joblib.dump(
    spinel_model,
    SPINEL_MODEL_OUT,
)


# ============================================================
# Feature metadata
# ============================================================

feature_metadata = {
    "training_rows":
        len(
            rows
        ),

    "unique_composition_groups":
        len(
            unique_groups
        ),

    "synthesis_routes":
        route_counts,

    "feature_count":
        len(
            feature_names
        ),

    "allowed_prefixes":
        list(
            ALLOWED_PREFIXES
        ),

    "allowed_exact_features":
        sorted(
            ALLOWED_EXACT_FEATURES
        ),

    "excluded_prefixes":
        list(
            EXCLUDED_PREFIXES
        ),

    "excluded_synthesis_features":
        sorted(
            EXCLUDED_SYNTHESIS_FEATURES
        ),

    "feature_names":
        feature_names,

    "feature_coverage":
        summarize_feature_coverage(
            rows
        ),
}


with FEATURES_OUT.open(
    "w"
) as f:

    json.dump(
        feature_metadata,
        f,
        indent=2,
    )


# ============================================================
# Metrics metadata
# ============================================================

metrics = {
    "training_rows":
        len(
            rows
        ),

    "unique_composition_groups":
        len(
            unique_groups
        ),

    "synthesis_routes":
        route_counts,

    "group_kfold_splits":
        n_splits,

    "model_configuration": {
        "type":
            "RandomForestClassifier",

        "n_estimators":
            N_ESTIMATORS,

        "max_depth":
            MAX_DEPTH,

        "min_samples_leaf":
            MIN_SAMPLES_LEAF,

        "class_weight":
            "balanced",

        "random_state":
            RANDOM_STATE,
    },

    "feature_policy": {
        "description":
            (
                "Composition + chemistry + OSES synthesis route. "
                "DFT and literature excluded from supervised "
                "phase-model training and retained for downstream "
                "candidate ranking."
            ),

        "allowed_prefixes":
            list(
                ALLOWED_PREFIXES
            ),

        "allowed_exact_features":
            sorted(
                ALLOWED_EXACT_FEATURES
            ),
    },

    "target_balance": {
        "rock_salt": {
            "positive":
                int(
                    y_rock.sum()
                ),

            "negative":
                int(
                    len(
                        y_rock
                    )
                    - y_rock.sum()
                ),
        },

        "spinel": {
            "positive":
                int(
                    y_spinel.sum()
                ),

            "negative":
                int(
                    len(
                        y_spinel
                    )
                    - y_spinel.sum()
                ),
        },
    },

    "grouped_cv": {
        "rock_salt":
            rock_metrics,

        "spinel":
            spinel_metrics,

        "rock_salt_classification_report":
            rock_report,

        "spinel_classification_report":
            spinel_report,

        "folds":
            fold_summaries,
    },
}


with METRICS_OUT.open(
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )


# ============================================================
# Feature importance
# ============================================================

rock_importance = sorted(
    zip(
        feature_names,
        rock_model.feature_importances_,
    ),
    key=lambda item:
        item[
            1
        ],
    reverse=True,
)

spinel_importance = sorted(
    zip(
        feature_names,
        spinel_model.feature_importances_,
    ),
    key=lambda item:
        item[
            1
        ],
    reverse=True,
)


print(
    "\n=== TOP ROCK-SALT FEATURES ==="
)

for (
    name,
    importance,
) in rock_importance[
    :20
]:

    print(
        f"{name:50} "
        f"{importance:.6f}"
    )


print(
    "\n=== TOP SPINEL FEATURES ==="
)

for (
    name,
    importance,
) in spinel_importance[
    :20
]:

    print(
        f"{name:50} "
        f"{importance:.6f}"
    )


# ============================================================
# Final summary
# ============================================================

print(
    "\n=== TRAINING COMPLETE ==="
)

print(
    "Vectorizer:",
    VECTORIZER_OUT,
)

print(
    "Rock-salt model:",
    ROCK_MODEL_OUT,
)

print(
    "Spinel model:",
    SPINEL_MODEL_OUT,
)

print(
    "Feature metadata:",
    FEATURES_OUT,
)

print(
    "Metrics:",
    METRICS_OUT,
)

print(
    "OOF predictions:",
    OOF_OUT,
)