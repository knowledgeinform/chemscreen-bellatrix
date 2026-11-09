from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# Paths
# ============================================================

ROOT = (
    Path.home()
    / "scratch"
    / "chemscreen_data"
)

MATERIALS = (
    ROOT
    / "materials"
    / "materials.jsonl"
)

COMPUTATIONAL = (
    ROOT
    / "computational"
    / "computational.jsonl"
)

LITERATURE = (
    ROOT
    / "literature"
    / "literature.jsonl"
)

EXPERIMENTS = (
    ROOT
    / "experimental"
    / "oses_experiments.jsonl"
)

TRAINING_DIR = (
    ROOT
    / "training"
)

TRAINING_OUT = (
    TRAINING_DIR
    / "phase_training.jsonl"
)

CANDIDATE_OUT = (
    TRAINING_DIR
    / "candidate_features.jsonl"
)

SUMMARY_OUT = (
    TRAINING_DIR
    / "feature_build_summary.json"
)


# ============================================================
# Basic IO helpers
# ============================================================

def read_jsonl(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(
                    line
                )


def write_jsonl(handle, obj):
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


# ============================================================
# Statistical helpers
# ============================================================

def stats(
    values,
    prefix,
):
    values = [
        float(v)
        for v in values
        if finite(v)
    ]

    if not values:
        return {}

    out = {
        f"{prefix}_count":
            len(values),

        f"{prefix}_min":
            min(values),

        f"{prefix}_max":
            max(values),

        f"{prefix}_mean":
            statistics.fmean(
                values
            ),
    }

    if len(values) > 1:

        out[
            f"{prefix}_std"
        ] = statistics.pstdev(
            values
        )

    else:

        out[
            f"{prefix}_std"
        ] = 0.0

    return out


def mode(values):
    values = [
        v
        for v in values
        if v not in (
            None,
            "",
        )
    ]

    if not values:
        return None

    return Counter(
        values
    ).most_common(1)[0][0]


# ============================================================
# Model-facing composition helpers
#
# IMPORTANT
#
# OSES experimental compositions are represented by the
# cation sublattice. LOOP material records generally include
# oxygen.
#
# To keep training and prediction in the same feature space,
# oxygen is removed and the remaining cations are
# renormalized to sum to 1.
#
# The original full oxide composition is still preserved in
# the source material records.
# ============================================================

def cation_only_elements(
    elements,
):
    if not isinstance(
        elements,
        dict,
    ):
        return {}

    filtered = {}

    for element, value in (
        elements.items()
    ):

        element = str(
            element
        )

        # Current ChemScreen / OSES application is oxide
        # focused, so oxygen is excluded from the model-facing
        # cation composition.
        if element == "O":
            continue

        if not isinstance(
            value,
            (int, float),
        ):
            continue

        value = float(
            value
        )

        if (
            not math.isfinite(
                value
            )
            or value <= 0
        ):
            continue

        filtered[
            element
        ] = value

    total = sum(
        filtered.values()
    )

    if total <= 0:
        return {}

    return {
        element:
            value / total

        for element, value
        in filtered.items()
    }


def normalized_composition_key(
    elements,
):
    """
    Build a canonical cation-only composition key.
    """

    normalized = cation_only_elements(
        elements
    )

    if not normalized:
        return None

    return "|".join(
        f"{element}:"
        f"{normalized[element]:.8f}"

        for element
        in sorted(
            normalized
        )
    )


def material_cation_elements(
    material_record,
):
    material = (
        material_record.get(
            "material"
        )
        or {}
    )

    return cation_only_elements(
        material.get(
            "elements"
        )
    )


def experiment_cation_elements(
    experiment,
    material_record,
):
    """
    Prefer LOOP trial elements because they represent the
    experimental cation composition directly.

    Fall back to the material record.
    """

    loop_elements = (
        experiment.get(
            "loop_elements"
        )
    )

    cations = cation_only_elements(
        loop_elements
    )

    if cations:
        return cations

    return material_cation_elements(
        material_record
    )


def composition_key(
    experiment,
    material_record,
):
    """
    Canonical cation-only composition key for an experiment.
    """

    cations = experiment_cation_elements(
        experiment,
        material_record,
    )

    key = normalized_composition_key(
        cations
    )

    if key:
        return key

    # Final fallback only if structured elemental composition
    # cannot be constructed.
    label = experiment.get(
        "composition"
    )

    if label:
        return (
            str(label)
            .replace(" ", "")
            .lower()
        )

    return experiment.get(
        "material_auid"
    )


def material_composition_key(
    material_record,
):
    return normalized_composition_key(
        material_cation_elements(
            material_record
        )
    )


# ============================================================
# Composition-derived chemistry descriptors
# ============================================================

_PERIODIC_SYMBOLS = (
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
    "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
    "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
    "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No",
    "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)

ATOMIC_NUMBER = {
    symbol:
        index

    for index, symbol
    in enumerate(
        _PERIODIC_SYMBOLS,
        start=1,
    )
}

ALKALI_METALS = {
    "Li", "Na", "K", "Rb", "Cs", "Fr",
}

ALKALINE_EARTH_METALS = {
    "Be", "Mg", "Ca", "Sr", "Ba", "Ra",
}

D_BLOCK_ELEMENTS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn",

    "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd",

    "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg",

    "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg", "Cn",
}

D_BLOCK_3D = {
    "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn",
}

LANTHANIDES = {
    "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
}

ACTINIDES = {
    "Ac", "Th", "Pa", "U", "Np",
    "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
}


def atomic_period(z):
    if z <= 2:
        return 1

    if z <= 10:
        return 2

    if z <= 18:
        return 3

    if z <= 36:
        return 4

    if z <= 54:
        return 5

    if z <= 86:
        return 6

    return 7


def chemistry_features(
    elements,
):
    """
    Generate chemistry descriptors from the same cation-only
    representation used by element_fraction__*.
    """

    fractions = cation_only_elements(
        elements
    )

    if not fractions:
        return {}

    # Ignore any symbols not present in the periodic table
    # lookup.
    fractions = {
        element:
            fraction

        for element, fraction
        in fractions.items()

        if element in ATOMIC_NUMBER
    }

    total = sum(
        fractions.values()
    )

    if total <= 0:
        return {}

    # Renormalize again in case an unsupported symbol was
    # removed.
    fractions = {
        element:
            fraction / total

        for element, fraction
        in fractions.items()
    }

    n = len(
        fractions
    )

    if n == 0:
        return {}

    fraction_values = list(
        fractions.values()
    )

    # --------------------------------------------------------
    # Mixing / composition diversity
    # --------------------------------------------------------

    entropy = -sum(
        x * math.log(x)

        for x in fraction_values

        if x > 0
    )

    concentration = sum(
        x ** 2

        for x in fraction_values
    )

    ideal_fraction = (
        1.0 / n
    )

    equimolar_deviation = sum(
        abs(
            x - ideal_fraction
        )

        for x in fraction_values
    )

    fraction_mean = (
        sum(
            fraction_values
        )
        / n
    )

    fraction_variance = (
        sum(
            (
                x
                - fraction_mean
            ) ** 2

            for x in fraction_values
        )
        / n
    )

    # --------------------------------------------------------
    # Atomic-number descriptors
    # --------------------------------------------------------

    z_values = {
        element:
            ATOMIC_NUMBER[
                element
            ]

        for element
        in fractions
    }

    z_mean = sum(
        fractions[element]
        * z_values[element]

        for element
        in fractions
    )

    z_variance = sum(
        fractions[element]
        * (
            z_values[element]
            - z_mean
        ) ** 2

        for element
        in fractions
    )

    z_min = min(
        z_values.values()
    )

    z_max = max(
        z_values.values()
    )

    # --------------------------------------------------------
    # Period descriptors
    # --------------------------------------------------------

    periods = {
        element:
            atomic_period(
                z_values[
                    element
                ]
            )

        for element
        in fractions
    }

    period_mean = sum(
        fractions[element]
        * periods[element]

        for element
        in fractions
    )

    period_variance = sum(
        fractions[element]
        * (
            periods[element]
            - period_mean
        ) ** 2

        for element
        in fractions
    )

    def family_fraction(
        family
    ):
        return sum(
            fraction

            for element, fraction
            in fractions.items()

            if element in family
        )

    return {
        "chem_num_elements":
            n,

        "chem_config_entropy_kb":
            entropy,

        "chem_effective_num_elements":
            math.exp(
                entropy
            ),

        "chem_concentration_index":
            concentration,

        "chem_equimolar_deviation_l1":
            equimolar_deviation,

        "chem_fraction_min":
            min(
                fraction_values
            ),

        "chem_fraction_max":
            max(
                fraction_values
            ),

        "chem_fraction_std":
            math.sqrt(
                fraction_variance
            ),

        "chem_atomic_number_mean":
            z_mean,

        "chem_atomic_number_std":
            math.sqrt(
                z_variance
            ),

        "chem_atomic_number_min":
            float(
                z_min
            ),

        "chem_atomic_number_max":
            float(
                z_max
            ),

        "chem_atomic_number_range":
            float(
                z_max
                - z_min
            ),

        "chem_period_mean":
            period_mean,

        "chem_period_std":
            math.sqrt(
                period_variance
            ),

        "chem_period_min":
            float(
                min(
                    periods.values()
                )
            ),

        "chem_period_max":
            float(
                max(
                    periods.values()
                )
            ),

        "chem_d_block_fraction":
            family_fraction(
                D_BLOCK_ELEMENTS
            ),

        "chem_3d_d_block_fraction":
            family_fraction(
                D_BLOCK_3D
            ),

        "chem_alkali_fraction":
            family_fraction(
                ALKALI_METALS
            ),

        "chem_alkaline_earth_fraction":
            family_fraction(
                ALKALINE_EARTH_METALS
            ),

        "chem_lanthanide_fraction":
            family_fraction(
                LANTHANIDES
            ),

        "chem_actinide_fraction":
            family_fraction(
                ACTINIDES
            ),
    }


# ============================================================
# Material / elemental features
# ============================================================

def element_features(
    material_record,
    model_elements,
):
    """
    Build elemental features from cation-only model elements.

    Material metadata is still available for provenance.
    """

    out = {}

    fractions = cation_only_elements(
        model_elements
    )

    for element, value in (
        fractions.items()
    ):

        out[
            f"element_fraction__{element}"
        ] = float(
            value
        )

    out[
        "material_num_elements"
    ] = len(
        fractions
    )

    material = (
        material_record.get(
            "material"
        )
        or {}
    )

    family = material.get(
        "structure_family"
    )

    if family:

        out[
            "material_structure_family"
        ] = str(
            family
        )

    return out


# ============================================================
# Synthesis features
# ============================================================

def synthesis_features(
    experiment
):
    steps = (
        experiment.get(
            "synthesis_steps"
        )
        or []
    )

    out = {}

    step_types = []

    heat_temps = []
    heat_times = []

    # --------------------------------------------------------
    # Prefer atmosphere resolved by OSES <-> LOOP linkage.
    # --------------------------------------------------------

    resolved_atmosphere = (
        experiment.get(
            "resolved_atmosphere"
        )
    )

    if resolved_atmosphere:

        out[
            "synth_atmosphere"
        ] = str(
            resolved_atmosphere
        ).lower()

    # --------------------------------------------------------
    # Walk synthesis steps
    # --------------------------------------------------------

    for step in steps:

        if not isinstance(
            step,
            dict,
        ):
            continue

        step_type = step.get(
            "step_type"
        )

        if step_type:

            step_types.append(
                str(
                    step_type
                )
            )

        # ----------------------------------------------------
        # Structured heat treatment
        # ----------------------------------------------------

        if (
            step_type
            == "heat_treatment"
        ):

            temp = step.get(
                "max_temp_c"
            )

            hold = step.get(
                "hold_time_hours"
            )

            atmosphere = (
                step.get(
                    "atmosphere"
                )
            )

            if finite(
                temp
            ):

                heat_temps.append(
                    float(
                        temp
                    )
                )

            if finite(
                hold
            ):

                heat_times.append(
                    float(
                        hold
                    )
                )

            if atmosphere:

                out[
                    "synth_atmosphere"
                ] = str(
                    atmosphere
                ).lower()

        # ----------------------------------------------------
        # Structured ball milling
        # ----------------------------------------------------

        elif (
            step_type
            == "ball_milling"
        ):

            for source, target in [
                (
                    "milling_time_hours",
                    "synth_milling_time_hours",
                ),
                (
                    "milling_rpm",
                    "synth_milling_rpm",
                ),
            ]:

                value = step.get(
                    source
                )

                if finite(
                    value
                ):

                    out[
                        target
                    ] = float(
                        value
                    )

        # ----------------------------------------------------
        # Structured pelletizing
        # ----------------------------------------------------

        elif (
            step_type
            == "pelletizing"
        ):

            value = step.get(
                "pressure_mpa"
            )

            if finite(
                value
            ):

                out[
                    "synth_pellet_pressure_mpa"
                ] = float(
                    value
                )

        # ----------------------------------------------------
        # Structured quenching
        # ----------------------------------------------------

        elif (
            step_type
            == "quenching"
        ):

            out[
                "synth_quenched"
            ] = 1.0

            medium = step.get(
                "quenching_medium"
            )

            if medium:

                out[
                    "synth_quench_medium"
                ] = str(
                    medium
                ).lower()

        # ----------------------------------------------------
        # Legacy/free-text OSES synthesis metadata
        # ----------------------------------------------------

        text = " ".join(
            str(
                step.get(
                    field
                )
                or ""
            )

            for field in (
                "notes",
                "description",
            )
        )

        text_lower = (
            text.lower()
        )

        structured_temp = (
            step_type
            == "heat_treatment"
            and finite(
                step.get(
                    "max_temp_c"
                )
            )
        )

        structured_hold = (
            step_type
            == "heat_treatment"
            and finite(
                step.get(
                    "hold_time_hours"
                )
            )
        )

        structured_atmosphere = (
            step_type
            == "heat_treatment"
            and bool(
                step.get(
                    "atmosphere"
                )
            )
        )

        # ----------------------------------------------------
        # Temperature fallback
        # ----------------------------------------------------

        if not structured_temp:

            temp_match = re.search(
                r"\bat\s+"
                r"([0-9]+(?:\.[0-9]+)?)"
                r"\s*°?\s*c\b",
                text_lower,
            )

            if temp_match:

                heat_temps.append(
                    float(
                        temp_match.group(
                            1
                        )
                    )
                )

        # ----------------------------------------------------
        # Hold-time fallback
        # ----------------------------------------------------

        if not structured_hold:

            hold_match = re.search(
                r"\bheld\s+for\s+"
                r"([0-9]+(?:\.[0-9]+)?)"
                r"\s*(?:hours?|hrs?|h)\b",
                text_lower,
            )

            if hold_match:

                heat_times.append(
                    float(
                        hold_match.group(
                            1
                        )
                    )
                )

        # ----------------------------------------------------
        # Atmosphere fallback
        # ----------------------------------------------------

        if (
            "synth_atmosphere"
            not in out
            and not structured_atmosphere
        ):

            if (
                "argon"
                in text_lower
            ):

                out[
                    "synth_atmosphere"
                ] = "ar"

            elif re.search(
                r"\bin air\b",
                text_lower,
            ):

                out[
                    "synth_atmosphere"
                ] = "air"

        # ----------------------------------------------------
        # Quench fallback
        # ----------------------------------------------------

        if (
            "quench"
            in text_lower
        ):

            out[
                "synth_quenched"
            ] = 1.0

            if (
                "air quenched"
                in text_lower
            ):

                out[
                    "synth_quench_medium"
                ] = "air"

            elif (
                "hot zone"
                in text_lower
            ):

                out[
                    "synth_quench_medium"
                ] = (
                    "rapid_hot_zone_removal"
                )

    # --------------------------------------------------------
    # Aggregate heat-treatment features
    # --------------------------------------------------------

    if heat_temps:

        out[
            "synth_max_temp_c"
        ] = max(
            heat_temps
        )

    if heat_times:

        out[
            "synth_total_hold_time_hours"
        ] = sum(
            heat_times
        )

    out[
        "synth_step_count"
    ] = len(
        steps
    )

    for step_type in set(
        step_types
    ):

        out[
            f"synth_has_step__{step_type}"
        ] = 1.0

    # --------------------------------------------------------
    # Experimentally supported OSES route classification
    # --------------------------------------------------------

    atmosphere = out.get(
        "synth_atmosphere"
    )

    quench_medium = out.get(
        "synth_quench_medium"
    )

    if (
        atmosphere == "air"
        and quench_medium == "air"
    ):

        out[
            "synth_route"
        ] = "air"

    elif (
        atmosphere == "ar"
        and quench_medium
        == "rapid_hot_zone_removal"
    ):

        out[
            "synth_route"
        ] = "ar"

    return out


# ============================================================
# Ensure output directory exists
# ============================================================

TRAINING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Materials
# ============================================================

materials = {}

for record in read_jsonl(
    MATERIALS
):

    material_auid = (
        record.get(
            "material_auid"
        )
    )

    if material_auid:

        materials[
            material_auid
        ] = record


# ============================================================
# Experiments
#
# Loaded early so we can derive the actual OSES experimental
# element domain from the same cation-only representation.
# ============================================================

experiments = list(
    read_jsonl(
        EXPERIMENTS
    )
)


# ============================================================
# Experimental element domain
# ============================================================

training_element_domain = set()

for experiment in experiments:

    material_auid = (
        experiment.get(
            "material_auid"
        )
    )

    material_record = (
        materials.get(
            material_auid,
            {},
        )
    )

    cations = (
        experiment_cation_elements(
            experiment,
            material_record,
        )
    )

    training_element_domain.update(
        cations
    )


# ============================================================
# Composition -> material index
#
# IMPORTANT:
# composition-level DFT/literature joins also use the same
# cation-only representation.
# ============================================================

materials_by_composition = defaultdict(
    list
)

material_composition_keys = {}

material_cation_map = {}

for (
    material_auid,
    record,
) in materials.items():

    cations = material_cation_elements(
        record
    )

    material_cation_map[
        material_auid
    ] = cations

    key = normalized_composition_key(
        cations
    )

    if not key:
        continue

    material_composition_keys[
        material_auid
    ] = key

    materials_by_composition[
        key
    ].append(
        material_auid
    )


# ============================================================
# Computational aggregates
# ============================================================

comp = defaultdict(
    lambda: {
        "count": 0,
        "source_classes": Counter(),
        "loop_formation": [],
        "xint_hf": [],
        "efa": [],
        "crystal_system": [],
        "spacegroup": [],
    }
)

for record in read_jsonl(
    COMPUTATIONAL
):

    material_auid = (
        record.get(
            "material_auid"
        )
    )

    if not material_auid:
        continue

    agg = comp[
        material_auid
    ]

    agg[
        "count"
    ] += 1

    source_class = (
        record.get(
            "source_class"
        )
    )

    if source_class:

        agg[
            "source_classes"
        ][
            str(
                source_class
            )
        ] += 1

    loop = (
        record.get(
            "loop_dft"
        )
        or {}
    )

    formation = (
        loop.get(
            "dft_formation_energy_ev"
        )
    )

    if finite(
        formation
    ):

        agg[
            "loop_formation"
        ].append(
            float(
                formation
            )
        )

    xint = (
        record.get(
            "xint"
        )
        or {}
    )

    hf = xint.get(
        "enthalpy_formation_atom"
    )

    if finite(
        hf
    ):

        agg[
            "xint_hf"
        ].append(
            float(
                hf
            )
        )

    crystal = xint.get(
        "crystal_system"
    )

    if crystal:

        agg[
            "crystal_system"
        ].append(
            str(
                crystal
            )
        )

    sg = xint.get(
        "spacegroup_relax"
    )

    if sg is not None:

        agg[
            "spacegroup"
        ].append(
            str(
                sg
            )
        )

    derived = (
        record.get(
            "derived"
        )
        or {}
    )

    efa = derived.get(
        "efa"
    )

    if finite(
        efa
    ):

        agg[
            "efa"
        ].append(
            float(
                efa
            )
        )


# ============================================================
# Composition-level computational index
# ============================================================

comp_by_composition = defaultdict(
    lambda: {
        "material_count": 0,
        "record_count": 0,
        "source_classes": Counter(),
        "loop_formation": [],
        "xint_hf": [],
        "efa": [],
        "crystal_system": [],
        "spacegroup": [],
    }
)

for (
    composition,
    material_ids,
) in materials_by_composition.items():

    agg = comp_by_composition[
        composition
    ]

    seen_materials = 0

    for material_auid in (
        material_ids
    ):

        c = comp.get(
            material_auid
        )

        if not c:
            continue

        seen_materials += 1

        agg[
            "record_count"
        ] += c[
            "count"
        ]

        agg[
            "source_classes"
        ].update(
            c[
                "source_classes"
            ]
        )

        agg[
            "loop_formation"
        ].extend(
            c[
                "loop_formation"
            ]
        )

        agg[
            "xint_hf"
        ].extend(
            c[
                "xint_hf"
            ]
        )

        agg[
            "efa"
        ].extend(
            c[
                "efa"
            ]
        )

        agg[
            "crystal_system"
        ].extend(
            c[
                "crystal_system"
            ]
        )

        agg[
            "spacegroup"
        ].extend(
            c[
                "spacegroup"
            ]
        )

    agg[
        "material_count"
    ] = seen_materials


# ============================================================
# Literature aggregates
# ============================================================

lit = defaultdict(
    lambda: {
        "count": 0,
        "successful": 0,
        "structure_family": Counter(),
        "temperatures": [],
        "hold_times": [],
        "atmospheres": Counter(),
    }
)

for record in read_jsonl(
    LITERATURE
):

    material_auid = (
        record.get(
            "material_auid"
        )
    )

    if not material_auid:
        continue

    agg = lit[
        material_auid
    ]

    agg[
        "count"
    ] += 1

    if (
        record.get(
            "synthesis_successful"
        )
        is True
    ):

        agg[
            "successful"
        ] += 1

    family = record.get(
        "structure_family"
    )

    if family:

        agg[
            "structure_family"
        ][
            str(
                family
            )
        ] += 1

    for step in (
        record.get(
            "synthesis_steps"
        )
        or []
    ):

        if not isinstance(
            step,
            dict,
        ):
            continue

        if (
            step.get(
                "step_type"
            )
            == "heat_treatment"
        ):

            temp = step.get(
                "max_temp_c"
            )

            hold = step.get(
                "hold_time_hours"
            )

            atmosphere = (
                step.get(
                    "atmosphere"
                )
            )

            if finite(
                temp
            ):

                agg[
                    "temperatures"
                ].append(
                    float(
                        temp
                    )
                )

            if finite(
                hold
            ):

                agg[
                    "hold_times"
                ].append(
                    float(
                        hold
                    )
                )

            if atmosphere:

                agg[
                    "atmospheres"
                ][
                    str(
                        atmosphere
                    ).lower()
                ] += 1


# ============================================================
# Composition-level literature index
# ============================================================

lit_by_composition = defaultdict(
    lambda: {
        "material_count": 0,
        "count": 0,
        "successful": 0,
        "structure_family": Counter(),
        "temperatures": [],
        "hold_times": [],
        "atmospheres": Counter(),
    }
)

for (
    composition,
    material_ids,
) in materials_by_composition.items():

    agg = lit_by_composition[
        composition
    ]

    seen_materials = 0

    for material_auid in (
        material_ids
    ):

        l = lit.get(
            material_auid
        )

        if not l:
            continue

        seen_materials += 1

        agg[
            "count"
        ] += l[
            "count"
        ]

        agg[
            "successful"
        ] += l[
            "successful"
        ]

        agg[
            "structure_family"
        ].update(
            l[
                "structure_family"
            ]
        )

        agg[
            "temperatures"
        ].extend(
            l[
                "temperatures"
            ]
        )

        agg[
            "hold_times"
        ].extend(
            l[
                "hold_times"
            ]
        )

        agg[
            "atmospheres"
        ].update(
            l[
                "atmospheres"
            ]
        )

    agg[
        "material_count"
    ] = seen_materials


# ============================================================
# Feature constructor
# ============================================================

def material_features(
    material_auid,
    composition=None,
    model_elements=None,
):
    features = {}

    material_record = (
        materials.get(
            material_auid,
            {},
        )
    )

    if model_elements is None:

        model_elements = (
            material_cation_map.get(
                material_auid,
                {},
            )
        )

    model_elements = cation_only_elements(
        model_elements
    )

    # --------------------------------------------------------
    # Composition features
    # --------------------------------------------------------

    features.update(
        element_features(
            material_record,
            model_elements,
        )
    )

    features.update(
        chemistry_features(
            model_elements
        )
    )

    # --------------------------------------------------------
    # Exact-material computational evidence
    # --------------------------------------------------------

    c = comp.get(
        material_auid
    )

    if c:

        features[
            "comp_record_count"
        ] = c[
            "count"
        ]

        for (
            source,
            n,
        ) in c[
            "source_classes"
        ].items():

            features[
                f"comp_source_count__{source}"
            ] = n

        features.update(
            stats(
                c[
                    "loop_formation"
                ],
                "comp_loop_formation_energy",
            )
        )

        features.update(
            stats(
                c[
                    "xint_hf"
                ],
                "comp_xint_hf",
            )
        )

        features.update(
            stats(
                c[
                    "efa"
                ],
                "comp_efa",
            )
        )

        crystal = mode(
            c[
                "crystal_system"
            ]
        )

        if crystal:

            features[
                "comp_crystal_system_mode"
            ] = crystal

        sg = mode(
            c[
                "spacegroup"
            ]
        )

        if sg:

            features[
                "comp_spacegroup_mode"
            ] = sg

    # --------------------------------------------------------
    # Exact-material literature evidence
    # --------------------------------------------------------

    l = lit.get(
        material_auid
    )

    if l:

        features[
            "lit_report_count"
        ] = l[
            "count"
        ]

        features[
            "lit_success_count"
        ] = l[
            "successful"
        ]

        features.update(
            stats(
                l[
                    "temperatures"
                ],
                "lit_heat_temp_c",
            )
        )

        features.update(
            stats(
                l[
                    "hold_times"
                ],
                "lit_hold_time_hours",
            )
        )

        for (
            family,
            n,
        ) in l[
            "structure_family"
        ].items():

            features[
                f"lit_structure_count__{family}"
            ] = n

        for (
            atmosphere,
            n,
        ) in l[
            "atmospheres"
        ].items():

            features[
                f"lit_atmosphere_count__{atmosphere}"
            ] = n

    # --------------------------------------------------------
    # Composition-equivalent evidence
    # --------------------------------------------------------

    if composition:

        c_equiv = (
            comp_by_composition.get(
                composition
            )
        )

        if (
            c_equiv
            and c_equiv[
                "record_count"
            ] > 0
        ):

            features[
                "equiv_comp_material_count"
            ] = c_equiv[
                "material_count"
            ]

            features[
                "equiv_comp_record_count"
            ] = c_equiv[
                "record_count"
            ]

            for (
                source,
                n,
            ) in c_equiv[
                "source_classes"
            ].items():

                features[
                    f"equiv_comp_source_count__{source}"
                ] = n

            features.update(
                stats(
                    c_equiv[
                        "loop_formation"
                    ],
                    "equiv_comp_loop_formation_energy",
                )
            )

            features.update(
                stats(
                    c_equiv[
                        "xint_hf"
                    ],
                    "equiv_comp_xint_hf",
                )
            )

            features.update(
                stats(
                    c_equiv[
                        "efa"
                    ],
                    "equiv_comp_efa",
                )
            )

            crystal = mode(
                c_equiv[
                    "crystal_system"
                ]
            )

            if crystal:

                features[
                    "equiv_comp_crystal_system_mode"
                ] = crystal

            sg = mode(
                c_equiv[
                    "spacegroup"
                ]
            )

            if sg:

                features[
                    "equiv_comp_spacegroup_mode"
                ] = sg

        l_equiv = (
            lit_by_composition.get(
                composition
            )
        )

        if (
            l_equiv
            and l_equiv[
                "count"
            ] > 0
        ):

            features[
                "equiv_lit_material_count"
            ] = l_equiv[
                "material_count"
            ]

            features[
                "equiv_lit_report_count"
            ] = l_equiv[
                "count"
            ]

            features[
                "equiv_lit_success_count"
            ] = l_equiv[
                "successful"
            ]

            features.update(
                stats(
                    l_equiv[
                        "temperatures"
                    ],
                    "equiv_lit_heat_temp_c",
                )
            )

            features.update(
                stats(
                    l_equiv[
                        "hold_times"
                    ],
                    "equiv_lit_hold_time_hours",
                )
            )

            for (
                family,
                n,
            ) in l_equiv[
                "structure_family"
            ].items():

                features[
                    f"equiv_lit_structure_count__{family}"
                ] = n

            for (
                atmosphere,
                n,
            ) in l_equiv[
                "atmospheres"
            ].items():

                features[
                    f"equiv_lit_atmosphere_count__{atmosphere}"
                ] = n

    return features


# ============================================================
# Experiments / supervised table
# ============================================================

experiment_materials = set()

training_row_count = 0

with TRAINING_OUT.open(
    "w"
) as out:

    for experiment in experiments:

        material_auid = (
            experiment.get(
                "material_auid"
            )
        )

        if not material_auid:
            continue

        experiment_materials.add(
            material_auid
        )

        material_record = (
            materials.get(
                material_auid,
                {},
            )
        )

        model_elements = (
            experiment_cation_elements(
                experiment,
                material_record,
            )
        )

        comp_key = normalized_composition_key(
            model_elements
        )

        if not comp_key:

            comp_key = composition_key(
                experiment,
                material_record,
            )

        features = material_features(
            material_auid,
            composition=comp_key,
            model_elements=model_elements,
        )

        features.update(
            synthesis_features(
                experiment
            )
        )

        record = {
            "oses_id":
                experiment.get(
                    "oses_id"
                ),

            "material_auid":
                material_auid,

            "recipe_auid":
                experiment.get(
                    "recipe_auid"
                ),

            "composition_key":
                comp_key,

            "model_cation_elements":
                model_elements,

            "in_training_element_domain":
                True,

            "unseen_elements":
                [],

            "confidence":
                experiment.get(
                    "confidence"
                ),

            "features":
                features,

            "targets": {
                "rock_salt_present":
                    int(
                        bool(
                            experiment.get(
                                "rock_salt_present"
                            )
                        )
                    ),

                "spinel_present":
                    int(
                        bool(
                            experiment.get(
                                "spinel_present"
                            )
                        )
                    ),

                "other_phase_present":
                    int(
                        bool(
                            experiment.get(
                                "other_phase_present"
                            )
                        )
                    ),
            },
        }

        write_jsonl(
            out,
            record,
        )

        training_row_count += 1


# ============================================================
# Candidate feature table
#
# One row per LOOP material.
#
# Candidate model chemistry is cation-only.
#
# The original full oxide elements are retained separately for
# provenance.
#
# Candidate rows do not receive synth_route here. Prediction
# injects Air and/or Ar later.
# ============================================================

candidate_row_count = 0

candidate_in_domain_count = 0
candidate_out_of_domain_count = 0

candidate_unseen_element_counts = Counter()

with CANDIDATE_OUT.open(
    "w"
) as out:

    for material_auid in sorted(
        materials
    ):

        material_record = (
            materials[
                material_auid
            ]
        )

        material_payload = (
            material_record.get(
                "material"
            )
            or {}
        )

        full_elements = (
            material_payload.get(
                "elements"
            )
            or {}
        )

        model_elements = (
            material_cation_map.get(
                material_auid,
                {},
            )
        )

        composition = (
            material_composition_keys.get(
                material_auid
            )
        )

        candidate_elements = set(
            model_elements
        )

        unseen_elements = sorted(
            candidate_elements
            - training_element_domain
        )

        in_training_element_domain = (
            len(
                unseen_elements
            )
            == 0
        )

        if in_training_element_domain:

            candidate_in_domain_count += 1

        else:

            candidate_out_of_domain_count += 1

            candidate_unseen_element_counts.update(
                unseen_elements
            )

        record = {
            "material_auid":
                material_auid,

            "composition_key":
                composition,

            # Model-facing normalized cation composition.
            "model_cation_elements":
                model_elements,

            # Full LOOP material composition retained for
            # provenance.
            "full_material_elements":
                full_elements,

            "training_element_domain":
                sorted(
                    training_element_domain
                ),

            "in_training_element_domain":
                in_training_element_domain,

            "unseen_elements":
                unseen_elements,

            "has_oses_experiment":
                (
                    material_auid
                    in experiment_materials
                ),

            "features":
                material_features(
                    material_auid,
                    composition=composition,
                    model_elements=model_elements,
                ),
        }

        write_jsonl(
            out,
            record,
        )

        candidate_row_count += 1


# ============================================================
# Summary / validation
# ============================================================

training_route_counts = Counter()

training_chemistry_coverage = 0
training_computational_coverage = 0
training_literature_coverage = 0

training_element_feature_counts = Counter()

for record in read_jsonl(
    TRAINING_OUT
):

    features = (
        record.get(
            "features"
        )
        or {}
    )

    route = features.get(
        "synth_route"
    )

    if route:

        training_route_counts[
            route
        ] += 1

    if (
        "chem_num_elements"
        in features
    ):

        training_chemistry_coverage += 1

    if (
        "comp_record_count"
        in features
        or "equiv_comp_record_count"
        in features
    ):

        training_computational_coverage += 1

    if (
        "lit_report_count"
        in features
        or "equiv_lit_report_count"
        in features
    ):

        training_literature_coverage += 1

    for key, value in (
        features.items()
    ):

        if (
            key.startswith(
                "element_fraction__"
            )
            and finite(
                value
            )
            and value > 0
        ):

            element = key.split(
                "__",
                1,
            )[1]

            training_element_feature_counts[
                element
            ] += 1


summary = {
    "materials":
        len(
            materials
        ),

    "materials_with_computation":
        len(
            comp
        ),

    "materials_with_literature":
        len(
            lit
        ),

    "unique_material_cation_compositions":
        len(
            materials_by_composition
        ),

    "composition_groups_with_computation":
        sum(
            1

            for value
            in comp_by_composition.values()

            if value[
                "record_count"
            ] > 0
        ),

    "composition_groups_with_literature":
        sum(
            1

            for value
            in lit_by_composition.values()

            if value[
                "count"
            ] > 0
        ),

    "oses_experiments":
        len(
            experiments
        ),

    "training_rows":
        training_row_count,

    "unique_experimental_materials":
        len(
            experiment_materials
        ),

    "candidate_rows":
        candidate_row_count,

    "composition_representation": {
        "model_basis":
            "cation_only",

        "excluded_from_model_composition": [
            "O",
        ],

        "normalization":
            "remaining cation fractions sum to 1",
    },

    "training_element_domain":
        sorted(
            training_element_domain
        ),

    "training_element_coverage":
        dict(
            sorted(
                training_element_feature_counts.items()
            )
        ),

    "candidate_domain_coverage": {
        "in_training_element_domain":
            candidate_in_domain_count,

        "out_of_training_element_domain":
            candidate_out_of_domain_count,

        "unseen_element_counts":
            dict(
                sorted(
                    candidate_unseen_element_counts.items()
                )
            ),
    },

    "training_feature_coverage": {
        "chemistry":
            training_chemistry_coverage,

        "computational":
            training_computational_coverage,

        "literature":
            training_literature_coverage,
    },

    "synthesis_routes": {
        "air":
            training_route_counts.get(
                "air",
                0,
            ),

        "ar":
            training_route_counts.get(
                "ar",
                0,
            ),

        "unknown":
            (
                training_row_count
                - sum(
                    training_route_counts.values()
                )
            ),
    },
}


with SUMMARY_OUT.open(
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print(
    json.dumps(
        summary,
        indent=2,
    )
)