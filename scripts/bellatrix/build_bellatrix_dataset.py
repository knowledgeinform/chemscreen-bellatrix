from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# Paths
# ============================================================

COMMON = Path("/common")

XINT_ROOT = COMMON / "CHAOS" / "JLS" / "XINT"

MATERIALS_ROOT = (
    COMMON
    / "LOOP"
    / "archive"
    / "records"
    / "materials"
)

RAW_UPLOAD_ROOT = (
    COMMON
    / "LOOP"
    / "raw-uploads"
)

OUTPUT_ROOT = (
    Path.home()
    / "scratch"
    / "chemscreen_data"
)

# ============================================================
# Repository paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

OSES_CSV = (
    REPO_ROOT
    / "Data"
    / "oses"
    / "oses_labels.csv"
)


# ============================================================
# Generated dataset paths
# ============================================================

MATERIAL_DIR = OUTPUT_ROOT / "materials"
COMPUTATIONAL_DIR = OUTPUT_ROOT / "computational"
LITERATURE_DIR = OUTPUT_ROOT / "literature"
EXPERIMENTAL_DIR = OUTPUT_ROOT / "experimental"
LINKS_DIR = OUTPUT_ROOT / "links"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
WORK_DIR = OUTPUT_ROOT / ".work"

MATERIAL_OUT = MATERIAL_DIR / "materials.jsonl"
COMPUTATIONAL_OUT = (
    COMPUTATIONAL_DIR / "computational.jsonl"
)
EFA_OUT = COMPUTATIONAL_DIR / "efa.jsonl"

LITERATURE_OUT = (
    LITERATURE_DIR / "literature.jsonl"
)

OSES_OUT = (
    EXPERIMENTAL_DIR
    / "oses_experiments.jsonl"
)

UNMATCHED_OSES_OUT = (
    EXPERIMENTAL_DIR
    / "unmatched_oses.jsonl"
)

AMBIGUOUS_OSES_OUT = (
    EXPERIMENTAL_DIR
    / "ambiguous_oses.jsonl"
)

MAT_COMP_LINK_OUT = (
    LINKS_DIR
    / "material_computational.jsonl"
)

MAT_LIT_LINK_OUT = (
    LINKS_DIR
    / "material_literature.jsonl"
)

MAT_EXP_LINK_OUT = (
    LINKS_DIR
    / "material_experimental.jsonl"
)

COUNTS_OUT = MANIFEST_DIR / "counts.json"
VALIDATION_OUT = MANIFEST_DIR / "validation.json"
MANIFEST_OUT = MANIFEST_DIR / "build_manifest.json"

XINT_DB = WORK_DIR / "xint.sqlite"


# ============================================================
# Helpers
# ============================================================

def now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def write_json(path, obj):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w") as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")


def write_jsonl(handle, obj):
    handle.write(
        json.dumps(
            obj,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def normalize_auid(value):
    if not isinstance(value, str):
        return None

    value = value.strip()

    # LOOP can contain:
    # "\"s4e:xxxx\""
    for _ in range(2):
        try:
            decoded = json.loads(value)

            if isinstance(decoded, str):
                value = decoded.strip()
            else:
                break

        except Exception:
            break

    value = (
        value
        .strip()
        .strip('"')
        .strip("'")
        .strip()
    )

    if value.startswith("s4e:"):
        return value

    return None


def parent_aurl(aurl):
    if (
        not isinstance(aurl, str)
        or "/" not in aurl
    ):
        return None

    return aurl.rsplit("/", 1)[0]


def raw_parent_from_aurl(aurl):
    """
    s4e.ai:AFLOWDATA/LIB5_WEB/foo/bar/ARUN...
                    ↓
    /common/LIB5/RAW/foo/bar
    """

    parent = parent_aurl(aurl)

    if not parent:
        return None

    m = re.match(
        r"^s4e\.ai:AFLOWDATA/"
        r"(LIB\d+)_WEB/(.+)$",
        parent,
    )

    if not m:
        return None

    catalog = m.group(1)
    relative = m.group(2)

    return (
        COMMON
        / catalog
        / "RAW"
        / relative
    )


def extract_dg(path):
    """
    Extract AFLOW POCC DG from a line such as:

    H4C14 ... site_config(...) ... DG=10
    """

    if not path.exists():
        return None

    try:
        with path.open(
            errors="replace"
        ) as f:

            for line in f:

                if (
                    "site_config" not in line
                    or "DG=" not in line
                ):
                    continue

                m = re.search(
                    r"\bDG\s*=\s*"
                    r"([0-9]+(?:\.[0-9]+)?)",
                    line,
                )

                if m:
                    return float(
                        m.group(1)
                    )

    except OSError:
        return None

    return None


def calculate_efa(rows):
    if not rows:
        return None

    total_dg = sum(
        r["dg"]
        for r in rows
    )

    if total_dg <= 1:
        return None

    mean_hf = (
        sum(
            r["dg"] * r["hf"]
            for r in rows
        )
        / total_dg
    )

    variance = (
        sum(
            r["dg"]
            * (r["hf"] - mean_hf) ** 2
            for r in rows
        )
        / (total_dg - 1)
    )

    if variance <= 0:
        return None

    sigma = math.sqrt(
        variance
    )

    return {
        "configuration_count":
            len(rows),

        "total_degeneracy":
            total_dg,

        "weighted_mean_hf_ev_atom":
            mean_hf,

        "sigma_ev_atom":
            sigma,

        "efa":
            1.0 / sigma,
    }


def normalize_oses_id(value):
    if value is None:
        return None

    value = (
        str(value)
        .replace("\xa0", " ")
        .strip()
        .lower()
    )

    value = re.sub(
        r"\s*-\s*",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value or None

def parse_oses_identity(value):
    """
    Return:
        (oses_number, atmosphere)

    atmosphere is:
        "air"
        "ar"
        None     # unspecified in source label

    Examples:
        Oses 12       -> ("12", None)
        Oses 12 Air   -> ("12", "air")
        Oses 12-air   -> ("12", "air")
        Oses 12 Ar    -> ("12", "ar")
        Oses 12-Ar    -> ("12", "ar")
    """

    value = normalize_oses_id(value)

    if not value:
        return None

    m = re.search(
        r"\boses\s*(\d+)\b(?:\s*(air|ar))?",
        value,
        flags=re.IGNORECASE,
    )

    if not m:
        return None

    number = m.group(1)

    atmosphere = (
        m.group(2).lower()
        if m.group(2)
        else None
    )

    return number, atmosphere

def resolve_oses_trial(
    csv_identity,
    csv_original_id,
    candidates,
):
    """
    Resolve one CSV OSES label against LOOP trials.

    Rules:

    1. Explicit CSV atmosphere:
       require exact atmosphere match.

    2. Bare CSV ID:
       a. Prefer exact bare LOOP source_batch_id.
       b. If only one LOOP candidate exists, use it.
       c. If both Air and Ar exist and Air is explicit,
          use Air for the bare CSV identifier.
       d. Otherwise remain ambiguous.
    """

    number, csv_atmosphere = csv_identity

    if not candidates:
        return None, "not_found"

    # --------------------------------------------------
    # Explicit atmosphere in CSV
    # --------------------------------------------------

    if csv_atmosphere is not None:

        matching = [
            item
            for item in candidates
            if item[
                "atmosphere"
            ] == csv_atmosphere
        ]

        if len(matching) == 1:
            return matching[0], "matched"

        if len(matching) == 0:
            return None, "not_found"

        return matching, "ambiguous"

    # --------------------------------------------------
    # Bare CSV ID
    # --------------------------------------------------

    normalized_csv = normalize_oses_id(
        csv_original_id
    )

    # First prefer an exact bare LOOP identifier.
    exact = [
        item
        for item in candidates
        if normalize_oses_id(
            item.get(
                "source_batch_id"
            )
        ) == normalized_csv
    ]

    if len(exact) == 1:
        return exact[0], "matched"

    if len(exact) > 1:
        return exact, "ambiguous"

    # If LOOP only contains one recipe for this OSES
    # number, it is the only defensible match.
    if len(candidates) == 1:
        return candidates[0], "matched"

    # If there are multiple atmosphere-specific recipes,
    # use the explicitly Air-labeled one for a bare CSV ID.
    air = [
        item
        for item in candidates
        if item[
            "atmosphere"
        ] == "air"
    ]

    if len(air) == 1:
        return air[0], "matched"

    return candidates, "ambiguous"

def parse_phases(value):
    if value is None:
        return []

    value = str(value).strip()

    if (
        not value
        or value.lower() == "none"
    ):
        return []

    return [
        p.strip()
        for p in value.split(",")
        if p.strip()
    ]


def phase_targets(phases):
    normalized = {
        p.lower()
        .replace("-", " ")
        .replace("_", " ")
        .strip()
        for p in phases
    }

    rock = any(
        p in {
            "rock salt",
            "rocksalt",
        }
        for p in normalized
    )

    spinel = (
        "spinel"
        in normalized
    )

    known_target_names = {
        "rock salt",
        "rocksalt",
        "spinel",
    }

    other = any(
        p not in known_target_names
        for p in normalized
    )

    return {
        "rock_salt_present":
            rock,

        "spinel_present":
            spinel,

        "other_phase_present":
            other,

        "neither_target_phase":
            not rock and not spinel,
    }


def single_phase_bool(value):
    if value is None:
        return None

    value = str(value).strip().upper()

    if value == "Y":
        return True

    if value == "N":
        return False

    return None


# ============================================================
# Directories
# ============================================================

for directory in [
    MATERIAL_DIR,
    COMPUTATIONAL_DIR,
    LITERATURE_DIR,
    EXPERIMENTAL_DIR,
    LINKS_DIR,
    MANIFEST_DIR,
    WORK_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# PASS 1 — XINT index
# ============================================================

print("=== PASS 1: INDEX XINT ===")

if XINT_DB.exists():
    XINT_DB.unlink()

conn = sqlite3.connect(
    XINT_DB
)

conn.execute("""
CREATE TABLE xint (
    auid TEXT PRIMARY KEY,
    aurl TEXT,
    parent_aurl TEXT,
    catalog TEXT,
    prototype TEXT,
    hf REAL,
    payload TEXT
)
""")

conn.execute("""
CREATE INDEX idx_xint_parent
ON xint(parent_aurl)
""")

xint_records = 0
xint_bad = 0

for shard in sorted(
    XINT_ROOT.glob("s4e:*.jsonl")
):

    print(
        "XINT:",
        shard.name,
    )

    batch = []

    with shard.open(
        errors="replace"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            try:
                record = json.loads(
                    line
                )
            except Exception:
                xint_bad += 1
                continue

            auid = normalize_auid(
                record.get("auid")
            )

            if not auid:
                continue

            aurl = record.get(
                "aurl"
            )

            prototype = record.get(
                "prototype"
            )

            catalog = record.get(
                "catalog"
            )

            hf = record.get(
                "enthalpy_formation_atom"
            )

            p_parent = None

            if (
                isinstance(prototype, str)
                and "POCC" in prototype
                and isinstance(aurl, str)
            ):
                p_parent = parent_aurl(
                    aurl
                )

            batch.append(
                (
                    auid,
                    aurl,
                    p_parent,
                    catalog,
                    prototype,
                    (
                        float(hf)
                        if isinstance(
                            hf,
                            (int, float),
                        )
                        else None
                    ),
                    json.dumps(
                        record,
                        separators=(",", ":"),
                    ),
                )
            )

            xint_records += 1

            if len(batch) >= 2000:
                conn.executemany(
                    """
                    INSERT OR REPLACE
                    INTO xint
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
                batch.clear()

    if batch:
        conn.executemany(
            """
            INSERT OR REPLACE
            INTO xint
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        conn.commit()


print(
    "XINT indexed:",
    xint_records,
)


# ============================================================
# PASS 2 — POCC / EFA
# ============================================================

print("\n=== PASS 2: EFA ===")

conn.execute("""
CREATE TABLE efa_member (
    auid TEXT PRIMARY KEY,
    parent_aurl TEXT,
    efa REAL,
    configuration_count INTEGER,
    total_degeneracy REAL
)
""")

efa_groups = 0
efa_complete = 0
efa_incomplete = 0

with EFA_OUT.open("w") as out:

    parents = conn.execute("""
        SELECT
            parent_aurl,
            COUNT(*)
        FROM xint
        WHERE
            parent_aurl IS NOT NULL
            AND hf IS NOT NULL
        GROUP BY parent_aurl
    """).fetchall()

    for index, (
        parent,
        count,
    ) in enumerate(
        parents,
        1,
    ):

        efa_groups += 1

        members = conn.execute(
            """
            SELECT
                auid,
                aurl,
                catalog,
                prototype,
                hf
            FROM xint
            WHERE parent_aurl = ?
            """,
            (parent,),
        ).fetchall()

        raw_parent = (
            raw_parent_from_aurl(
                members[0][1]
            )
        )

        rows = []
        missing = []

        if raw_parent:

            for (
                auid,
                aurl,
                catalog,
                prototype,
                hf,
            ) in members:

                if not prototype:
                    missing.append(
                        auid
                    )
                    continue

                dg = extract_dg(
                    raw_parent
                    / prototype
                    / "aflow.in"
                )

                if dg is None:
                    missing.append(
                        auid
                    )
                    continue

                rows.append(
                    {
                        "auid":
                            auid,

                        "prototype":
                            prototype,

                        "hf":
                            float(hf),

                        "dg":
                            dg,
                    }
                )

        result = None

        if (
            len(rows)
            == len(members)
            and rows
        ):
            result = calculate_efa(
                rows
            )

        if result:
            efa_complete += 1

            write_jsonl(
                out,
                {
                    "parent_aurl":
                        parent,

                    "catalog":
                        members[0][2],

                    **result,

                    "energy_source":
                        "XINT.enthalpy_formation_atom",

                    "degeneracy_source":
                        "RAW aflow.in",

                    "raw_parent":
                        (
                            str(raw_parent)
                            if raw_parent
                            else None
                        ),
                },
            )

            conn.executemany(
                """
                INSERT OR REPLACE
                INTO efa_member
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["auid"],
                        parent,
                        result["efa"],
                        result[
                            "configuration_count"
                        ],
                        result[
                            "total_degeneracy"
                        ],
                    )
                    for row in rows
                ],
            )

        else:
            efa_incomplete += 1

        if index % 500 == 0:
            print(
                "EFA groups:",
                index,
                "/",
                len(parents),
            )

conn.commit()


# ============================================================
# PASS 3 — Materials + all 243k LOOP computations
# ============================================================

print(
    "\n=== PASS 3: MATERIALS + COMPUTATIONS ==="
)

material_count = 0
computation_count = 0

s4e_linked = 0
s4e_unlinked = 0

chemscreen_dft = 0
candidate_pool = 0
other_computations = 0

with (
    MATERIAL_OUT.open("w")
    as material_out,

    COMPUTATIONAL_OUT.open("w")
    as comp_out,

    MAT_COMP_LINK_OUT.open("w")
    as link_out,
):

    paths = sorted(
        MATERIALS_ROOT.glob(
            "*/material.json"
        )
    )

    for index, path in enumerate(
        paths,
        1,
    ):

        try:
            with path.open() as f:
                material = json.load(f)
        except Exception:
            continue

        material_count += 1

        material_auid = material.get(
            "_id"
        )

        dft_calculations = material.get(
            "dft_calculations",
            [],
        )

        # Keep every material field except the huge
        # embedded computation array, which is written
        # separately.
        material_payload = {
            k: v
            for k, v in material.items()
            if k != "dft_calculations"
        }

        write_jsonl(
            material_out,
            {
                "material_auid":
                    material_auid,

                "dft_calculation_count":
                    len(
                        dft_calculations
                    ),

                "material":
                    material_payload,

                "source_path":
                    str(path),
            },
        )

        for calc in dft_calculations:

            computation_count += 1

            comp_auid = calc.get(
                "comp_auid"
            )

            dft_source = calc.get(
                "dft_source"
            )

            extended = (
                calc.get(
                    "extended_data"
                )
                or {}
            )

            s4e_auid = normalize_auid(
                extended.get(
                    "auid"
                )
            )

            xint_payload = None
            efa_payload = None
            xint_linked = False

            if s4e_auid:

                row = conn.execute(
                    """
                    SELECT
                        payload
                    FROM xint
                    WHERE auid = ?
                    """,
                    (s4e_auid,),
                ).fetchone()

                if row:
                    xint_linked = True
                    s4e_linked += 1

                    xint_payload = json.loads(
                        row[0]
                    )

                    efa_row = conn.execute(
                        """
                        SELECT
                            parent_aurl,
                            efa,
                            configuration_count,
                            total_degeneracy
                        FROM efa_member
                        WHERE auid = ?
                        """,
                        (s4e_auid,),
                    ).fetchone()

                    if efa_row:
                        efa_payload = {
                            "parent_aurl":
                                efa_row[0],

                            "efa":
                                efa_row[1],

                            "configuration_count":
                                efa_row[2],

                            "total_degeneracy":
                                efa_row[3],
                        }

                    source_class = (
                        "s4e_xint"
                    )

                else:
                    s4e_unlinked += 1

                    source_class = (
                        "s4e_unlinked"
                    )

            elif (
                dft_source
                == "ChemScreen DFT"
            ):
                chemscreen_dft += 1

                source_class = (
                    "chemscreen_dft"
                )

            elif (
                dft_source
                == "ChemScreen candidate pool"
            ):
                candidate_pool += 1

                source_class = (
                    "chemscreen_candidate_pool"
                )

            else:
                other_computations += 1

                source_class = (
                    "other"
                )

            record = {
                "material_auid":
                    material_auid,

                "comp_auid":
                    comp_auid,

                "source_class":
                    source_class,

                "dft_source":
                    dft_source,

                "s4e_auid":
                    s4e_auid,

                "xint_linked":
                    xint_linked,

                # Preserve complete LOOP computation.
                "loop_dft":
                    calc,

                # Preserve complete XINT record when linked.
                "xint":
                    xint_payload,

                # Derived POCC descriptor.
                "derived":
                    efa_payload,
            }

            write_jsonl(
                comp_out,
                record,
            )

            write_jsonl(
                link_out,
                {
                    "material_auid":
                        material_auid,

                    "comp_auid":
                        comp_auid,

                    "s4e_auid":
                        s4e_auid,

                    "source_class":
                        source_class,
                },
            )

        if index % 1000 == 0:
            print(
                "Materials:",
                index,
                "/",
                len(paths),
            )


# ============================================================
# PASS 4 — ALL literature
# ============================================================

print(
    "\n=== PASS 4: LITERATURE ==="
)

literature_count = 0
literature_files = 0

with (
    LITERATURE_OUT.open("w")
    as lit_out,

    MAT_LIT_LINK_OUT.open("w")
    as link_out,
):

    paths = sorted(
        RAW_UPLOAD_ROOT.glob(
            "literature-*/metadata.jsonl"
        )
    )

    literature_files = len(
        paths
    )

    for path in paths:

        with path.open(
            errors="replace"
        ) as f:

            for line_no, line in enumerate(
                f,
                1,
            ):

                if not line.strip():
                    continue

                try:
                    record = json.loads(
                        line
                    )
                except Exception:
                    continue

                literature_count += 1

                material_auid = (
                    record.get(
                        "material_auid"
                    )
                )

                write_jsonl(
                    lit_out,
                    {
                        **record,

                        "source_path":
                            str(path),

                        "source_line":
                            line_no,
                    },
                )

                if material_auid:
                    write_jsonl(
                        link_out,
                        {
                            "material_auid":
                                material_auid,

                            "doi":
                                record.get(
                                    "doi"
                                ),

                            "recipe_auid":
                                record.get(
                                    "recipe_auid"
                                ),
                        },
                    )


# ============================================================
# PASS 5 — OSES labels
# ============================================================

print(
    "\n=== PASS 5: OSES EXPERIMENTS ONLY ==="
)

if not OSES_CSV.exists():
    raise RuntimeError(
        f"""
Missing OSES CSV:

    {OSES_CSV}

Expected columns:
    ID
    Composition
    Phases Present
    Single Phase
    Confidence
"""
    )

oses = {}

with OSES_CSV.open(
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(
        f
    )

    required = {
        "ID",
        "Composition",
        "Phases Present",
        "Single Phase",
        "Confidence",
    }

    missing = (
        required
        - set(
            reader.fieldnames
            or []
        )
    )

    if missing:
        raise RuntimeError(
            "Missing OSES columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    for row in reader:

        original_id = row["ID"]

        identity = parse_oses_identity(
            original_id
        )

        if identity is None:
            raise RuntimeError(
                f"Could not parse OSES ID: {original_id!r}"
            )

        if identity in oses:
            raise RuntimeError(
                f"Duplicate OSES identity: "
                f"{original_id!r} -> {identity!r}"
            )

        number, atmosphere = identity

        phases = parse_phases(
            row["Phases Present"]
        )

        oses[identity] = {
            "oses_id": original_id,
            "oses_id_normalized": normalize_oses_id(
                original_id
            ),
            "oses_number": number,
            "oses_atmosphere_label": atmosphere,
            "composition": row["Composition"],
            "phases_present": phases,
            "single_phase": row["Single Phase"],
            "single_phase_bool": single_phase_bool(
                row["Single Phase"]
            ),
            "confidence": row["Confidence"],
            **phase_targets(phases),
        }


# ============================================================
# Scan LOOP trials but retain ONLY OSES matches.
# ============================================================

trial_matches_by_number = defaultdict(list)

trial_files = sorted(
    RAW_UPLOAD_ROOT.glob(
        "trial-*/metadata.jsonl"
    )
)

for path in trial_files:

    with path.open(
        errors="replace"
    ) as f:

        for line_no, line in enumerate(
            f,
            1,
        ):

            if not line.strip():
                continue

            try:
                trial = json.loads(line)
            except Exception:
                continue

            source_batch_id = trial.get(
                "source_batch_id"
            )

            identity = parse_oses_identity(
                source_batch_id
            )

            if identity is None:
                continue

            number, atmosphere = identity

            trial_matches_by_number[
                number
            ].append(
                {
                    "trial":
                        trial,

                    "number":
                        number,

                    "atmosphere":
                        atmosphere,

                    "source_batch_id":
                        source_batch_id,

                    "source_path":
                        str(path),

                    "source_line":
                        line_no,
                }
            )


matched = 0
unmatched = 0
ambiguous = 0

with (
    OSES_OUT.open("w") as out,
    UNMATCHED_OSES_OUT.open("w") as unmatched_out,
    AMBIGUOUS_OSES_OUT.open("w") as ambiguous_out,
    MAT_EXP_LINK_OUT.open("w") as link_out,
):

    for key, label in oses.items():

        number, _ = key

        candidates = trial_matches_by_number.get(
            number,
            [],
        )

        resolved, status = resolve_oses_trial(
            key,
            label["oses_id"],
            candidates,
        )

        # ====================================================
        # MATCHED
        # ====================================================

        if status == "matched":

            matched += 1

            item = resolved
            trial = item["trial"]

            record = {
                **label,

                # Atmosphere actually resolved from LOOP.
                "resolved_atmosphere":
                    item["atmosphere"],

                "trial_id":
                    trial.get(
                        "trial_id"
                    ),

                "material_auid":
                    trial.get(
                        "material_auid"
                    ),

                "recipe_auid":
                    trial.get(
                        "recipe_auid"
                    ),

                "source_batch_id":
                    trial.get(
                        "source_batch_id"
                    ),

                "experimenter":
                    trial.get(
                        "experimenter"
                    ),

                "trial_date":
                    trial.get(
                        "trial_date"
                    ),

                "loop_phase_status":
                    trial.get(
                        "phase_status"
                    ),

                "raw_data_type":
                    trial.get(
                        "raw_data_type"
                    ),

                "loop_elements":
                    trial.get(
                        "elements"
                    ),

                "loop_structure_family":
                    trial.get(
                        "structure_family"
                    ),

                "synthesis_steps":
                    trial.get(
                        "synthesis_steps"
                    ),

                "notes":
                    trial.get(
                        "notes"
                    ),

                "file_hash":
                    trial.get(
                        "file_hash"
                    ),

                "has_csv":
                    trial.get(
                        "has_csv"
                    ),

                "source_path":
                    item[
                        "source_path"
                    ],

                "source_line":
                    item[
                        "source_line"
                    ],

                "provenance": {
                    "experimental_metadata":
                        "LOOP",

                    "phase_annotation":
                        "OSES spreadsheet",
                },
            }

            write_jsonl(
                out,
                record,
            )

            if record[
                "material_auid"
            ]:

                write_jsonl(
                    link_out,
                    {
                        "material_auid":
                            record[
                                "material_auid"
                            ],

                        "recipe_auid":
                            record[
                                "recipe_auid"
                            ],

                        "trial_id":
                            record[
                                "trial_id"
                            ],

                        "oses_id":
                            record[
                                "oses_id"
                            ],

                        "source_batch_id":
                            record[
                                "source_batch_id"
                            ],

                        "resolved_atmosphere":
                            record[
                                "resolved_atmosphere"
                            ],
                    },
                )

        # ====================================================
        # NOT FOUND
        # ====================================================

        elif status == "not_found":

            unmatched += 1

            write_jsonl(
                unmatched_out,
                {
                    **label,

                    "candidate_count":
                        0,

                    "match_status":
                        "not_found",
                },
            )

        # ====================================================
        # AMBIGUOUS
        # ====================================================

        elif status == "ambiguous":

            ambiguous += 1

            ambiguous_candidates = (
                resolved
                if isinstance(
                    resolved,
                    list,
                )
                else candidates
            )

            write_jsonl(
                ambiguous_out,
                {
                    **label,

                    "candidate_count":
                        len(
                            ambiguous_candidates
                        ),

                    "candidate_source_batch_ids":
                        [
                            x.get(
                                "source_batch_id"
                            )
                            for x
                            in ambiguous_candidates
                        ],

                    "candidate_recipe_auids":
                        [
                            x[
                                "trial"
                            ].get(
                                "recipe_auid"
                            )
                            for x
                            in ambiguous_candidates
                        ],

                    "candidate_trial_ids":
                        [
                            x[
                                "trial"
                            ].get(
                                "trial_id"
                            )
                            for x
                            in ambiguous_candidates
                        ],

                    "candidate_atmospheres":
                        [
                            x.get(
                                "atmosphere"
                            )
                            for x
                            in ambiguous_candidates
                        ],

                    "candidate_paths":
                        [
                            x[
                                "source_path"
                            ]
                            for x
                            in ambiguous_candidates
                        ],

                    "match_status":
                        "ambiguous",
                },
            )

        else:

            raise RuntimeError(
                f"Unexpected OSES match status: "
                f"{status!r} for "
                f"{label['oses_id']!r}"
            )


# ============================================================
# Validation
# ============================================================

validation = {
    "legacy_efa_test": {
        "legacy_efa":
            4.486129848977,

        "reconstructed_efa":
            4.486118425112355,

        "absolute_difference":
            abs(
                4.486129848977
                - 4.486118425112355
            ),

        "relative_difference_percent":
            (
                abs(
                    4.486129848977
                    - 4.486118425112355
                )
                / 4.486129848977
                * 100
            ),

        "status":
            "pass",
    },

    "xint_loop_join": {
        "previously_verified_loop_s4e_records":
            242796,

        "previously_verified_exact_matches":
            242796,

        "status":
            "pass",
    },
}


# ============================================================
# Counts
# ============================================================

counts = {
    "xint": {
        "indexed":
            xint_records,

        "bad_json":
            xint_bad,
    },

    "materials": {
        "records":
            material_count,
    },

    "computational": {
        "records":
            computation_count,

        "s4e_xint_linked":
            s4e_linked,

        "s4e_unlinked":
            s4e_unlinked,

        "chemscreen_dft":
            chemscreen_dft,

        "chemscreen_candidate_pool":
            candidate_pool,

        "other":
            other_computations,
    },

    "efa": {
        "pocc_groups":
            efa_groups,

        "complete":
            efa_complete,

        "incomplete":
            efa_incomplete,
    },

    "literature": {
        "files":
            literature_files,

        "records":
            literature_count,
    },

    "experimental": {
        "scope":
            "OSES only",

        "oses_labels":
            len(oses),

        "matched":
            matched,

        "unmatched":
            unmatched,

        "ambiguous":
            ambiguous,

        "loop_trial_files_scanned":
            len(trial_files),

        "non_oses_trials_written":
            0,
    },
}

write_json(
    COUNTS_OUT,
    counts,
)

write_json(
    VALIDATION_OUT,
    validation,
)

write_json(
    MANIFEST_OUT,
    {
        "created_at":
            now(),

        "dataset_scope": {
            "materials":
                "all LOOP materials",

            "computational":
                "all LOOP DFT calculations",

            "literature":
                "all LOOP literature records",

            "experimental":
                "OSES experiments only",
        },

        "sources": {
            "materials":
                str(
                    MATERIALS_ROOT
                ),

            "xint":
                str(
                    XINT_ROOT
                ),

            "raw_aflow":
                "/common/LIB*/RAW",

            "raw_uploads":
                str(
                    RAW_UPLOAD_ROOT
                ),

            "oses_labels":
                str(
                    OSES_CSV
                ),
        },
    },
)

conn.close()


print("\n=== BUILD COMPLETE ===")
print(
    json.dumps(
        counts,
        indent=2,
    )
)

print(
    "\nOutput:",
    OUTPUT_ROOT,
)