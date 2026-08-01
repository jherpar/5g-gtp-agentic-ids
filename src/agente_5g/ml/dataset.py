"""Dataset construction for all four Phase 6 arms.

- **Arm A1 (official baseline, primary)**: `data/processed/Combined/Combined.csv`
  (the dataset authors' Argus-derived flow export, GTP-stripped), with our
  own documented preprocessing -- categorical columns are readable
  (`Proto`, `sDSb`, `dDSb`, `Cause`, `State`), so every transform is
  interpretable and citable.
- **Arm A2 (official baseline, secondary/reproducibility check)**:
  `data/processed/Encoded/Encoded.csv`, the authors' own pre-encoded
  columns used near-verbatim (only NaN imputation). NOT the primary
  official-baseline claim -- many of its one-hot column names are
  uninterpretable artifacts of the authors' own encoding step, so results
  from this arm are reported only as a secondary check against the
  published representation.
- **Arm B/C (GTP-U informed)**: built from real labeled `PDUSessionRecord`
  data (`preprocessing/labeling.py`'s output) -- arm B trains classical ML
  on these features, arm C is the existing agentic pipeline
  (`agents/teid_agent.py` etc.) evaluated on the same split, not trained.

Split strategy (risk #6 in the original plan, `experiment_plan.md`):
Combined.csv/Encoded.csv have no session/TEID identifiers, so arm A1/A2 use
a chronological split by `Seq` (row order == capture order). Arm B/C use a
TEID-safe split -- a (ue_ip, teid) group is never split across train/test,
computed per attack-type file (so no single file's traffic type is
entirely held out) then concatenated. These are different units of
analysis and are reported/compared as such, never conflated as if measured
on identical units.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from agente_5g.models.session import PDUSessionRecord

SEED = 42
DEFAULT_TEST_FRACTION = 0.3

COMBINED_CATEGORICAL_COLUMNS = ["Proto", "sDSb", "dDSb", "Cause", "State"]
COMBINED_TARGET_COLUMN = "Label"
COMBINED_SPLIT_COLUMN = "Seq"
_NON_FEATURE_COLUMNS = {"_row_index", "Seq", "Attack Type", "Attack Tool", "Label", "is_attack"}

GTP_SESSION_FEATURE_COLUMNS = [
    "window_size_s",
    "duration_s",
    "traffic_volume_bytes",
    "flow_diversity",
    "port_diversity",
    "destination_diversity",
    "state_transition_rate",
    "temporal_entropy",
]


def load_combined_csv(path: Path) -> pd.DataFrame:
    """Load Combined.csv, add a boolean `is_attack` target, and one-hot
    encode the categorical columns over the WHOLE file before any split --
    this fixes a consistent column set for train/test (a fixed protocol
    vocabulary, not something that depends on the target), it does not
    leak label information into the encoding itself."""
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={df.columns[0]: "_row_index"})
    df["is_attack"] = df[COMBINED_TARGET_COLUMN] == "Malicious"
    for col in COMBINED_CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("__MISSING__").astype(str)
    return pd.get_dummies(
        df, columns=COMBINED_CATEGORICAL_COLUMNS, prefix=COMBINED_CATEGORICAL_COLUMNS
    )


def load_encoded_csv(path: Path) -> pd.DataFrame:
    """Load Encoded.csv near-verbatim -- the authors' own columns (many
    with uninterpretable one-hot names) are kept as-is; only the row-index
    column is renamed and the boolean target added."""
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={df.columns[0]: "_row_index"})
    df["is_attack"] = df[COMBINED_TARGET_COLUMN] == "Malicious"
    return df


def to_arm_a_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a loaded arm-A1/A2 DataFrame into (X, y). NaN in numeric
    columns is imputed with -1 -- a sentinel distinguishing "structurally
    not applicable" (e.g. TCP-only fields on a UDP flow) from an observed
    value; tree ensembles handle a fixed sentinel natively, unlike linear
    models."""
    y = df["is_attack"]
    x = df.drop(columns=[c for c in _NON_FEATURE_COLUMNS if c in df.columns])
    x = x.select_dtypes(include=["number", "bool"]).fillna(-1)
    return x, y


def chronological_split(
    df: pd.DataFrame, order_column: str, test_fraction: float = DEFAULT_TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by `order_column` (ascending) and split so the test set is the
    chronologically LATER fraction -- never a random shuffle, since that
    would leak future traffic patterns into training."""
    ordered = df.sort_values(order_column)
    split_idx = int(len(ordered) * (1 - test_fraction))
    return ordered.iloc[:split_idx], ordered.iloc[split_idx:]


def build_gtp_session_dataset(
    sessions_by_attack_type: dict[str, list[PDUSessionRecord]],
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TEID-safe chronological split, computed independently per attack-type
    file then concatenated, so every attack type contributes to both train
    and test. A (ue_ip, teid) group's sessions always stay entirely on one
    side of the split -- ordered by the group's earliest `start_time`.

    Returns (train_df, test_df), each with `GTP_SESSION_FEATURE_COLUMNS`
    plus `is_attack`, `label_confidence`, `attack_type`, `session_id`
    (metadata columns, not model inputs -- see `to_gtp_matrix`).
    """
    train_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []

    for attack_type, sessions in sessions_by_attack_type.items():
        labeled = [s for s in sessions if s.is_attack is not None]
        groups: dict[tuple[str, int], list[PDUSessionRecord]] = defaultdict(list)
        for s in labeled:
            groups[(s.ue_ip, s.teid)].append(s)

        group_keys = sorted(groups, key=lambda k: min(s.start_time for s in groups[k]))
        split_idx = int(len(group_keys) * (1 - test_fraction))
        train_keys = set(group_keys[:split_idx])

        for key, group_sessions in groups.items():
            target = train_rows if key in train_keys else test_rows
            for s in group_sessions:
                row: dict[str, object] = {
                    col: getattr(s, col) for col in GTP_SESSION_FEATURE_COLUMNS
                }
                row["is_attack"] = bool(s.is_attack)
                row["label_confidence"] = s.label_confidence.value if s.label_confidence else None
                row["attack_type"] = attack_type
                row["session_id"] = s.session_id
                target.append(row)

    columns = [
        *GTP_SESSION_FEATURE_COLUMNS,
        "is_attack",
        "label_confidence",
        "attack_type",
        "session_id",
    ]
    return pd.DataFrame(train_rows, columns=columns), pd.DataFrame(test_rows, columns=columns)


def to_gtp_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a `build_gtp_session_dataset` frame into (X, y) -- numeric
    feature matrix and boolean target. Metadata columns
    (label_confidence/attack_type/session_id) are dropped from X but not
    lost from the source frame, so callers can still filter/group rows by
    confidence tier for the view A/B/C evaluation split."""
    y = df["is_attack"]
    x = df[GTP_SESSION_FEATURE_COLUMNS].copy()
    return x, y
