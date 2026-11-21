"""Data model for drift-batch-blame.

A *stream* is a reference sample plus a sequence of timestamped
ingestion batches, each from a named source.  A *scenario* plants
0-3 culprit batches carrying a known shift; the planted truth is the
ground truth every blame method is scored against.

All data is synthetic and generated from the declared process in
``generate.py``.  There is no real patient or production data
anywhere in this project.
"""

from dataclasses import dataclass, field

NUMERIC_FEATURES = ["age", "systolic_bp", "glucose_mg_dl", "creatinine", "bmi"]
CATEGORICAL_FEATURE = "device_model"
CATEGORIES = ["A", "B", "C"]
LABEL = "readmitted"
ALL_FEATURES = NUMERIC_FEATURES + [CATEGORICAL_FEATURE, LABEL]

SOURCES = ["north-feed", "south-feed", "partner-api", "device-hub"]

SHIFT_MEAN = "mean_shift"
SHIFT_LABEL = "label_rate_shift"
SHIFT_SCHEMA = "schema_value_shift"
SHIFT_TYPES = [SHIFT_MEAN, SHIFT_LABEL, SHIFT_SCHEMA]

# Unseen category value planted by a schema-value shift.
SCHEMA_NEW_VALUE = "D"

# Base distribution of the synthetic data-generating process.
BASE_MEANS = {
    "age": 55.0,
    "systolic_bp": 122.0,
    "glucose_mg_dl": 104.0,
    "creatinine": 1.0,
    "bmi": 27.0,
}
BASE_SDS = {
    "age": 14.0,
    "systolic_bp": 16.0,
    "glucose_mg_dl": 26.0,
    "creatinine": 0.32,
    "bmi": 4.6,
}
CATEGORY_PROBS = {"A": 0.45, "B": 0.35, "C": 0.20}


@dataclass
class Batch:
    """One ingestion batch: rows delivered by one source on one day."""

    batch_id: str
    source: str
    day: int
    rows: list

    @property
    def size(self):
        return len(self.rows)


@dataclass
class Culprit:
    """A planted shift: which batch, which mechanism, how strong."""

    batch_index: int
    shift_type: str
    feature: str = ""        # for mean shifts: the shifted feature
    magnitude: float = 0.0   # mean: sigmas; label: target rate; schema: fraction


@dataclass
class Scenario:
    """A full seeded evaluation scenario."""

    seed: int
    n_batches: int = 10
    batch_size: int = 300
    ref_size: int = 3000
    culprits: list = field(default_factory=list)


@dataclass
class Stream:
    """A generated stream plus its planted ground truth."""

    reference_rows: list
    batches: list
    truth: dict  # batch_id -> shift_type, culprit batches only

    @property
    def culprit_ids(self):
        return sorted(self.truth.keys())
