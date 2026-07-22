"""Training telemetry contracts and append-only indexed storage."""

from trade_rl.telemetry import training as _training
from trade_rl.telemetry.indexed_training import (
    IndexedTrainingTelemetryWriter,
    StrictTrainingTelemetryRecord,
    indexed_training_telemetry_status,
    read_indexed_training_telemetry,
)

setattr(_training, "TrainingTelemetryRecord", StrictTrainingTelemetryRecord)
setattr(_training, "TrainingTelemetryWriter", IndexedTrainingTelemetryWriter)
setattr(_training, "read_training_telemetry", read_indexed_training_telemetry)
setattr(_training, "training_telemetry_status", indexed_training_telemetry_status)

TELEMETRY_SCHEMA_VERSION = _training.TELEMETRY_SCHEMA_VERSION
TrainingTelemetryPage = _training.TrainingTelemetryPage
TrainingTelemetryRecord = StrictTrainingTelemetryRecord
TrainingTelemetryStatus = _training.TrainingTelemetryStatus
TrainingTelemetryWriter = IndexedTrainingTelemetryWriter
read_training_telemetry = read_indexed_training_telemetry
training_telemetry_status = indexed_training_telemetry_status

__all__ = [
    "TELEMETRY_SCHEMA_VERSION",
    "TrainingTelemetryPage",
    "TrainingTelemetryRecord",
    "TrainingTelemetryStatus",
    "TrainingTelemetryWriter",
    "read_training_telemetry",
    "training_telemetry_status",
]
