"""Compatibility imports for canonical indexed training telemetry."""

from trade_rl.telemetry.training import (
    TrainingTelemetryRecord,
    TrainingTelemetryWriter,
    read_training_telemetry,
    training_telemetry_status,
)

StrictTrainingTelemetryRecord = TrainingTelemetryRecord
IndexedTrainingTelemetryWriter = TrainingTelemetryWriter
read_indexed_training_telemetry = read_training_telemetry
indexed_training_telemetry_status = training_telemetry_status

__all__ = [
    "IndexedTrainingTelemetryWriter",
    "StrictTrainingTelemetryRecord",
    "indexed_training_telemetry_status",
    "read_indexed_training_telemetry",
]
