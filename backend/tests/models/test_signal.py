from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import inspect

from app.core.database import create_schema, get_engine, get_session_factory
from app.datasets.mafaulda_parser import KNOWN_STATES
from app.models.signal import Dataset, Signal, SignalLabel, SignalRecord, SignalWindow


def _base_kwargs(**overrides):
    kwargs = {
        "dataset_id": 1,
        "sampling_rate": 50000.0,
        "channel": 0,
        "file_path": "normal/12.288.csv",
        "label": SignalLabel.NORMAL,
    }
    kwargs.update(overrides)
    return kwargs


# --- sampling_rate validation (AC1) ---


def test_valid_sampling_rate_is_accepted() -> None:
    record = SignalRecord(**_base_kwargs(sampling_rate=50000.0))
    assert record.sampling_rate == 50000.0


def test_zero_sampling_rate_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SignalRecord(**_base_kwargs(sampling_rate=0))


def test_negative_sampling_rate_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SignalRecord(**_base_kwargs(sampling_rate=-1))


# --- label validation (AC2) ---


def test_valid_real_label_is_accepted() -> None:
    record = SignalRecord(**_base_kwargs(label="imbalance"))
    assert record.label == SignalLabel.IMBALANCE


def test_unknown_label_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SignalRecord(**_base_kwargs(label="this_is_not_a_real_class"))


def test_signal_label_enum_matches_audit_confirmed_states() -> None:
    """SignalLabel must mirror mafaulda_parser.KNOWN_STATES exactly -- the module-level
    assert in signal.py already enforces this at import time; this test makes the
    invariant explicit and independently checkable."""
    assert {member.value for member in SignalLabel} == KNOWN_STATES


def test_signal_label_excludes_unknown_sentinel() -> None:
    assert "UNKNOWN" not in {member.value for member in SignalLabel}
    assert not any(member.name == "UNKNOWN" for member in SignalLabel)


# --- channel bounds (grounded in the audit-confirmed 8-channel structure) ---


def test_channel_within_confirmed_bounds_is_accepted() -> None:
    for channel in range(8):
        record = SignalRecord(**_base_kwargs(channel=channel))
        assert record.channel == channel


def test_channel_outside_confirmed_bounds_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SignalRecord(**_base_kwargs(channel=8))
    with pytest.raises(ValidationError):
        SignalRecord(**_base_kwargs(channel=-1))


# --- optional fields never invented ---


def test_machine_id_and_operating_condition_default_to_none() -> None:
    record = SignalRecord(**_base_kwargs())
    assert record.machine_id is None
    assert record.operating_condition is None


def test_operating_condition_can_be_set_for_non_normal_states() -> None:
    record = SignalRecord(**_base_kwargs(label="imbalance", operating_condition="10g"))
    assert record.operating_condition == "10g"


# --- database schema (test DB only, never production) ---


def test_database_schema_creates_expected_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "test_signal_models.db"
    engine = get_engine(f"sqlite:///{db_path}")

    create_schema(engine)

    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    assert {"datasets", "signals", "signal_windows"}.issubset(table_names)


def test_signal_sampling_rate_check_constraint_is_enforced_at_db_level(tmp_path: Path) -> None:
    db_path = tmp_path / "test_signal_constraint.db"
    engine = get_engine(f"sqlite:///{db_path}")
    create_schema(engine)
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        dataset = Dataset(name="mafaulda")
        session.add(dataset)
        session.flush()

        bad_signal = Signal(
            dataset_id=dataset.id,
            sampling_rate=-5.0,
            channel=0,
            file_path="normal/12.288.csv",
            label=SignalLabel.NORMAL,
        )
        session.add(bad_signal)

        with pytest.raises(Exception):  # sqlite3.IntegrityError wrapped by SQLAlchemy
            session.commit()


def test_signal_and_signal_window_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "test_signal_roundtrip.db"
    engine = get_engine(f"sqlite:///{db_path}")
    create_schema(engine)
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        dataset = Dataset(name="mafaulda")
        session.add(dataset)
        session.flush()

        signal = Signal(
            dataset_id=dataset.id,
            sampling_rate=50000.0,
            channel=0,
            file_path="normal/12.288.csv",
            label=SignalLabel.NORMAL,
        )
        session.add(signal)
        session.flush()

        window = SignalWindow(signal_id=signal.id, start_sample=0, end_sample=1000)
        session.add(window)
        session.commit()

        assert signal.id is not None
        assert window.signal_id == signal.id
