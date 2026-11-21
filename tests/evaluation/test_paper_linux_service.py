"""Linux operating platform for attempt 3: layout, systemd user units and clock report.

No network, no systemd, no host state: units are rendered text, the clock commands are
fakes, and the Windows recipe in the specification is compared with the Linux units so
the two platforms cannot drift apart in cadence or restart budget.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

import pytest

from trade_rl.evaluation.paper import attempt3, cli, linux_service
from trade_rl.evaluation.paper.linux_service import (
    CHECK_TIMER,
    CHECK_UNIT,
    SUPERVISOR_UNIT,
    LinuxServiceRefusal,
    build_layout,
    check_command,
    clock_report,
    render_check_service,
    render_check_timer,
    render_supervisor_unit,
    render_units,
    run_read_only,
    supervisor_command,
    write_units,
)
from trade_rl.evaluation.paper.watchdog import DEFAULT_STALE_SECONDS, popen_launcher

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "docs" / "specs" / "funding-carry-paper.md"
HOME = "/home/carry"
COMMIT = "d0a76538f77a5a99825fc0c698a27153d8181933"
DATE = "20261201"
DIGEST = "ab" * 32
START = datetime(2026, 12, 1, 0, 0, tzinfo=UTC)
AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-4242424242"


def layout():
    return build_layout(HOME, DATE, COMMIT)


def units() -> dict[str, str]:
    return render_units(
        layout(), start_at=START, authorization=AUTH, protocol_sha256=DIGEST
    )


def words(unit: str, key: str) -> list[str]:
    (line,) = [item for item in unit.splitlines() if item.startswith(f"{key}=")]
    return shlex.split(line.removeprefix(f"{key}="))


# ------------------------------------------------------------------------- layout


def test_layout_uses_linux_run_and_frozen_roots_under_the_home():
    value = layout()
    assert str(value.root) == "/home/carry/trade_rl_runs/attempt3/root-20261201"
    assert str(value.probe_root) == (
        "/home/carry/trade_rl_runs/attempt3/storage-probe-20261201"
    )
    assert str(value.probe_report) == (
        "/home/carry/trade_rl_runs/attempt3/storage-probe-20261201.report.json"
    )
    assert str(value.supervisor_log) == (
        "/home/carry/trade_rl_runs/attempt3/logs/supervisor-20261201.log"
    )
    assert str(value.checkout) == "/home/carry/trade_rl_frozen/attempt3-d0a76538"
    assert (
        str(value.python)
        == "/home/carry/trade_rl_frozen/attempt3-d0a76538/.venv/bin/python"
    )
    assert str(value.predecessor_root) == (
        "/home/carry/trade_rl_runs/attempt2/root-20261107"
    )
    assert value.to_dict()["schema"] == "paper_linux_layout_v1"
    assert all(
        isinstance(item, str) and "\\" not in item for item in value.to_dict().values()
    )


def test_layout_keeps_roots_apart_from_the_checkout_and_the_attempt_two_root():
    value = layout()
    predecessor = value.predecessor_root
    for path in (value.root, value.probe_root, value.supervisor_log):
        assert predecessor not in path.parents and path not in predecessor.parents
        assert value.checkout not in path.parents
    assert value.root not in value.supervisor_log.parents
    assert value.root != value.probe_root
    assert "attempt2" not in str(value.root)
    # The real launcher refuses a log inside the sealed root; the layout never does.
    popen_launcher(str(value.root), DIGEST, str(value.supervisor_log))


@pytest.mark.parametrize(
    "home",
    [
        "carry",
        "C:/Users/carry",
        "/",
        "/home/carry/",
        "/home/car ry",
        "/home/../etc",
        "/home/./carry",
        "//home/carry",
        "/home/carry%h",
        "/home/$USER",
        "/home/car'ry",
        "",
    ],
)
def test_layout_refuses_an_unsafe_home(home):
    with pytest.raises(LinuxServiceRefusal):
        build_layout(home, DATE, COMMIT)


@pytest.mark.parametrize("date", ["2026-12-01", "20261301", "2026120", "abcdefgh", ""])
def test_layout_refuses_a_bad_seal_date(date):
    with pytest.raises(LinuxServiceRefusal):
        build_layout(HOME, date, COMMIT)


@pytest.mark.parametrize("commit", ["d0a76538", COMMIT.upper(), COMMIT + "0", "g" * 40])
def test_layout_refuses_a_bad_launch_commit(commit):
    with pytest.raises(LinuxServiceRefusal):
        build_layout(HOME, DATE, commit)


# -------------------------------------------------------------------------- units


def test_supervisor_unit_text_is_exact():
    unit = units()[SUPERVISOR_UNIT]
    assert unit == (
        "# Rendered by trade_rl.evaluation.paper.linux_service for the attempt-3 carry\n"
        "# paper screen (public data only, no orders). Do not edit after the seal.\n"
        "[Unit]\n"
        "Description=Attempt 3 carry paper screen supervisor "
        "(public data only, no orders)\n"
        "StartLimitIntervalSec=600\n"
        "StartLimitBurst=4\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        "WorkingDirectory=/home/carry/trade_rl_frozen/attempt3-d0a76538\n"
        "ExecStart=/home/carry/trade_rl_frozen/attempt3-d0a76538/.venv/bin/python "
        "-m trade_rl.evaluation.paper.attempt3 launch "
        "--root /home/carry/trade_rl_runs/attempt3/root-20261201 "
        "--start-at 2026-12-01T00:00:00+00:00 "
        '--authorization "https://github.com/shuntatsu/trade_rl/issues/810'
        '#issuecomment-4242424242" '
        "--allow-network --protocol-sha256 " + DIGEST + " "
        "--log-file /home/carry/trade_rl_runs/attempt3/logs/supervisor-20261201.log\n"
        "Restart=on-failure\n"
        "RestartSec=60\n"
        "RestartPreventExitStatus=1 3 4\n"
        "KillMode=control-group\n"
        "SendSIGHUP=no\n"
        "TimeoutStopSec=30\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def test_check_service_and_timer_text_are_exact():
    rendered = units()
    assert rendered[CHECK_UNIT] == (
        "# Rendered by trade_rl.evaluation.paper.linux_service for the attempt-3 carry\n"
        "# paper screen (public data only, no orders). Do not edit after the seal.\n"
        "[Unit]\n"
        "Description=Attempt 3 carry paper screen liveness check "
        "(stateless, one run per timer tick)\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        "WorkingDirectory=/home/carry/trade_rl_frozen/attempt3-d0a76538\n"
        "ExecStart=/home/carry/trade_rl_frozen/attempt3-d0a76538/.venv/bin/python "
        "-m trade_rl.evaluation.paper.cli check "
        "--root /home/carry/trade_rl_runs/attempt3/root-20261201 "
        "--protocol-sha256 " + DIGEST + "\n"
    )
    assert rendered[CHECK_TIMER] == render_check_timer()
    assert "OnCalendar=minutely" in rendered[CHECK_TIMER]
    assert f"Unit={CHECK_UNIT}" in rendered[CHECK_TIMER]
    assert "WantedBy=timers.target" in rendered[CHECK_TIMER]
    assert set(rendered) == {SUPERVISOR_UNIT, CHECK_UNIT, CHECK_TIMER}
    assert (SUPERVISOR_UNIT, CHECK_UNIT, CHECK_TIMER) == (
        "carry-a3-supervisor.service",
        "carry-a3-check.service",
        "carry-a3-check.timer",
    )


def test_rendered_commands_are_accepted_by_the_real_attempt3_and_cli_parsers(
    monkeypatch,
):
    launch = words(units()[SUPERVISOR_UNIT], "ExecStart")
    assert launch == supervisor_command(
        layout(), start_at=START, authorization=AUTH, protocol_sha256=DIGEST
    )
    parsed = attempt3.build_parser().parse_args(launch[3:])
    assert parsed.command == "launch"
    assert parsed.root == Path("/home/carry/trade_rl_runs/attempt3/root-20261201")
    assert parsed.start_at == START
    assert parsed.authorization == AUTH
    assert parsed.allow_network is True
    assert parsed.protocol_sha256 == DIGEST
    assert str(parsed.log_file).endswith("supervisor-20261201.log")
    assert launch[:4] == [
        str(layout().python),
        "-m",
        "trade_rl.evaluation.paper.attempt3",
        "launch",
    ]

    check = words(units()[CHECK_UNIT], "ExecStart")
    assert check == check_command(layout(), protocol_sha256=DIGEST)
    seen = {}

    def fake_check(root, *, protocol_sha256, stale_seconds):
        seen.update(root=root, digest=protocol_sha256, stale=stale_seconds)
        return dict(severity="healthy")

    monkeypatch.setattr(cli, "check_collection", fake_check)
    assert cli.main(check[3:]) == 0
    assert seen["digest"] == DIGEST and seen["stale"] == DEFAULT_STALE_SECONDS
    assert seen["root"] == Path("/home/carry/trade_rl_runs/attempt3/root-20261201")


def test_units_mirror_the_windows_task_budget_and_cadence():
    spec = " ".join(SPEC.read_text(encoding="utf-8").split())
    assert "-RestartCount 3" in spec
    assert "-RestartInterval (New-TimeSpan -Minutes 1)" in spec
    assert "-RepetitionInterval (New-TimeSpan -Minutes 1)" in spec
    assert linux_service.RESTART_SECONDS == 60
    # systemd counts the first start: one start plus the Windows RestartCount of three.
    assert linux_service.START_LIMIT_BURST == 3 + 1
    assert linux_service.START_LIMIT_INTERVAL_SECONDS > (
        linux_service.START_LIMIT_BURST * linux_service.RESTART_SECONDS
    )
    # The watchdog cadence is code, not a unit option: the units pass no policy flag.
    assert "--stale-seconds" not in units()[CHECK_UNIT]
    assert "--stale-seconds" not in units()[SUPERVISOR_UNIT]
    assert DEFAULT_STALE_SECONDS == 150.0


def test_the_supervisor_never_restarts_into_a_terminal_or_owned_root():
    unit = units()[SUPERVISOR_UNIT]
    assert "Restart=on-failure" in unit
    assert "RestartPreventExitStatus=1 3 4" in unit
    # SIGTERM reaches the collector on stop, logout without lingering and shutdown.
    assert "KillMode=control-group" in unit
    assert "SendSIGHUP=no" in unit
    assert "Restart=always" not in unit
    assert linux_service.NO_RESTART_EXIT_CODES == (1, 3, 4)
    assert attempt3.EXIT_LOCK_HELD == 4


@pytest.mark.parametrize(
    "authorization",
    [
        "",
        "https://example.com/issues/810#issuecomment-1",
        AUTH + "; rm -rf /",
        AUTH + " x",
        AUTH + "%h",
        AUTH + "$HOME",
    ],
)
def test_units_refuse_an_unsafe_or_malformed_authorization(authorization):
    with pytest.raises(LinuxServiceRefusal):
        render_supervisor_unit(
            layout(),
            start_at=START,
            authorization=authorization,
            protocol_sha256=DIGEST,
        )


@pytest.mark.parametrize("digest", ["", "AB" * 32, "ab" * 31, "ab" * 33, "xy" * 32])
def test_units_refuse_a_bad_protocol_digest(digest):
    with pytest.raises(LinuxServiceRefusal):
        render_check_service(layout(), protocol_sha256=digest)
    with pytest.raises(LinuxServiceRefusal):
        render_supervisor_unit(
            layout(), start_at=START, authorization=AUTH, protocol_sha256=digest
        )


def test_units_refuse_a_start_without_a_utc_offset_and_normalise_to_utc():
    with pytest.raises(LinuxServiceRefusal, match="start_not_timezone_aware"):
        render_supervisor_unit(
            layout(),
            start_at=datetime(2026, 12, 1),
            authorization=AUTH,
            protocol_sha256=DIGEST,
        )
    tokyo = timezone(timedelta(hours=9))
    unit = render_supervisor_unit(
        layout(),
        start_at=datetime(2026, 12, 1, 9, 0, tzinfo=tokyo),
        authorization=AUTH,
        protocol_sha256=DIGEST,
    )
    assert "--start-at 2026-12-01T00:00:00+00:00" in unit


def test_write_units_creates_files_exclusively_and_never_overwrites(tmp_path):
    rendered = units()
    written = write_units(tmp_path, rendered)
    assert written == sorted(rendered)
    for name, text in rendered.items():
        data = (tmp_path / name).read_bytes()
        assert data == text.encode("utf-8") and b"\r" not in data
    before = {name: (tmp_path / name).read_bytes() for name in rendered}
    with pytest.raises(LinuxServiceRefusal, match="unit_file_exists"):
        write_units(tmp_path, {**rendered, SUPERVISOR_UNIT: "changed\n"})
    assert before == {name: (tmp_path / name).read_bytes() for name in rendered}
    with pytest.raises(LinuxServiceRefusal, match="existing_directory"):
        write_units(tmp_path / "missing", rendered)


def test_write_units_is_all_or_nothing_when_one_name_exists(tmp_path):
    (tmp_path / CHECK_TIMER).write_text("mine\n", encoding="utf-8")
    with pytest.raises(LinuxServiceRefusal, match=CHECK_TIMER):
        write_units(tmp_path, units())
    assert sorted(path.name for path in tmp_path.iterdir()) == [CHECK_TIMER]


# --------------------------------------------------------------------- command line


def test_cli_paths_render_and_refusal(tmp_path, capsys):
    base = ["--home", HOME, "--seal-date", DATE, "--launch-commit", COMMIT]
    assert linux_service.main(["paths", *base]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["root"] == "/home/carry/trade_rl_runs/attempt3/root-20261201"
    render = [
        "render",
        *base,
        "--start-at",
        "2026-12-01T00:00:00Z",
        "--authorization",
        AUTH,
        "--protocol-sha256",
        DIGEST,
    ]
    assert linux_service.main(render) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["written"] == [] and set(shown["units"]) == set(units())
    assert not list(tmp_path.iterdir())
    assert linux_service.main([*render, "--output-dir", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["written"] == sorted(units())
    assert linux_service.main([*render, "--output-dir", str(tmp_path)]) == 2
    assert "unit_file_exists" in capsys.readouterr().err
    bad = [
        "paths",
        "--home",
        "relative",
        "--seal-date",
        DATE,
        "--launch-commit",
        COMMIT,
    ]
    assert linux_service.main(bad) == 2
    assert "home_must_be" in capsys.readouterr().err


# ------------------------------------------------------------------------- clock

CHRONY_OK = """\
Reference ID    : C0A80001 (ntp.example)
Stratum         : 3
Ref time (UTC)  : Thu Nov 19 03:00:00 2026
System time     : 0.000012345 seconds fast of NTP time
Last offset     : +0.000001234 seconds
RMS offset      : 0.000020000 seconds
Frequency       : 1.234 ppm slow
Leap status     : Normal
"""
SYNCED = "NTPSynchronized=yes\n"
TIMESYNC = (
    "       Server: 10.0.0.1 (ntp.example)\n      Stratum: 2\n       Offset: -1.058ms\n"
)


def runner(answers):
    seen = []

    def run(arguments):
        seen.append(tuple(arguments))
        return answers.get(tuple(arguments), (127, ""))

    run.seen = seen  # type: ignore[attr-defined]
    return run


COMMANDS = linux_service.READ_ONLY_CLOCK_COMMANDS
KERNEL, CHRONY, TIMESYNCD = COMMANDS


def test_clock_is_ok_from_chrony_and_reports_a_signed_offset():
    run = runner({KERNEL: (0, SYNCED), CHRONY: (0, CHRONY_OK)})
    report = clock_report(run)
    assert report["ok"] is True and report["source"] == "chrony"
    assert report["offset_seconds"] == pytest.approx(1.2345e-5)
    assert report["reasons"] == [] and report["kernel_synchronized"] is True
    assert report["production_eligible"] is False
    assert TIMESYNCD not in run.seen
    slow = CHRONY_OK.replace("fast", "slow")
    assert clock_report(runner({KERNEL: (0, SYNCED), CHRONY: (0, slow)}))[
        "offset_seconds"
    ] == pytest.approx(-1.2345e-5)


@pytest.mark.parametrize(
    ("chrony", "reason"),
    [
        (CHRONY_OK.replace("0.000012345", "0.300000000"), "offset_exceeds_limit"),
        (
            CHRONY_OK.replace("0.000012345", "0.300000000").replace("fast", "slow"),
            "offset_exceeds_limit",
        ),
        (
            CHRONY_OK.replace("Normal", "Not synchronised"),
            "chrony_leap_or_stratum_not_normal",
        ),
        (
            CHRONY_OK.replace("Stratum         : 3", "Stratum         : 0"),
            "chrony_leap_or_stratum_not_normal",
        ),
        (
            CHRONY_OK.replace("System time     :", "Sys time        :"),
            "offset_unavailable",
        ),
    ],
)
def test_clock_fails_closed_on_a_bad_chrony_state(chrony, reason):
    report = clock_report(runner({KERNEL: (0, SYNCED), CHRONY: (0, chrony)}))
    assert report["ok"] is False
    assert reason in report["reasons"]


def test_clock_falls_back_to_systemd_timesyncd_with_unit_conversion():
    run = runner({KERNEL: (0, SYNCED), TIMESYNCD: (0, TIMESYNC)})
    report = clock_report(run)
    assert report["ok"] is True and report["source"] == "systemd-timesyncd"
    assert report["offset_seconds"] == pytest.approx(-1.058e-3)
    assert run.seen == [KERNEL, CHRONY, TIMESYNCD]
    for text, seconds in (("12us", 12e-6), ("3.5ms", 3.5e-3), ("0.1s", 0.1)):
        answers = {KERNEL: (0, SYNCED), TIMESYNCD: (0, f"  Offset: {text}\n")}
        assert clock_report(runner(answers))["offset_seconds"] == pytest.approx(seconds)
    slow = {KERNEL: (0, SYNCED), TIMESYNCD: (0, "  Offset: +900ms\n")}
    assert clock_report(runner(slow))["reasons"] == ["offset_exceeds_limit"]


def test_clock_fails_closed_without_an_answer_or_a_synchronised_kernel():
    nothing = clock_report(runner({}))
    assert nothing["ok"] is False
    assert nothing["source"] is None and nothing["offset_seconds"] is None
    assert nothing["reasons"] == [
        "kernel_clock_state_unavailable",
        "no_ntp_client_answered",
    ]
    unsynchronised = clock_report(
        runner({KERNEL: (0, "NTPSynchronized=no\n"), CHRONY: (0, CHRONY_OK)})
    )
    assert unsynchronised["ok"] is False
    assert unsynchronised["reasons"] == ["kernel_clock_not_synchronized"]
    garbled = clock_report(runner({KERNEL: (0, "garbage\n"), CHRONY: (1, CHRONY_OK)}))
    assert garbled["ok"] is False


def test_clock_only_ever_runs_read_only_commands():
    verbs = {command[1] for command in COMMANDS}
    assert verbs == {"show", "tracking", "timesync-status"}
    assert not [
        c for c in COMMANDS if any("set" in word or "step" in word for word in c)
    ]
    clock_report(runner({}))
    with pytest.raises(LinuxServiceRefusal, match="read_only"):
        run_read_only(("timedatectl", "set-ntp", "true"))
    with pytest.raises(LinuxServiceRefusal, match="read_only"):
        run_read_only(("chronyc", "makestep"))


def test_a_missing_or_hanging_clock_tool_is_a_failed_read_not_an_error(monkeypatch):
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("timedatectl")

    monkeypatch.setattr(linux_service.subprocess, "run", missing)
    assert run_read_only(KERNEL) == (127, "")

    def hanging(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("timedatectl", 10)

    monkeypatch.setattr(linux_service.subprocess, "run", hanging)
    assert run_read_only(CHRONY) == (127, "")


def test_clock_command_prints_a_report_and_maps_failure_to_exit_one(
    monkeypatch, capsys
):
    monkeypatch.setattr(linux_service, "run_read_only", runner({}))
    assert linux_service.main(["clock"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    good = runner({KERNEL: (0, SYNCED), CHRONY: (0, CHRONY_OK)})
    monkeypatch.setattr(linux_service, "run_read_only", good)
    assert linux_service.main(["clock"]) == 0


def test_layout_type_is_posix_even_on_a_windows_host():
    assert isinstance(layout().home, PurePosixPath)
    assert isinstance(layout().root, PurePosixPath)
