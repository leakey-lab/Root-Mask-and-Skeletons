"""A Measure CSV's provenance names the generation run behind its predictions.

``run_record.generation_summary`` reads ``_sprouts_run.json`` and ``_sprouts_runs.jsonl``
of the folder the predictions and the CSV share; ``MeasureRun`` writes it as
``inputs.generation`` of the CSV's sidecar. The reader never raises, whatever the two
files hold.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading

import cv2
import numpy as np
import pytest

from app import provenance
from app.inference import measure_run, run_record
from app.inference.batch_engine import BatchReport
from app.inference.measure_run import AREA, LENGTH, MeasureRun
from app.inference.run_record import generation_summary

_DEPTH = 100_000
NESTED = {
    "array": "[" * _DEPTH + "]" * _DEPTH,
    "object": '{"a":' * _DEPTH + "1" + "}" * _DEPTH,
}
#: Stack of the thread the JSON is parsed on: 100,000 levels overflow it on every OS
#: (as ``tests/test_json_nesting_robustness.py``: since Python 3.14 the limit is the stack).
_PARSE_STACK = 2 << 20

_NAMES = [
    "24EFCAM1_T001_L001_2024.08.06_102012_001_SMM",
    "24EFCAM2_T005_L056_2024.08.26_085019_003_SMM",
]


@pytest.fixture
def small_stack_json(monkeypatch):
    """``json.loads`` (and ``json.load``) parse on a thread with a 2 MB stack."""
    real_loads = json.loads

    def loads(*args, **kwargs):
        outcome = {}

        def parse():
            try:
                outcome["value"] = real_loads(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                outcome["error"] = exc

        previous = threading.stack_size(_PARSE_STACK)
        try:
            worker = threading.Thread(target=parse, name="json-small-stack")
            worker.start()
        finally:
            threading.stack_size(previous)
        worker.join()
        if "error" in outcome:
            raise outcome["error"]
        return outcome["value"]

    monkeypatch.setattr(json, "loads", loads)


def _checkpoint(tmp_path, name: str, data: bytes) -> str:
    path = tmp_path / "ckpt" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _generate(tmp_path, out, *, checkpoint=b"weights-1", names=("a.jpg", "b.jpg")):
    """A real manifest and history line, as a mask run writes them."""
    os.makedirs(out, exist_ok=True)
    return run_record.write_run_manifest(
        "mask_generation",
        input_dir=str(tmp_path),
        output_dir=str(out),
        input_names=list(names),
        report=BatchReport(total=len(names), written=list(names), batch_size=2),
        models={"mask": _checkpoint(tmp_path, f"m-{checkpoint.decode()}.pth", checkpoint)},
        parameters={"batch_size": 2, "skip_existing": False},
    )


def _manifest(out) -> dict:
    with open(provenance.run_manifest_path(out), encoding="utf-8") as fh:
        return json.load(fh)


def _write_manifest_bytes(out, data: bytes) -> None:
    os.makedirs(out, exist_ok=True)
    with open(provenance.run_manifest_path(out), "wb") as fh:
        fh.write(data)


def _write_history_lines(out, lines) -> None:
    os.makedirs(out, exist_ok=True)
    with open(provenance.run_history_path(out), "wb") as fh:
        for line in lines:
            fh.write((line if isinstance(line, bytes) else json.dumps(line).encode()) + b"\n")


# ------------------------------------------------------------------ the reader
def test_a_real_manifest_and_history_are_summarised(tmp_path):
    out = tmp_path / "mask"
    path = _generate(tmp_path, out)
    record = _manifest(out)

    summary = generation_summary(str(out))

    with open(path, "rb") as fh:
        assert summary["record_sha256"] == hashlib.sha256(fh.read()).hexdigest()
    assert summary["record"] == provenance.RUN_MANIFEST_FILENAME == "_sprouts_run.json"
    assert summary["kind"] == record["kind"] == "mask_generation"
    assert summary["created_utc"] == record["created_utc"]
    assert summary["app_version"] == record["app"]["version"]
    assert summary["git_commit"] == record["git_commit"]
    assert summary["status"] == record["outputs"]["status"] == "completed"
    assert summary["models"] == {"mask": hashlib.sha256(b"weights-1").hexdigest()}
    assert summary["parameters"] == record["parameters"]
    for key in ("torch", "inference"):
        if key in record["runtime"]:
            assert summary[key] == record["runtime"][key]
        else:
            assert key not in summary
    assert "provenance_error" not in summary and "error" not in summary
    assert summary["history"] == {
        "file": "_sprouts_runs.jsonl",
        "n_runs": 1,
        "n_unreadable": 0,
        "models": {"mask": [hashlib.sha256(b"weights-1").hexdigest()]},
        "app_versions": [record["app"]["version"]],
        "git_commits": [record["git_commit"]] if record["git_commit"] else [],
        "norm_stats": [],  # a mask run has none
        "batch_sizes": [2],
        "batch_policies": [],  # this test's run records none
        "first_utc": record["created_utc"],
        "last_utc": record["created_utc"],
    }
    json.dumps(summary)  # JSON-able


def test_a_folder_no_generate_run_wrote_into_has_no_summary(tmp_path):
    assert generation_summary(str(tmp_path)) is None
    assert generation_summary(str(tmp_path / "missing")) is None


def test_history_only_has_no_record(tmp_path):
    out = tmp_path / "mask"
    _generate(tmp_path, out)
    os.remove(provenance.run_manifest_path(out))

    summary = generation_summary(str(out))

    assert set(summary) == {"history"}
    assert summary["history"]["n_runs"] == 1


def test_two_runs_of_different_weights_and_versions_are_both_listed(tmp_path):
    out = tmp_path / "skeletons"
    lines = [
        {
            "created_utc": "2026-11-02T10:00:00Z",
            "app_version": "3.1.0",
            "git_commit": "b" * 40,
            "models": {"skeleton": "f" * 64, "mask": None},
        },
        {
            "created_utc": "2026-10-01T09:00:00Z",
            "app_version": "3.0.0",
            "git_commit": "a" * 40,
            "models": {"skeleton": "0" * 64},
        },
        {
            "created_utc": "2026-11-03T10:00:00Z",
            "app_version": "3.1.0",
            "git_commit": None,
            "models": {"skeleton": "f" * 64},
        },
    ]
    _write_history_lines(out, lines)

    history = generation_summary(str(out))["history"]

    assert history["n_runs"] == 3
    assert history["n_unreadable"] == 0
    assert history["models"] == {"skeleton": ["0" * 64, "f" * 64], "mask": []}
    assert history["app_versions"] == ["3.0.0", "3.1.0"]
    assert history["git_commits"] == ["a" * 40, "b" * 40]
    assert history["first_utc"] == "2026-10-01T09:00:00Z"
    assert history["last_utc"] == "2026-11-03T10:00:00Z"


def test_a_history_line_keeps_only_the_named_fields(tmp_path):
    out = tmp_path / "mask"
    names = [f"{i}.jpg" for i in range(50)]
    _generate(tmp_path, out, names=names)

    history = generation_summary(str(out))["history"]

    assert "written" not in json.dumps(history)
    assert set(history) == {
        "file",
        "n_runs",
        "n_unreadable",
        "models",
        "app_versions",
        "git_commits",
        "norm_stats",
        "batch_sizes",
        "batch_policies",
        "first_utc",
        "last_utc",
    }


def test_a_history_line_carries_the_runs_mode_parameters():
    record = provenance.build_record(
        "skeleton_generation",
        parameters={"batch_size": 4, "batch_policy": "auto", "norm_stats": "image", "x": 1},
    )
    entry = run_record.history_entry(record, BatchReport(written=["a.jpg"]))
    assert entry["norm_stats"] == "image"
    assert entry["batch_size"] == 4
    assert entry["batch_policy"] == "auto"
    assert "x" not in entry

    mask = provenance.build_record("mask_generation", parameters={"batch_size": 8})
    entry = run_record.history_entry(mask, BatchReport(written=["a.jpg"]))
    assert entry["batch_size"] == 8
    assert "norm_stats" not in entry and "batch_policy" not in entry
    minimal = provenance.minimal_record("mask_generation", outputs={}, reason="OSError: no")
    entry = run_record.history_entry(minimal, BatchReport(written=["a.jpg"]))
    assert not {"norm_stats", "batch_size", "batch_policy"} & set(entry)


def test_two_runs_in_two_modes_are_both_listed(tmp_path):
    """A *Keep predictions* run after the switch was set adds predictions of another mode
    (and here another batch size) to the folder: the history lists both."""
    out = tmp_path / "skeletons"
    _write_history_lines(
        out,
        [
            {"models": {"skeleton": "f" * 64}, "norm_stats": "running", "batch_size": 8},
            {"models": {"skeleton": "f" * 64}, "norm_stats": "image", "batch_size": 4},
            {"models": {"skeleton": "f" * 64}, "batch_policy": "fixed", "norm_stats": None},
            {"models": {"skeleton": "f" * 64}},  # an old line: none of the three
        ],
    )

    history = generation_summary(str(out))["history"]

    assert history["n_runs"] == 4
    assert history["norm_stats"] == ["image", "running"]
    assert history["batch_sizes"] == [4, 8]
    assert history["batch_policies"] == ["fixed"]


def test_wrong_typed_mode_parameters_are_left_out(tmp_path):
    out = tmp_path / "skeletons"
    _write_history_lines(
        out,
        [
            {"norm_stats": 3, "batch_size": "8", "batch_policy": ["auto"]},
            {"norm_stats": {"a": 1}, "batch_size": True, "batch_policy": None},
            {"batch_size": 2.5},
            {"norm_stats": "image", "batch_size": 4, "batch_policy": "auto"},
        ],
    )

    history = generation_summary(str(out))["history"]

    assert history["n_runs"] == 4 and history["n_unreadable"] == 0
    assert history["norm_stats"] == ["image"]
    assert history["batch_sizes"] == [4]
    assert history["batch_policies"] == ["auto"]
    json.dumps(history)


def test_a_minimal_record_carries_its_provenance_error(tmp_path):
    out = tmp_path / "mask"
    os.makedirs(out)
    record = provenance.minimal_record(
        "mask_generation", outputs={"status": "completed"}, reason="OSError: no"
    )
    provenance.write_run_manifest(str(out), record)

    summary = generation_summary(str(out))

    assert summary["provenance_error"] == "OSError: no"
    assert summary["models"] == {}
    assert summary["git_commit"] is None
    assert summary["status"] == "completed"
    assert summary["parameters"] == {}
    assert "torch" not in summary and "inference" not in summary
    assert "history" not in summary


@pytest.mark.parametrize(
    "data",
    [b"", b"not json", b"[1, 2, 3]", b'"text"', b"42", b"\xff\xfe{}", b'{"kind": "x"'],
    ids=["empty", "not-json", "list", "string", "number", "bad-utf8", "truncated"],
)
def test_a_broken_manifest_is_an_error_and_the_history_still_counts(tmp_path, data):
    out = tmp_path / "mask"
    _write_manifest_bytes(out, data)
    _write_history_lines(out, [{"app_version": "3.0.0", "models": {"mask": "a" * 64}}])

    summary = generation_summary(str(out))

    assert summary["record"] == "_sprouts_run.json"
    assert isinstance(summary["error"], str) and summary["error"]
    assert set(summary) == {"record", "error", "history"}
    assert summary["history"]["n_runs"] == 1
    json.dumps(summary)


@pytest.mark.parametrize("kind", sorted(NESTED))
def test_a_deeply_nested_manifest_and_history_never_raise(tmp_path, small_stack_json, kind):
    out = tmp_path / "mask"
    _write_manifest_bytes(out, NESTED[kind].encode())
    _write_history_lines(out, [NESTED[kind].encode(), {"app_version": "3.0.0"}])

    summary = generation_summary(str(out))

    assert summary["record"] == "_sprouts_run.json"
    assert "error" in summary
    assert summary["history"]["n_runs"] == 1
    assert summary["history"]["n_unreadable"] == 1


@pytest.mark.parametrize("kind", sorted(NESTED))
def test_a_deeply_nested_manifest_parsed_on_a_big_stack_never_raises(tmp_path, kind):
    # Parsed (a big stack) or not, the reader gives a summary that can be written.
    out = tmp_path / "mask"
    _write_manifest_bytes(out, NESTED[kind].encode())

    summary = generation_summary(str(out))

    assert summary["record"] == "_sprouts_run.json"
    json.dumps(summary, indent=2)


def test_a_deeply_nested_field_is_left_out(tmp_path):
    out = tmp_path / "mask"
    deep: dict = {}
    node = deep
    for _ in range(200):
        node["a"] = {}
        node = node["a"]
    _write_manifest_bytes(out, json.dumps({"kind": "mask_generation", "parameters": deep}).encode())

    summary = generation_summary(str(out))

    assert summary["kind"] == "mask_generation"
    assert "parameters" not in summary
    json.dumps(summary, indent=2)


def test_wrong_typed_fields_are_left_out(tmp_path):
    out = tmp_path / "mask"
    record = {
        "kind": 7,
        "created_utc": ["2026"],
        "app": "SPROUTS 3",
        "git_commit": 12,
        "outputs": [],
        "models": ["mask"],
        "parameters": "batch 8",
        "runtime": {"torch": "2.15", "inference": 3},
        "provenance_error": {"why": "no"},
    }
    _write_manifest_bytes(out, json.dumps(record).encode())

    summary = generation_summary(str(out))

    assert set(summary) == {"record", "record_sha256"}


def test_wrong_typed_inner_fields_are_left_out(tmp_path):
    out = tmp_path / "mask"
    record = {
        "app": {"version": 3},
        "outputs": {"status": None},
        "models": {"mask": "abc", "skeleton": {"sha256": 5}, "ok": {"sha256": "d" * 64}},
        "runtime": [],
    }
    _write_manifest_bytes(out, json.dumps(record).encode())

    summary = generation_summary(str(out))

    assert "app_version" not in summary
    assert summary["status"] is None  # recorded as null: copied as recorded
    assert summary["models"] == {"ok": "d" * 64}
    assert "torch" not in summary and "inference" not in summary


def test_a_malformed_history_line_is_counted_and_the_rest_kept(tmp_path):
    out = tmp_path / "mask"
    _write_history_lines(
        out,
        [
            {"app_version": "3.0.0", "models": {"mask": "a" * 64}},
            b"not json",
            b"42",
            b"[1]",
            b"\xff\xfe",
            b"",
            {"app_version": 3, "git_commit": ["x"], "models": ["mask"], "created_utc": 5},
            {"app_version": "3.1.0", "models": {"mask": "b" * 64, "skeleton": 7}},
        ],
    )

    history = generation_summary(str(out))["history"]

    assert history["n_runs"] == 3
    assert history["n_unreadable"] == 4  # the blank line is not a line
    assert history["models"] == {"mask": ["a" * 64, "b" * 64]}
    assert history["app_versions"] == ["3.0.0", "3.1.0"]
    assert history["git_commits"] == []
    assert history["first_utc"] is None and history["last_utc"] is None


_GOOD = {
    "kind": "mask_generation",
    "created_utc": "2026-11-10T00:00:00Z",
    "app": {"version": "3.0.0"},
    "git_commit": "abc",
    "outputs": {"status": "completed"},
    "models": {"mask": {"sha256": "d" * 64}},
    "parameters": {"batch_size": 4},
}


def _with_lone_surrogate(record: dict) -> bytes:
    """``record`` as JSON with its ``"x"`` value written as the escape of a lone surrogate:
    valid JSON in valid UTF-8 bytes, but no text that UTF-8 can hold once parsed."""
    return json.dumps(record).replace('"x"', '"\\ud800"').encode()


_SURROGATES = {
    "parameters": (_with_lone_surrogate({**_GOOD, "parameters": {"note": "x"}}), None),
    "git_commit": (_with_lone_surrogate({**_GOOD, "git_commit": "x"}), None),
    "history-app-version": (
        None,
        _with_lone_surrogate({"app_version": "x", "models": {"mask": "d" * 64}}),
    ),
}


def _write_surrogate_case(out, case: str) -> None:
    manifest, history = _SURROGATES[case]
    os.makedirs(out, exist_ok=True)
    if manifest is not None:
        _write_manifest_bytes(out, manifest)
    if history is not None:
        _write_history_lines(out, [history])


@pytest.mark.parametrize("case", sorted(_SURROGATES))
def test_a_lone_surrogate_gives_the_error_form(tmp_path, case):
    out = tmp_path / "mask"
    _write_surrogate_case(out, case)

    summary = generation_summary(str(out))

    assert set(summary) == {"record", "error"}  # no partial copy
    assert summary["record"] == "_sprouts_run.json"
    assert "surrogates not allowed" in summary["error"]
    json.dumps(summary, ensure_ascii=False).encode("utf-8")


@pytest.mark.parametrize("case", sorted(_SURROGATES))
def test_a_lone_surrogate_keeps_the_measure_sidecar(tmp_path, case):
    folder, images = _project(tmp_path, AREA)
    _write_surrogate_case(folder, case)

    _, record = _measure(AREA, images, folder, tmp_path)

    assert record is not None  # the sidecar was written
    assert record["calibration"]["source"] == "default"
    assert set(record["inputs"]["generation"]) == {"record", "error"}


def test_an_unreadable_manifest_is_an_error(tmp_path):
    out = tmp_path / "mask"
    os.makedirs(os.path.join(out, provenance.RUN_MANIFEST_FILENAME))  # a folder, not a file
    os.makedirs(os.path.join(out, provenance.RUN_HISTORY_FILENAME))

    summary = generation_summary(str(out))

    assert summary["record"] == "_sprouts_run.json"
    assert "error" in summary
    assert summary["history"]["file"] == "_sprouts_runs.jsonl"
    assert "error" in summary["history"]
    json.dumps(summary)


def test_the_bytes_hashed_are_the_bytes_parsed(tmp_path, monkeypatch):
    out = tmp_path / "mask"
    _write_manifest_bytes(out, json.dumps({"kind": "mask_generation"}).encode())
    opened = []
    real_open = open

    def counting_open(path, *args, **kwargs):
        if os.path.basename(os.fspath(path)) == provenance.RUN_MANIFEST_FILENAME:
            opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)
    summary = generation_summary(str(out))

    assert len(opened) == 1
    assert summary["kind"] == "mask_generation"


# ------------------------------------------------------------------ the Measure sidecar
def _write_png(path, img) -> None:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(str(path))


def _project(tmp_path, spec):
    """A prediction folder of two images, as Generate leaves it (without its record)."""
    folder = tmp_path / ("skeletons" if spec is LENGTH else "mask")
    folder.mkdir(parents=True)
    images = {}
    for i, name in enumerate(_NAMES):
        img = np.zeros((256, 341), np.uint8)
        cv2.line(img, (10 + i, 10), (200, 120 + i), 255, 3)
        path = folder / (f"{name}_fake.png" if spec is LENGTH else f"{name}.png")
        _write_png(path, img)
        images[name] = str(path)
    return folder, images


def _measure(spec, images, folder, tmp_path):
    run = MeasureRun(spec, images, str(folder), project_root=str(tmp_path))
    run.compute()
    csv_path = run.write()
    with open(csv_path, "rb") as fh:
        csv_bytes = fh.read()
    return csv_bytes, provenance.read_sidecar(csv_path)


@pytest.mark.parametrize("spec", [LENGTH, AREA], ids=["length", "area"])
def test_the_sidecar_names_the_generation_run(tmp_path, spec):
    folder, images = _project(tmp_path, spec)
    without_csv, without = _measure(spec, images, folder, tmp_path)
    assert "generation" in without["inputs"] and without["inputs"]["generation"] is None
    with open(provenance.sidecar_path(folder / spec.csv_name), encoding="utf-8") as fh:
        assert '"generation": null' in fh.read()

    _generate(tmp_path, folder, names=[f"{n}.jpg" for n in _NAMES])
    with_csv, record = _measure(spec, images, folder, tmp_path)

    assert with_csv == without_csv
    assert record["inputs"]["generation"] == generation_summary(str(folder))
    assert record["inputs"]["generation"]["kind"] == "mask_generation"
    assert record["schema"] == "sprouts-provenance/1"
    assert record["models"] == {}


def _tagged_manifest(folder, tag: str) -> None:
    record = {**_GOOD, "git_commit": tag, "created_utc": tag}
    _write_manifest_bytes(folder, json.dumps(record).encode())


@pytest.mark.parametrize("spec", [LENGTH, AREA], ids=["length", "area"])
def test_a_retry_of_the_write_names_the_run_the_rows_were_measured_from(tmp_path, spec):
    """The generation record is read when the rows are measured: a Retry of the write after
    another Generate run still names the run behind the rows."""
    folder, images = _project(tmp_path, spec)
    _tagged_manifest(folder, "run1")
    run = MeasureRun(spec, images, str(folder), project_root=str(tmp_path))
    run.compute()
    _tagged_manifest(folder, "run2")  # a Generate run while the Retry box was open

    run.write()

    generation = provenance.read_sidecar(run.csv_path)["inputs"]["generation"]
    assert generation["git_commit"] == "run1"


def test_a_stopped_run_reads_no_generation_record(tmp_path, monkeypatch):
    folder, images = _project(tmp_path, AREA)
    read = []
    monkeypatch.setattr(run_record, "generation_summary", lambda folder: read.append(folder))
    run = MeasureRun(AREA, images, str(folder), project_root=str(tmp_path))
    assert run.compute(should_stop=lambda: True) is None
    assert read == []


def test_rows_set_without_compute_write_the_record_read_at_write_time(tmp_path):
    folder, images = _project(tmp_path, AREA)
    _tagged_manifest(folder, "run1")
    run = MeasureRun(AREA, images, str(folder), project_root=str(tmp_path))
    run.rows = []  # as tests and tools build a run
    _tagged_manifest(folder, "run2")

    run.write()

    generation = provenance.read_sidecar(run.csv_path)["inputs"]["generation"]
    assert generation["git_commit"] == "run2"


@pytest.mark.parametrize("spec", [LENGTH, AREA], ids=["length", "area"])
def test_every_key_the_sidecar_had_is_still_there(tmp_path, spec, monkeypatch):
    folder, images = _project(tmp_path, spec)
    _generate(tmp_path, folder, names=[f"{n}.jpg" for n in _NAMES])
    built = []
    real_build = measure_run.provenance.build_record

    def recording_build(*args, **kwargs):
        record = real_build(*args, **kwargs)
        built.append(record)
        return record

    monkeypatch.setattr(measure_run.provenance, "build_record", recording_build)
    _, record = _measure(spec, images, folder, tmp_path)

    (built_record,) = built
    inputs = dict(built_record["inputs"])
    assert list(inputs) == [  # the keys before, in their order, then the new one
        "folder",
        "n_images",
        "names_sha256",
        "n_without_image",
        "n_duplicate",
        "without_image",
        "duplicates",
        "sizes_px",
        "generation",
        "photos",
    ]
    del inputs["generation"], inputs["photos"]
    before = real_build(
        spec.activity,
        inputs=inputs,
        parameters=built_record["parameters"],
        outputs=built_record["outputs"],
        calibration=built_record["calibration"],
    )
    for key in ("created_utc", "runtime"):  # time and loaded libraries vary per call
        before[key] = record[key]
    assert list(record) == list(before)
    assert list(record["inputs"])[:-2] == list(before["inputs"])
    for key, value in before.items():
        if key == "inputs":
            for name, inner in value.items():
                assert record["inputs"][name] == inner
        else:
            assert record[key] == value, key
