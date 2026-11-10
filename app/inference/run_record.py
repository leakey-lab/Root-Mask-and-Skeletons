"""Provenance manifest for one generation run (FR-02 / PI-09).

Every mask or skeleton run writes ``<output_dir>/_sprouts_run.json``
(``app.provenance.RUN_MANIFEST_FILENAME``): app version and git commit, model
checkpoint SHA-256, the resolved inference runtime (device, precision "tf32"/"fp32",
cuDNN benchmark/deterministic flags), run parameters, a digest of the input names,
and the outcome (status "completed" / "cancelled" / "failed" with the error, counts,
failed inputs with reasons, skipped stems, cancel flag).
Every image scanner filters by image extension, so the JSON is never read as data.
The file describes the latest run into that folder that wrote or failed an input
(including one that raised after writing outputs, R7); a run that did neither
leaves it alone, so a *Keep predictions* run with nothing missing, a cancel
before the first batch and a run that raised before writing anything keep the
record of the run that made the outputs. Every such run is also one line of
``_sprouts_runs.jsonl`` (``provenance.RUN_HISTORY_FILENAME``; O16). When the full record cannot be
built or written, a minimal one (``provenance.minimal_record``: the outcome and
``provenance_error``) replaces it, and if even that fails the outdated file is
removed: the folder never describes an older run than its outputs (R5).

:func:`generation_summary` reads both files back for a Measure run, whose CSV sits in the
same folder as the predictions it measured: the CSV's sidecar then names the generation run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from app import provenance
from app.inference.batch_engine import BatchReport

logger = logging.getLogger(__name__)

#: A history line lists the written names up to this many; beyond it, a count and a digest.
HISTORY_NAMES_LIMIT = 1000
#: A mapping copied into a generation summary is nested at most this deep; a deeper one is
#: left out, so the sidecar that holds the summary can always be written (its pretty-printing
#: encoder recurses once per level).
SUMMARY_DEPTH_LIMIT = 32
#: A field the manifest does not have (unlike one recorded as ``null``).
_MISSING = object()


def _reason(exc: BaseException) -> str:
    """``"<ExceptionType>: <message>"``, the form ``outputs.error`` uses."""
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def write_run_manifest(
    kind: str,
    *,
    input_dir: str,
    output_dir: str,
    input_names: Sequence[str],
    report: BatchReport,
    models: Mapping[str, str],
    parameters: Mapping[str, Any],
) -> str | None:
    """Write the run manifest and append the run to the history; returns the manifest path.

    A run that wrote nothing, failed nothing and did not raise (a *Keep
    predictions* run with nothing missing, a cancel before the first batch) changed nothing:
    it writes nothing and returns None, so the record of the run that made the outputs
    stays. A provenance failure is logged and never fails the run (the outputs exist);
    the minimal record, or else the removal of the outdated one, keeps the folder from
    describing an older run (R5); None means it could not be written. Call from the
    worker thread: the first call hashes the checkpoints (~0.3 s).
    """
    if not report.written and not report.failed and report.error is None:
        logger.info("Run changed nothing in %s; the run manifest is left as is", output_dir)
        return None
    try:
        record = provenance.build_record(
            kind,
            inputs={
                "folder": provenance.portable_path(input_dir),
                "n_images": len(input_names),
                "names_sha256": provenance.names_digest(input_names),
            },
            models=models,
            parameters=parameters,
            outputs=report.summary(),
        )
        path = provenance.write_run_manifest(output_dir, record)
    except Exception as exc:  # noqa: BLE001 - (R7) any provenance failure, not only I/O
        logger.warning("Could not write the run manifest in %s: %s", output_dir, exc, exc_info=exc)
        return _write_minimal_manifest(kind, output_dir, report, exc)
    _append_history(output_dir, record, report)
    return path


def history_entry(record: Mapping[str, Any], report: BatchReport) -> dict[str, Any]:
    """The line of ``_sprouts_runs.jsonl`` for one run: the record's identity and the outcome.

    Written names are listed (sorted) up to ``HISTORY_NAMES_LIMIT``; a longer run
    carries ``written_count`` and ``written_sha256`` instead, so a line stays small.
    """
    app = record.get("app") or {}
    models = record.get("models") or {}
    entry: dict[str, Any] = {
        "created_utc": record.get("created_utc"),
        "kind": record.get("kind"),
        "app_version": app.get("version"),
        "git_commit": record.get("git_commit"),
        "models": {name: (info or {}).get("sha256") for name, info in models.items()},
        "status": report.status,
        "n_written": len(report.written),
        "n_failed": len(report.failed),
        "n_skipped": len(report.skipped),
    }
    if len(report.written) <= HISTORY_NAMES_LIMIT:
        entry["written"] = sorted(report.written)
    else:
        entry["written_count"] = len(report.written)
        entry["written_sha256"] = provenance.names_digest(report.written)
    return entry


def _append_history(output_dir: str, record: Mapping[str, Any], report: BatchReport) -> None:
    """Append this run to the history; a failure is logged and never fails the run."""
    try:
        provenance.append_run_history(output_dir, history_entry(record, report))
    except Exception as exc:  # noqa: BLE001 - same rule as the manifest: the outputs exist
        logger.warning("Could not append to the run history in %s: %s", output_dir, exc)


def _write_minimal_manifest(
    kind: str, output_dir: str, report: BatchReport, cause: BaseException
) -> str | None:
    """(R5) Replace the manifest by a minimal record, else remove the outdated one.

    Never raises. Returns the path written, or None.
    """
    try:
        record = provenance.minimal_record(kind, outputs=report.summary(), reason=_reason(cause))
        path = provenance.write_run_manifest(output_dir, record)
    except Exception as exc:  # noqa: BLE001 - e.g. the disk is full: nothing can be written
        logger.warning("Could not write a minimal run manifest in %s: %s", output_dir, exc)
    else:
        _append_history(output_dir, record, report)
        return path
    path = provenance.run_manifest_path(output_dir)
    try:
        os.remove(path)
        logger.warning("Removed the outdated run manifest %s: it described an older run", path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.error("Could not remove the outdated run manifest %s: %s", path, exc)
    return None


def record_failed_run(
    report: BatchReport, error: BaseException, write_manifest: Callable[[], object]
) -> None:
    """Describe a run that raised (R7); call from its ``except`` block, then re-raise.

    Outputs already on disk changed the folder, so ``write_manifest()`` replaces the
    previous manifest with one for this run (``status`` "failed", the error, the
    partial counts). A run that wrote nothing leaves the previous manifest alone: it
    still describes the outputs there. Never raises, so the run's own error is the
    one that propagates.
    """
    try:
        report.mark_failed(error)
        if report.written:
            write_manifest()
        else:
            logger.info("Run failed before writing any output; the run manifest is left as is")
    except Exception as exc:  # noqa: BLE001 - must not mask the run's own error
        logger.error("Could not record the failed run: %s", exc, exc_info=exc)


def generation_summary(output_dir: str) -> dict[str, Any] | None:
    """Which generation run made the predictions in ``output_dir``, for a Measure sidecar.

    ``None`` when the folder has neither ``_sprouts_run.json`` nor ``_sprouts_runs.jsonl``
    (no Generate run of this app wrote into it: predictions copied in by hand, GT only).
    Otherwise a small JSON-able dict: ``record`` and ``record_sha256`` (the manifest's name
    and the SHA-256 of the bytes parsed), the manifest's ``kind``, ``created_utc``,
    ``app_version``, ``git_commit``, ``status``, ``models`` (``{name: sha256}``),
    ``parameters``, ``torch`` and ``inference`` (from ``runtime``) and, for a minimal record,
    ``provenance_error``; and ``history``, the distinct model digests, app versions and
    commits of every run in ``_sprouts_runs.jsonl``. The manifest describes the latest run
    into the folder that wrote or failed an input; a prediction kept from an earlier run
    (*Keep predictions*) can come from another run, which is why the history's distinct
    digests are listed.

    Never raises for anything those two files hold: a manifest that cannot be read or is not
    a JSON object gives ``record`` and ``error`` only (the history is still summarised), a
    field of the wrong type (or nested deeper than ``SUMMARY_DEPTH_LIMIT``) is left out, and
    a history line that is not a JSON object is counted in ``n_unreadable``.
    """
    try:
        summary = _manifest_summary(provenance.run_manifest_path(output_dir))
        history = _history_summary(provenance.run_history_path(output_dir))
    except Exception as exc:  # noqa: BLE001 - a Measure run must never fail on its sidecar
        logger.warning("Could not summarise the generation record of %s: %s", output_dir, exc)
        return {"record": provenance.RUN_MANIFEST_FILENAME, "error": _reason(exc)}
    if summary is None and history is None:
        return None
    summary = summary if summary is not None else {}
    if history is not None:
        summary["history"] = history
    return summary


def _manifest_summary(path: str) -> dict[str, Any] | None:
    """The manifest's part of :func:`generation_summary`; ``None`` when there is no file."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except FileNotFoundError:
        return None
    except OSError as exc:  # a folder of that name, no permission, a read error
        return {"record": provenance.RUN_MANIFEST_FILENAME, "error": _reason(exc)}
    try:
        record = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError) as exc:  # bad UTF-8, bad JSON, nested too deep
        return {"record": provenance.RUN_MANIFEST_FILENAME, "error": _reason(exc)}
    if not isinstance(record, dict):
        return {
            "record": provenance.RUN_MANIFEST_FILENAME,
            "error": f"not a JSON object: {type(record).__name__}",
        }
    summary: dict[str, Any] = {
        "record": provenance.RUN_MANIFEST_FILENAME,
        "record_sha256": hashlib.sha256(data).hexdigest(),
    }
    app = record.get("app")
    outputs = record.get("outputs")
    runtime = record.get("runtime")
    fields = {
        "kind": record.get("kind", _MISSING),
        "created_utc": record.get("created_utc", _MISSING),
        "app_version": app.get("version", _MISSING) if isinstance(app, dict) else _MISSING,
        "git_commit": record.get("git_commit", _MISSING),
        "status": outputs.get("status", _MISSING) if isinstance(outputs, dict) else _MISSING,
    }
    for key, value in fields.items():
        if value is None or isinstance(value, str):
            summary[key] = value
    models = record.get("models")
    if isinstance(models, dict):
        summary["models"] = {
            name: info["sha256"]
            for name, info in models.items()
            if isinstance(info, dict) and _text_or_none(info.get("sha256", _MISSING))
        }
    if _shallow_mapping(record.get("parameters")):
        summary["parameters"] = record["parameters"]
    if isinstance(runtime, dict):
        for key in ("torch", "inference"):
            if _shallow_mapping(runtime.get(key)):
                summary[key] = runtime[key]
    if isinstance(record.get("provenance_error"), str):
        summary["provenance_error"] = record["provenance_error"]
    return summary


def _history_summary(path: str) -> dict[str, Any] | None:
    """The history's part of :func:`generation_summary`; ``None`` when there is no file.

    Read line by line; of each line only its models, app version, commit and time are kept
    (a line can list 1,000 file names). A blank line is not a run."""
    n_runs = n_unreadable = 0
    models: dict[str, set[str]] = {}
    versions: set[str] = set()
    commits: set[str] = set()
    times: set[str] = set()
    try:
        with open(path, "rb") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                try:
                    entry = json.loads(raw.decode("utf-8"))
                except ValueError, RecursionError:
                    entry = None
                if not isinstance(entry, dict):
                    n_unreadable += 1
                    continue
                n_runs += 1
                digests = entry.get("models")
                if isinstance(digests, dict):
                    for name, digest in digests.items():
                        if isinstance(digest, str):
                            models.setdefault(name, set()).add(digest)
                        elif digest is None:
                            models.setdefault(name, set())
                for value, seen in (
                    (entry.get("app_version"), versions),
                    (entry.get("git_commit"), commits),
                    (entry.get("created_utc"), times),
                ):
                    if isinstance(value, str):
                        seen.add(value)
    except FileNotFoundError:
        return None
    except OSError as exc:  # a folder of that name, no permission, a read error
        return {"file": provenance.RUN_HISTORY_FILENAME, "error": _reason(exc)}
    return {
        "file": provenance.RUN_HISTORY_FILENAME,
        "n_runs": n_runs,
        "n_unreadable": n_unreadable,
        "models": {name: sorted(digests) for name, digests in models.items()},
        "app_versions": sorted(versions),
        "git_commits": sorted(commits),
        "first_utc": min(times) if times else None,
        "last_utc": max(times) if times else None,
    }


def _text_or_none(value: object) -> bool:
    """True for a recorded digest: text, or ``null`` (a checkpoint that could not be read)."""
    return value is None or isinstance(value, str)


def _shallow_mapping(value: object) -> bool:
    """True when ``value`` is a dict nested at most ``SUMMARY_DEPTH_LIMIT`` deep (no recursion)."""
    if not isinstance(value, dict):
        return False
    stack: list[tuple[object, int]] = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        children = node.values() if isinstance(node, dict) else node
        for child in children:  # type: ignore[union-attr]
            if isinstance(child, (dict, list)):
                if depth >= SUMMARY_DEPTH_LIMIT:
                    return False
                stack.append((child, depth + 1))
    return True


__all__ = [
    "HISTORY_NAMES_LIMIT",
    "SUMMARY_DEPTH_LIMIT",
    "generation_summary",
    "history_entry",
    "record_failed_run",
    "write_run_manifest",
]
