"""The inference runtime (``app/inference/runtime.py``): device, precision, caches, batch policy.

Promises: importing it loads no torch; the device falls back to the CPU when CUDA is absent or
has no device (R3); cuDNN is deterministic unless ``SPROUTS_CUDNN_BENCHMARK`` opts in; the
precision is reported truthfully; ``describe_runtime`` never loads a model; a missing mask
checkpoint is ``ModelUnavailableError``; only a GPU run limits its pool threads' torch threads;
releasing the CUDA cache runs once, only with CUDA, and never raises (R2). Batch policy (O4):
``SPROUTS_BATCH_POLICY``, the owner's fixed sizes, the pre-flight floors of free GPU memory
(the free amount rounded down in the message), and the sentence the user sees for a memory
stop, without a traceback. CPU-only safe; GPU memory is simulated.
"""

from __future__ import annotations

import dataclasses
import logging
import subprocess
import sys

import pytest
import torch

from app import config
from app.inference import runtime
from app.inference.batch_engine import (
    FIXED_MEMORY_MESSAGE,
    FIXED_OOM_MESSAGE,
    FixedBatchMemoryTooLow,
    FixedBatchOutOfMemory,
)


@pytest.fixture
def fresh_runtime(monkeypatch):
    """Resolve the runtime from scratch and restore the global cuDNN flags afterwards."""
    import torch

    saved = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    runtime.get_runtime.cache_clear()
    yield runtime
    runtime.get_runtime.cache_clear()
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = saved


def test_importing_runtime_does_not_import_torch():
    code = (
        "import sys, app.inference.runtime as r; "
        "print('torch' in sys.modules, r.peek_runtime() is None)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    ).stdout.split()
    assert out == ["False", "True"]


def test_cudnn_benchmark_is_off_and_deterministic_by_default(fresh_runtime, monkeypatch):
    import torch

    monkeypatch.delenv(runtime.CUDNN_BENCHMARK_ENV, raising=False)
    torch.backends.cudnn.benchmark = True
    rt = fresh_runtime.get_runtime()
    assert rt.cudnn_benchmark is False
    assert rt.cudnn_deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


def test_cudnn_benchmark_env_opt_in(fresh_runtime, monkeypatch):
    import torch

    monkeypatch.setenv(runtime.CUDNN_BENCHMARK_ENV, "1")
    rt = fresh_runtime.get_runtime()
    assert rt.cudnn_benchmark is True and rt.cudnn_deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_precision_is_reported_truthfully(fresh_runtime, monkeypatch):
    monkeypatch.delenv(runtime.CUDNN_BENCHMARK_ENV, raising=False)
    rt = fresh_runtime.get_runtime()
    if rt.is_cuda and rt.compute_capability and rt.compute_capability >= (8, 0):
        assert rt.precision == "tf32"
    else:
        assert rt.precision == "fp32"


def test_peek_runtime_never_resolves(fresh_runtime):
    assert fresh_runtime.peek_runtime() is None
    rt = fresh_runtime.get_runtime()
    assert fresh_runtime.peek_runtime() is rt


def test_describe_runtime_before_resolution_loads_nothing(fresh_runtime, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("describe_runtime must not load a model")

    monkeypatch.setattr(runtime, "get_skeleton_model", boom)
    monkeypatch.setattr(runtime, "get_mask_model", boom)
    summary = fresh_runtime.describe_runtime()
    assert fresh_runtime.peek_runtime() is None
    assert summary.device == "unknown"
    assert isinstance(summary.text, str) and summary.text
    assert summary.mask_loaded is False and summary.skeleton_loaded is False


def test_describe_runtime_after_resolution(fresh_runtime, monkeypatch, tmp_path):
    missing = str(tmp_path / "nope.pth")
    monkeypatch.setattr(runtime, "mask_weights_path", lambda: missing)
    fresh_runtime.get_runtime()
    summary = fresh_runtime.describe_runtime()
    assert summary.device in ("GPU", "CPU")
    assert summary.mask_weights is False
    assert summary.text.startswith(summary.device)
    assert "mask model missing" in summary.text


def test_get_mask_model_reports_missing_weights(fresh_runtime, monkeypatch, tmp_path):
    # fresh_runtime: get_mask_model resolves the runtime first; it must not stay resolved
    runtime.get_mask_model.cache_clear()
    monkeypatch.setattr(runtime, "mask_weights_path", lambda: str(tmp_path / "nope.pth"))
    with pytest.raises(runtime.ModelUnavailableError, match="nope.pth"):
        runtime.get_mask_model()
    runtime.get_mask_model.cache_clear()


def test_only_a_gpu_run_gives_its_pool_threads_one_torch_thread():
    """(S12-01) A property, not a field: the run records stay as they were."""
    import torch

    cpu = runtime.RuntimeConfig(
        device=torch.device("cpu"),
        use_channels_last=False,
        pin_memory=False,
        io_threads=2,
        cudnn_benchmark=False,
        cudnn_deterministic=True,
        precision="fp32",
    )
    cuda = dataclasses.replace(cpu, device=torch.device("cuda"), use_channels_last=True)
    assert cuda.batch_torch_threads == 1
    assert cpu.batch_torch_threads is None
    assert "batch_torch_threads" not in {f.name for f in dataclasses.fields(runtime.RuntimeConfig)}


def test_release_cuda_cache_runs_once_and_only_with_cuda(monkeypatch):
    import torch

    calls = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    runtime.release_cuda_cache()
    assert calls == ["empty"]
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    runtime.release_cuda_cache()
    assert calls == ["empty"]


STICKY = "CUDA error: an illegal memory access was encountered (simulated, sticky)"


def _sticky_cuda(monkeypatch):
    """From now on CUDA looks initialised and every cache release fails (sticky error).

    Armed after the stub model is built: its ``torch.manual_seed`` would otherwise
    try to seed the (absent) CUDA generators.
    """
    calls = []

    def empty_cache():
        calls.append(1)
        raise RuntimeError(STICKY)

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    return calls


def test_release_cuda_cache_logs_a_failure_instead_of_raising(monkeypatch, caplog):
    calls = _sticky_cuda(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=runtime.__name__):
        runtime.release_cuda_cache()  # must not raise
    assert calls == [1]
    assert STICKY in caplog.text


def test_cuda_available_with_zero_devices_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    runtime.get_runtime.cache_clear()
    try:
        assert runtime.get_runtime().device.type == "cpu"
    finally:
        runtime.get_runtime.cache_clear()


GIB = 1024**3


@pytest.fixture
def gpu_memory(monkeypatch):
    """Fake ``torch.cuda.mem_get_info``; the release before it is recorded (no GPU use)."""
    state = {"free": 8 * GIB, "calls": 0, "events": []}

    def mem_get_info(device=None):
        state["calls"] += 1
        state["events"].append("measure")
        return state["free"], 16 * GIB

    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    monkeypatch.setattr(runtime, "release_cuda_cache", lambda: state["events"].append("release"))
    monkeypatch.delenv(runtime.SKELETON_NORM_STATS_ENV, raising=False)  # the default mode
    return state


@pytest.mark.parametrize(
    ("model", "n", "mode", "needed"),
    [
        ("mask", 4, None, 3.6),
        ("skeleton", 64, None, 4.7),  # the default norm statistics, "image"
        ("skeleton", 64, "running", 3.8),
        ("mask", 16, None, 12.0),
    ],
)
def test_too_little_free_memory_stops_before_the_first_batch(
    gpu_memory, monkeypatch, model, n, mode, needed
):
    gpu_memory["free"] = int(2.05 * GIB)
    if mode is not None:
        monkeypatch.setenv(runtime.SKELETON_NORM_STATS_ENV, mode)

    with pytest.raises(FixedBatchMemoryTooLow) as caught:
        runtime.check_fixed_batch_memory(model, n, "cuda")

    assert str(caught.value) == FIXED_MEMORY_MESSAGE.format(n=n, free=2.0, needed=needed)
    assert str(caught.value).startswith(
        f"Not enough free GPU memory for the fixed batch size ({n}): 2.0 GiB free, "
        f"{needed:.1f} GiB needed. "
    )
    assert isinstance(caught.value, FixedBatchOutOfMemory)
    assert caught.value.free_gib == pytest.approx(2.05)
    assert caught.value.needed_gib == needed
    assert gpu_memory["events"] == ["release", "measure"]  # collected + emptied first


def test_free_memory_just_below_the_floor_never_reads_as_enough(gpu_memory):
    gpu_memory["free"] = int(3.59 * GIB)
    with pytest.raises(FixedBatchMemoryTooLow, match=r"3\.5 GiB free, 3\.6 GiB needed"):
        runtime.check_fixed_batch_memory("mask", 4, "cuda")


def test_the_skeleton_floor_keeps_a_margin(gpu_memory):
    gpu_memory["free"] = int(3.7 * GIB)  # enough for the bare cap (about 3.4 GiB), no margin
    with pytest.raises(FixedBatchMemoryTooLow):
        runtime.check_fixed_batch_memory("skeleton", 64, "cuda", norm_stats="running")
    gpu_memory["free"] = int(4.6 * GIB)  # "image" (the default): its bare cap is 4.25 GiB
    with pytest.raises(FixedBatchMemoryTooLow):
        runtime.check_fixed_batch_memory("skeleton", 64, "cuda")


def test_enough_free_memory_returns_it(gpu_memory):
    gpu_memory["free"] = int(7.5 * GIB)
    assert runtime.check_fixed_batch_memory("mask", 4, "cuda") == pytest.approx(7.5, abs=0.01)


def test_a_size_without_a_measured_floor_is_measured_but_not_stopped(gpu_memory):
    gpu_memory["free"] = 1 * GIB
    assert runtime.check_fixed_batch_memory("mask", 2, "cuda") == pytest.approx(1.0)
    assert runtime.check_fixed_batch_memory("skeleton", 32, "cuda") == pytest.approx(1.0)


def test_the_cpu_never_asks_the_gpu(gpu_memory):
    assert runtime.check_fixed_batch_memory("mask", 4, "cpu") is None
    assert gpu_memory["calls"] == 0


def test_a_failed_memory_query_does_not_stop_the_run(monkeypatch, caplog):
    def broken(device=None):
        raise RuntimeError("driver says no (simulated)")

    monkeypatch.setattr(torch.cuda, "mem_get_info", broken)
    monkeypatch.setattr(runtime, "release_cuda_cache", lambda: None)
    with caplog.at_level(logging.WARNING):
        assert runtime.check_fixed_batch_memory("mask", 4, "cuda") is None
    assert "driver says no" in caplog.text


def test_batch_policy_reads_the_environment(monkeypatch, caplog):
    monkeypatch.delenv(runtime.BATCH_POLICY_ENV, raising=False)
    assert runtime.DEFAULT_BATCH_POLICY == "fixed"
    assert runtime.BATCH_POLICY_ENV == "SPROUTS_BATCH_POLICY"
    assert runtime.BATCH_POLICIES == ("fixed", "auto")
    assert runtime.batch_policy() == "fixed"
    monkeypatch.setenv(runtime.BATCH_POLICY_ENV, "AUTO")
    assert runtime.batch_policy() == "auto"
    monkeypatch.setenv(runtime.BATCH_POLICY_ENV, " fixed ")
    assert runtime.batch_policy() == "fixed"
    monkeypatch.setenv(runtime.BATCH_POLICY_ENV, "bogus")
    with caplog.at_level(logging.WARNING):
        assert runtime.batch_policy() == runtime.DEFAULT_BATCH_POLICY
    assert len([r for r in caplog.records if "bogus" in r.getMessage()]) == 1


def test_fixed_sizes_are_the_owner_sizes():
    # O4 memo: masks at 4 on a GPU (16 needs about 12 GiB free), skeletons at 64.
    assert runtime.fixed_batch_size("mask", "cuda") == 4 == config.MASK_BATCH_SIZE_CPU
    assert runtime.fixed_batch_size("skeleton", "cuda") == 64
    assert runtime.fixed_batch_size("mask", "cpu") == config.MASK_BATCH_SIZE_CPU == 4
    assert runtime.fixed_batch_size("skeleton", "cpu") == config.SKELETON_BATCH_SIZE_CPU == 8
    with pytest.raises(ValueError):
        runtime.fixed_batch_size("roots", "cuda")


@pytest.mark.parametrize(
    "exc",
    [FixedBatchOutOfMemory(4), FixedBatchMemoryTooLow(64, free_gib=2.0, needed_gib=3.8)],
    ids=["out-of-memory", "pre-flight"],
)
def test_a_fixed_memory_stop_is_shown_as_its_sentence(exc, caplog):
    from app.workers import describe_error, is_user_condition

    assert is_user_condition(exc)
    assert describe_error(exc) == str(exc)  # no "FixedBatchOutOfMemory: " prefix


def test_a_fixed_memory_stop_logs_no_traceback(qapp, caplog):
    from app.workers import LOGGER_NAME, GuardedThread

    class _Thread(GuardedThread):
        def work(self):
            raise FixedBatchOutOfMemory(4)

    messages = []
    thread = _Thread()
    thread.failed.connect(messages.append)
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        thread.run()  # synchronously, on this thread
    assert messages == [FIXED_OOM_MESSAGE.format(n=4)]
    assert not any(record.exc_info for record in caplog.records)


# --- the mask model cache: a strict load of the checkpoint ---------------------------------


@pytest.fixture
def mask_checkpoint(tmp_path, monkeypatch, cpu_runtime):
    """``torch.load`` gives the state dict of a seeded mask network (no 0.35 GB file is
    written); yields ``(state, weights path, {path: load arguments})``. The model cache is
    cleared before and after."""
    from app.mask_model.model import ResNetSkeleton

    torch.manual_seed(0)
    state = ResNetSkeleton(num_classes=1, pretrained=False).state_dict()
    weights = tmp_path / "best_mask_model_V5.pth"
    weights.write_bytes(b"stands for the checkpoint")
    monkeypatch.setattr(runtime, "mask_weights_path", lambda: str(weights))
    loaded = {}

    def fake_load(path, **kwargs):
        loaded[str(path)] = kwargs
        return dict(state)

    monkeypatch.setattr(torch, "load", fake_load)
    runtime.get_mask_model.cache_clear()
    yield state, str(weights), loaded
    runtime.get_mask_model.cache_clear()


def test_the_mask_model_is_the_checkpoint_loaded_strictly_in_eval_mode(mask_checkpoint):
    state, weights, loaded = mask_checkpoint

    model = runtime.get_mask_model()

    assert loaded == {weights: {"map_location": torch.device("cpu"), "weights_only": True}}
    assert not model.training
    got = model.state_dict()
    assert got.keys() == state.keys()
    assert all(torch.equal(got[name], state[name]) for name in state)
    assert not any(t.is_meta for t in [*model.parameters(), *model.buffers()])
    assert runtime.get_mask_model() is model  # cached


@pytest.mark.parametrize("change", ["a key missing", "an unknown key"])
def test_a_checkpoint_that_does_not_fit_the_network_is_unavailable(mask_checkpoint, change):
    state, weights, _loaded = mask_checkpoint
    if change == "a key missing":
        del state["mask_output.weight"]
    else:
        state["mask_output.extra"] = torch.zeros(1)

    with pytest.raises(runtime.ModelUnavailableError, match="best_mask_model_V5.pth"):
        runtime.get_mask_model()


def test_the_warm_up_imports_the_models_code_and_loads_no_checkpoint(cpu_runtime, monkeypatch):
    imported = []
    monkeypatch.setattr(runtime.importlib, "import_module", imported.append)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("the warm-up read a checkpoint"))
    runtime.get_mask_model.cache_clear()
    runtime.get_skeleton_model.cache_clear()

    assert runtime.warm_up() is cpu_runtime

    assert imported == ["app.inference.skeleton_inference", "app.mask_model.model"]
    assert runtime.get_mask_model.cache_info().currsize == 0
    assert runtime.get_skeleton_model.cache_info().currsize == 0
    imported.clear()
    runtime.warm_up(models=False)
    assert imported == []
