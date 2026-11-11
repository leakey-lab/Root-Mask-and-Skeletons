"""How the skeleton generator's BatchNorm layers get their statistics (``norm_stats``).

``"image"`` (the default): every image of a batch is normalised with its own per-channel
mean and biased variance over H x W, then the checkpoint's ``weight`` / ``bias``; no
running statistic is used or updated and Dropout is off. At batch size one that is what
the pix2pix ``test.py`` without ``--eval`` computed with its Dropout layers off, and an
image's output never depends on the other rows of its batch.
``"running"`` (asked for explicitly or with ``SPROUTS_SKELETON_NORM_STATS=running``):
``netG.eval()``, the checkpoint's running statistics, bit for bit.

CPU only, random weights, no checkpoint: the network is built with randomised BatchNorm
affine parameters and running statistics, saved to a temporary checkpoint and loaded
through ``runtime.skeleton_weights_path`` as the app loads the real one.
"""

from __future__ import annotations

import copy
import json
import logging

import pytest
import torch

from app import provenance
from app.inference import runtime
from app.inference import skeleton_inference as si
from tests.test_generation_inference import _images, _StubSkeletonModel

SIZE = 64  # spatial size of the test inputs (the app's is 256; the layers do not care)


def _random_generator(seed: int = 0) -> si.EnhancedResnetGenerator:
    """The app's generator with random weights and random BatchNorm affine + statistics."""
    torch.manual_seed(seed)
    net = si.EnhancedResnetGenerator(
        si.INPUT_NC,
        si.OUTPUT_NC,
        si.NGF,
        norm_layer=si.get_norm_layer(norm_type=si.NORM),
        use_dropout=si.USE_DROPOUT,
        n_blocks=9,
    )
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                n = module.num_features
                module.weight.copy_(torch.rand(n) + 0.5)
                module.bias.copy_(torch.randn(n) * 0.2)
                module.running_mean.copy_(torch.randn(n) * 0.3)
                module.running_var.copy_(torch.rand(n) + 0.5)
                module.num_batches_tracked.fill_(int(torch.randint(1, 1000, ())))
    return net


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    """A random generator saved where ``runtime.skeleton_weights_path`` points; the net."""
    net = _random_generator()
    path = tmp_path / "latest_net_G.pth"
    torch.save(net.state_dict(), path)
    monkeypatch.setattr(runtime, "skeleton_weights_path", lambda: str(path))
    monkeypatch.delenv(runtime.SKELETON_NORM_STATS_ENV, raising=False)
    return net


@pytest.fixture
def cpu_runtime(monkeypatch):
    runtime.get_runtime.cache_clear()
    runtime.get_skeleton_model.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield runtime.get_runtime()
    runtime.get_skeleton_model.cache_clear()
    runtime.get_runtime.cache_clear()


def _inputs(n: int, seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(n, 3, SIZE, SIZE) * 2 - 1  # the transform's [-1, 1] range


def _bn_state(net) -> dict[str, torch.Tensor]:
    return {
        name: tensor.clone()
        for name, tensor in net.state_dict().items()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }


def _train_mode_reference(net, x: torch.Tensor) -> torch.Tensor:
    """pix2pix ``test.py`` without ``--eval``, Dropout off: ``asrun_nodrop`` (one image)."""
    ref = copy.deepcopy(net)
    ref.train()
    for module in ref.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    with torch.no_grad():
        return ref(x)


# ---------------------------------------------------------------- "image" statistics
def test_image_mode_at_batch_one_is_the_train_mode_forward_without_dropout(checkpoint):
    model = si.SkeletonModel(gpu_ids=[], norm_stats="image")
    for k in range(3):
        x = _inputs(1, seed=10 + k)
        out = model.run_batch(x)
        ref = _train_mode_reference(checkpoint, x)
        # Exact on the CPU: both call the same batch-norm kernel on a (1, C, H, W) input.
        assert torch.equal(out, ref)


def test_the_image_stats_layer_is_exactly_per_image():
    torch.manual_seed(5)
    layer = si.ImageStatsBatchNorm2d(16)
    with torch.no_grad():
        layer.weight.copy_(torch.rand(16) + 0.5)
        layer.bias.copy_(torch.randn(16))
    x = torch.randn(4, 16, 32, 32) * 3 + 1
    padded = torch.zeros_like(x)
    padded[0] = x[2]
    with torch.inference_mode():
        alone = torch.cat([layer(x[i : i + 1]) for i in range(4)])
        assert torch.equal(layer(x), alone)  # bit for bit, whatever else is in the batch
        assert torch.equal(layer(padded)[0], alone[2])
        assert torch.equal(layer(padded)[1:], layer.bias.view(1, -1, 1, 1).expand(3, 16, 32, 32))
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        by_hand = (x - mean) / torch.sqrt(var + layer.eps) * layer.weight.view(1, -1, 1, 1)
        by_hand = by_hand + layer.bias.view(1, -1, 1, 1)
        assert torch.allclose(layer(x), by_hand, rtol=0, atol=1e-5)


def test_image_mode_never_depends_on_the_rest_of_the_batch(checkpoint):
    model = si.SkeletonModel(gpu_ids=[], norm_stats="image")
    batch = _inputs(4, seed=3)
    alone = torch.cat([model.run_batch(batch[i : i + 1]) for i in range(4)])
    together = model.run_batch(batch)
    padded = torch.zeros(4, 3, SIZE, SIZE)
    padded[0] = batch[2]  # run_batches(fixed=True) pads a short tail with zero rows
    with_zeros = model.run_batch(padded)

    # Not bit for bit: the CPU convolutions themselves (shared by both modes, oneDNN or
    # not) round differently at another batch size; "running" mode differs by ~3e-7 on
    # this network, the per-image normalisation then scales that to ~3e-6. The norm
    # layer itself is exact (test above). 1e-5 is ~1/800 of one uint8 output level.
    assert torch.allclose(together, alone, rtol=0, atol=1e-5)
    assert torch.allclose(with_zeros[0], alone[2], rtol=0, atol=1e-5)
    assert torch.isfinite(with_zeros).all()  # an all-zero row has variance 0: no NaN
    # Statistics over the whole batch (train mode at batch 4) give another result: the
    # statistics above really are per image.
    batch_stats = _train_mode_reference(checkpoint, batch)
    assert not torch.allclose(batch_stats, alone, rtol=0, atol=1e-3)


def test_image_mode_leaves_the_running_statistics_alone(checkpoint):
    model = si.SkeletonModel(gpu_ids=[], norm_stats="image")
    before = _bn_state(model.netG)
    model.run_batch(_inputs(2))
    after = _bn_state(model.netG)
    assert before.keys() == after.keys() and before
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert all(torch.equal(before[name], checkpoint.state_dict()[name].clone()) for name in before)


def test_image_mode_keeps_the_checkpoint_keys_and_load_properties(checkpoint):
    model = si.SkeletonModel(gpu_ids=[], norm_stats="image")
    loaded = model.netG.state_dict()
    saved = checkpoint.state_dict()
    assert loaded.keys() == saved.keys()
    assert all(torch.equal(loaded[key], saved[key]) for key in saved)
    assert not any(t.is_meta for t in [*model.netG.parameters(), *model.netG.buffers()])
    assert not model.netG.training
    assert model.norm_stats == "image"


# ---------------------------------------------------------------- "running" statistics
def test_running_mode_is_eval_mode_bit_for_bit(checkpoint, monkeypatch):
    reference = copy.deepcopy(checkpoint).eval()
    x = _inputs(3)
    with torch.no_grad():
        expected = reference(x)
    explicit = si.SkeletonModel(gpu_ids=[], norm_stats="running")
    assert explicit.norm_stats == "running"
    assert torch.equal(explicit.run_batch(x), expected)
    monkeypatch.setenv(runtime.SKELETON_NORM_STATS_ENV, "running")  # the earlier behaviour
    from_env = si.SkeletonModel(gpu_ids=[])
    assert from_env.norm_stats == "running"
    assert torch.equal(from_env.run_batch(x), expected)


# ---------------------------------------------------------------- the default
def test_the_default_is_image_mode():
    assert runtime.DEFAULT_SKELETON_NORM_STATS == "image"


def test_the_default_model_is_the_per_image_reference(checkpoint):
    model = si.SkeletonModel(gpu_ids=[])  # environment unset (the fixture): the default
    assert model.norm_stats == "image" == runtime.DEFAULT_SKELETON_NORM_STATS
    for k in range(2):
        x = _inputs(1, seed=20 + k)
        assert torch.equal(model.run_batch(x), _train_mode_reference(checkpoint, x))
    eval_ref = copy.deepcopy(checkpoint).eval()
    x = _inputs(1)
    with torch.no_grad():
        assert not torch.allclose(model.run_batch(x), eval_ref(x), rtol=0, atol=1e-3)


def test_the_two_modes_differ(checkpoint):
    x = _inputs(1)
    running = si.SkeletonModel(gpu_ids=[], norm_stats="running").run_batch(x)
    image = si.SkeletonModel(gpu_ids=[], norm_stats="image").run_batch(x)
    assert not torch.allclose(running, image, rtol=0, atol=1e-3)


def test_an_unknown_explicit_mode_is_refused(checkpoint):
    with pytest.raises(ValueError, match="norm"):
        si.SkeletonModel(gpu_ids=[], norm_stats="batch")


# ---------------------------------------------------------------- the environment
def test_the_environment_selects_the_mode(monkeypatch, caplog):
    env = runtime.SKELETON_NORM_STATS_ENV
    assert env == "SPROUTS_SKELETON_NORM_STATS"
    assert runtime.SKELETON_NORM_STATS_MODES == ("running", "image")
    monkeypatch.delenv(env, raising=False)
    assert runtime.skeleton_norm_stats() == "image"
    for raw, expected in (
        ("", "image"),
        ("   ", "image"),
        (" Running ", "running"),
        ("RUNNING", "running"),
        (" Image ", "image"),
        ("IMAGE", "image"),
        ("\trunning\n", "running"),
    ):
        monkeypatch.setenv(env, raw)
        with caplog.at_level(logging.WARNING, logger=runtime.__name__):
            caplog.clear()
            assert runtime.skeleton_norm_stats() == expected, raw
            assert not caplog.records, raw
    monkeypatch.setenv(env, "per-image")
    with caplog.at_level(logging.WARNING, logger=runtime.__name__):
        caplog.clear()
        assert runtime.skeleton_norm_stats() == runtime.DEFAULT_SKELETON_NORM_STATS == "image"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and env in warnings[0].getMessage()


def test_an_explicit_mode_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv(runtime.SKELETON_NORM_STATS_ENV, "running")
    assert runtime.resolve_skeleton_norm_stats(None) == "running"
    assert runtime.resolve_skeleton_norm_stats("image") == "image"
    monkeypatch.delenv(runtime.SKELETON_NORM_STATS_ENV)
    assert runtime.resolve_skeleton_norm_stats(None) == "image"
    assert runtime.resolve_skeleton_norm_stats("running") == "running"
    with pytest.raises(ValueError):
        runtime.resolve_skeleton_norm_stats("Image")  # explicit values are not normalised


def test_the_model_reads_the_environment_when_it_is_built(checkpoint, monkeypatch):
    monkeypatch.setenv(runtime.SKELETON_NORM_STATS_ENV, "running")
    model = si.SkeletonModel(gpu_ids=[])
    monkeypatch.setenv(runtime.SKELETON_NORM_STATS_ENV, "image")
    assert model.norm_stats == "running"  # fixed at load time
    x = _inputs(1)
    with torch.no_grad():
        assert torch.equal(model.run_batch(x), copy.deepcopy(checkpoint).eval()(x))


# ---------------------------------------------------------------- the model cache
def test_the_cache_never_serves_another_mode(checkpoint, cpu_runtime, monkeypatch):
    cpu = torch.device("cpu")
    env = runtime.SKELETON_NORM_STATS_ENV
    monkeypatch.setenv(env, "running")
    running = runtime.get_skeleton_model(cpu)
    assert running.norm_stats == "running"
    assert runtime.get_skeleton_model(cpu) is running  # still one cached model per mode
    monkeypatch.setenv(env, "image")
    image = runtime.get_skeleton_model(cpu)
    assert image is not running and image.norm_stats == "image"
    monkeypatch.delenv(env)
    assert runtime.get_skeleton_model(cpu) is image  # the default
    assert runtime.get_skeleton_model(cpu, norm_stats="running").norm_stats == "running"
    monkeypatch.setenv(env, "running")
    assert runtime.get_skeleton_model(cpu).norm_stats == "running"
    assert runtime.describe_runtime().skeleton_loaded is True


# ---------------------------------------------------------------- the run manifest
def _manifest_parameters(out) -> dict:
    with open(provenance.run_manifest_path(out), encoding="utf-8") as fh:
        return json.load(fh)["parameters"]


class _ModeStub(_StubSkeletonModel):
    def __init__(self, norm_stats):
        super().__init__()
        self.norm_stats = norm_stats


@pytest.mark.parametrize("mode", ["image", "running", None])
def test_the_manifest_records_the_mode_that_ran(tmp_path, cpu_runtime, mode):
    model = _StubSkeletonModel() if mode is None else _ModeStub(mode)  # None: no attribute
    src = _images(str(tmp_path / "imgs"), ["a.jpg", "b.jpg"])
    out = str(tmp_path / "imgs" / "skeletons")
    si.run_skeleton_generation(src, out, model=model, batch_size=2)
    parameters = _manifest_parameters(out)
    assert parameters["norm_stats"] == mode
    assert set(parameters) == {
        "batch_size",
        "batch_policy",
        "load_size",
        "aspect_ratio",
        "output_size_wh",
        "norm_stats",
    }


@pytest.mark.parametrize("env, expected", [(None, "image"), ("running", "running")])
def test_the_manifest_records_the_cached_models_mode(
    tmp_path, checkpoint, cpu_runtime, monkeypatch, env, expected
):
    if env is not None:
        monkeypatch.setenv(runtime.SKELETON_NORM_STATS_ENV, env)
    src = _images(str(tmp_path / "imgs"), ["a.jpg"])
    out = str(tmp_path / "imgs" / "skeletons")
    si.run_skeleton_generation(src, out, batch_size=1)
    assert _manifest_parameters(out)["norm_stats"] == expected


# ---------------------------------------------------------------- the pre-flight floor
GIB = 1024**3


@pytest.fixture
def free_gpu_memory(monkeypatch):
    """Fake driver free memory for ``check_fixed_batch_memory`` (no GPU use)."""
    state = {"free": 8 * GIB}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (state["free"], 16 * GIB))
    monkeypatch.setattr(runtime, "release_cuda_cache", lambda: None)
    monkeypatch.delenv(runtime.SKELETON_NORM_STATS_ENV, raising=False)
    return state


def test_the_image_mode_has_its_own_measured_floor():
    assert runtime.FIXED_SKELETON_IMAGE_VRAM_FLOOR_GIB == 4.7
    assert runtime.FIXED_SKELETON_VRAM_FLOOR_GIB == 3.8  # "running" unchanged
    assert "FIXED_SKELETON_IMAGE_VRAM_FLOOR_GIB" in runtime.__all__


def test_the_image_mode_stops_below_its_floor_and_says_which(free_gpu_memory):
    from app.inference.batch_engine import FIXED_MEMORY_MESSAGE, FixedBatchMemoryTooLow

    free_gpu_memory["free"] = int(4.55 * GIB)  # enough for "running", not for "image"
    with pytest.raises(FixedBatchMemoryTooLow) as caught:
        runtime.check_fixed_batch_memory("skeleton", 64, "cuda", norm_stats="image")
    assert str(caught.value) == FIXED_MEMORY_MESSAGE.format(n=64, free=4.5, needed=4.7)
    assert caught.value.needed_gib == 4.7
    assert runtime.check_fixed_batch_memory(
        "skeleton", 64, "cuda", norm_stats="running"
    ) == pytest.approx(4.55)


def test_the_floor_follows_the_environment_when_no_mode_is_given(free_gpu_memory, monkeypatch):
    from app.inference.batch_engine import FixedBatchMemoryTooLow

    free_gpu_memory["free"] = int(4.0 * GIB)
    # the default ("image") has the 4.7 GiB floor
    with pytest.raises(FixedBatchMemoryTooLow, match=r"4\.0 GiB free, 4\.7 GiB needed"):
        runtime.check_fixed_batch_memory("skeleton", 64, "cuda")
    monkeypatch.setenv(runtime.SKELETON_NORM_STATS_ENV, "running")
    assert runtime.check_fixed_batch_memory("skeleton", 64, "cuda") == pytest.approx(4.0)
    # an explicit mode wins over the environment
    with pytest.raises(FixedBatchMemoryTooLow):
        runtime.check_fixed_batch_memory("skeleton", 64, "cuda", norm_stats="image")


def test_the_image_floor_is_only_for_its_measured_size_and_model(free_gpu_memory):
    free_gpu_memory["free"] = int(1.0 * GIB)
    # no floor measured for another skeleton size: measured, never stopped (as "running")
    assert runtime.check_fixed_batch_memory(
        "skeleton", 32, "cuda", norm_stats="image"
    ) == pytest.approx(1.0)
    free_gpu_memory["free"] = int(4.0 * GIB)
    # the mask floors ignore the skeleton mode
    assert runtime.check_fixed_batch_memory("mask", 4, "cuda", norm_stats="image") == (
        pytest.approx(4.0)
    )
    assert runtime.check_fixed_batch_memory("skeleton", 64, "cpu", norm_stats="image") is None
    with pytest.raises(ValueError):
        runtime.check_fixed_batch_memory("skeleton", 64, "cuda", norm_stats="per-image")


@pytest.mark.parametrize("mode", ["image", "running", None])
def test_a_fixed_run_checks_the_floor_of_the_mode_that_runs(
    tmp_path, cpu_runtime, monkeypatch, mode
):
    calls = []

    def check(model, batch_size, device_type, *, norm_stats=None):
        calls.append((model, batch_size, device_type, norm_stats))
        return 9.0

    monkeypatch.setattr(runtime, "check_fixed_batch_memory", check)
    monkeypatch.setenv(runtime.SKELETON_NORM_STATS_ENV, "running")  # never read for the model's
    model = _StubSkeletonModel() if mode is None else _ModeStub(mode)
    src = _images(str(tmp_path / "imgs"), ["a.jpg"])
    si.run_skeleton_generation(
        src, str(tmp_path / "imgs" / "out"), model=model, batch_size=2, batch_policy="fixed"
    )
    assert calls == [("skeleton", 2, "cpu", mode)]
