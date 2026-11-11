"""The published model checkpoints and what they compute (local only: they are not in git).

Promises: the files in ``checkpoints/`` are the registered ones (SHA-256, ``app/models.py``);
their state-dict layout is the one recorded in ``tests/data/model_checkpoint_layout.json``
(which the CI tests of ``test_mask_model.py`` / ``test_skeleton_generator.py`` load into the
networks); and the real models, run on the CPU through the app's own entry points
(``generate_masks``, ``run_skeleton_generation``, fixed batches) on a copy of the three photos
of ``docs/examples``, give exactly the pixels recorded in ``tests/data/model_outputs_cpu.json``,
with both norm-statistics modes of the skeleton model (``image``, the default, and
``running``), and the same pixels at batch size 1.

The recorded outputs are the CPU's, not the committed ``docs/examples`` files: those are GPU
outputs (the masks of the production run of 2026-05/06, which no configuration reproduces
bit for bit; the skeletons of a Generate run in the ``image`` mode at batch 64), and they
stay as they are. The CPU outputs were the same under torch 2.8 nightly and 2.15.0, with
1, 2 or 16 threads and batch 1, 4 or 8; CPU kernels differ by instruction set, so the outputs
are compared only on the machine kind they were recorded on (the test skips elsewhere).

Re-record (``tests/record_model_pins.py``) only together
with an entry under "Scientific output changes" in ``CHANGELOG.md``, which is the owner's: a
new pin means the masks or skeletons, and so the areas or lengths, changed. These tests take
5-15 s (two model loads, five runs over three photos), most of it the CPU forward.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil

import numpy as np
import pytest
import torch
from PIL import Image

from app import config
from app.inference import mask_inference, metrics, runtime
from app.inference import skeleton_inference as si
from app.models import MODELS, verify_all

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO, "docs", "examples")
LAYOUT_PIN = os.path.join(REPO, "tests", "data", "model_checkpoint_layout.json")
OUTPUTS_PIN = os.path.join(REPO, "tests", "data", "model_outputs_cpu.json")
RUNS = ("mask", "skeleton_running", "skeleton_image")
RECORDER = "tests/record_model_pins.py"

HAVE_CHECKPOINTS = os.path.isfile(config.MASK_WEIGHTS_PATH) and os.path.isfile(
    config.SKELETON_WEIGHTS_PATH
)
needs_checkpoints = pytest.mark.skipif(
    not HAVE_CHECKPOINTS,
    reason="the model checkpoints are not in checkpoints/ (they are not in git): "
    "run scripts/fetch_models.py (docs/MODELS.md)",
)


def _pin(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def this_machine() -> dict[str, str]:
    """The machine kind the CPU outputs depend on: architecture and torch's CPU capability."""
    return {
        "machine": platform.machine(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
    }


def checkpoint_layout(path: str) -> dict[str, list]:
    """``name -> [shape, dtype]`` of a checkpoint's state dict, in its own order."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {name: [list(t.shape), str(t.dtype)] for name, t in state.items()}


def copy_photos(folder: str) -> list[str]:
    """Copy the example photos (only them) into ``folder``; their stems, sorted."""
    os.makedirs(folder, exist_ok=True)
    stems = []
    for name in sorted(os.listdir(EXAMPLES)):
        if name.endswith(".jpg"):
            shutil.copyfile(os.path.join(EXAMPLES, name), os.path.join(folder, name))
            stems.append(name[:-4])
    return stems


def run_models(folder: str, run: str, *, batch_size: int | None = None) -> str:
    """One of ``RUNS`` over the photos in ``folder`` through the app's entry point, fixed
    batches, on the runtime's device; returns the output folder."""
    if run == "mask":
        out = os.path.join(folder, "mask")
        mask_inference.generate_masks(folder, out, batch_size=batch_size, batch_policy="fixed")
        return out
    mode = run.removeprefix("skeleton_")
    model = runtime.get_skeleton_model(runtime.get_runtime().device, norm_stats=mode)
    assert model.norm_stats == mode
    out = os.path.join(folder, "skeletons")
    si.run_skeleton_generation(
        folder, out, model=model, batch_size=batch_size, batch_policy="fixed"
    )
    return out


def describe_output(path: str, run: str) -> dict:
    """What the pin records of one output file: shape, pixel digest, foreground, the value
    Measure computes from it (default field of view, rounded as the CSV)."""
    with Image.open(path) as image:
        pixels = np.asarray(image)
    record = {
        "shape": list(pixels.shape),
        "pixels_sha256": hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest(),
    }
    if run == "mask":
        record["foreground"] = int((pixels == 255).sum())
        record["area_mm2"] = round(metrics.root_area_mm2(metrics.load_mask(path)), 2)
    else:
        skeleton = metrics.preprocess_image(path)
        record["foreground"] = int(np.count_nonzero(skeleton))
        record["length_mm"] = round(metrics.root_length_mm(skeleton), 2)
    return record


def describe_run(out: str, stems: list[str], run: str) -> dict[str, dict]:
    suffix = ".png" if run == "mask" else "_fake.png"
    return {stem: describe_output(os.path.join(out, stem + suffix), run) for stem in stems}


@pytest.fixture
def real_models(cpu_runtime):
    """The real checkpoints on the CPU (forced: CUDA reads as unavailable); the models are
    released afterwards (the mask model alone holds about 0.35 GB)."""
    runtime.get_mask_model.cache_clear()
    runtime.get_skeleton_model.cache_clear()
    yield
    runtime.get_mask_model.cache_clear()
    runtime.get_skeleton_model.cache_clear()


@pytest.fixture
def recorded_outputs():
    """The CPU pin, when this machine is the kind it was recorded on."""
    pin = _pin(OUTPUTS_PIN)
    here, there = this_machine(), pin["recorded_on"]
    if any(here[key] != there[key] for key in here):
        pytest.skip(
            f"the CPU outputs were recorded on {there['machine']} / {there['cpu_capability']}, "
            f"this is {here['machine']} / {here['cpu_capability']}: CPU kernels differ by "
            f"instruction set. Re-record here with {RECORDER} --outputs (only together with "
            "a 'Scientific output changes' entry in CHANGELOG.md)."
        )
    return pin


@needs_checkpoints
def test_shipped_checkpoints_verify():
    statuses = verify_all()
    assert statuses["mask"].state == "ok"
    assert statuses["skeleton"].state == "ok"


@needs_checkpoints
def test_the_recorded_layout_is_the_published_checkpoints():
    pin = _pin(LAYOUT_PIN)["models"]
    for key, spec in MODELS.items():
        assert pin[key]["sha256"] == spec.sha256, key  # recorded from the registered file
        assert checkpoint_layout(spec.path()) == pin[key]["entries"], key


@needs_checkpoints
@pytest.mark.usefixtures("real_models")
@pytest.mark.parametrize("run", RUNS)
def test_the_real_models_give_the_recorded_cpu_outputs(tmp_path, recorded_outputs, run):
    expected = recorded_outputs["runs"][run]
    assert recorded_outputs["checkpoints"] == {k: s.sha256 for k, s in MODELS.items()}
    stems = copy_photos(str(tmp_path))

    out = run_models(str(tmp_path), run)

    assert describe_run(out, stems, run) == expected["files"]


@needs_checkpoints
@pytest.mark.usefixtures("real_models")
@pytest.mark.parametrize("run", ["mask", "skeleton_running"])
def test_one_image_at_a_time_gives_the_same_cpu_outputs(tmp_path, recorded_outputs, run):
    stems = copy_photos(str(tmp_path))

    out = run_models(str(tmp_path), run, batch_size=1)

    assert describe_run(out, stems, run) == recorded_outputs["runs"][run]["files"]
