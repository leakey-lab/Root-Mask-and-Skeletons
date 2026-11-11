"""Record the two model pins of the test suite from the real checkpoints.

    D:/Root-Mask-and-Skeletons/venv/Scripts/python.exe \
        tests/record_model_pins.py [--layout] [--outputs]

(from the repository root; no option: both. Needs the checkpoints in `checkpoints/`.)

- ``tests/data/model_checkpoint_layout.json``: the state-dict layout (name, shape, dtype) of
  each registered checkpoint, with its SHA-256.
- ``tests/data/model_outputs_cpu.json``: what the real models give on the CPU for the three
  ``docs/examples`` photos, through the app's own entry points with fixed batches: the mask
  run and the skeleton run in both norm-statistics modes (``running``, ``image``); per output
  file the shape, the SHA-256 of the pixels, the foreground count and the area / length that
  Measure computes from it. The machine kind (``platform.machine()``,
  ``torch.backends.cpu.get_cpu_capability()``) is recorded with it: the test compares only on
  that kind of machine.

Re-record only together with an entry under "Scientific output changes" in CHANGELOG.md (the
owner's). CUDA is hidden before torch loads, so the run is the CPU's whatever the machine has.
The photos are copied to a temporary folder: nothing is written under docs/. The helpers are
the test's own (``tests/test_checkpoints.py``), so the pin and its reader cannot drift.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch: the CPU, as the test forces it
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from app import config  # noqa: E402
from app.inference import runtime  # noqa: E402
from app.models import MODELS  # noqa: E402
from tests import test_checkpoints as tc  # noqa: E402

SCRIPT = "tests/record_model_pins.py"


def _write(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=1)
        fh.write("\n")
    print("wrote", os.path.relpath(path, REPO))


def _today() -> str:
    return datetime.date.today().isoformat()


def record_layout() -> None:
    models = {}
    for key, spec in MODELS.items():
        models[key] = {
            "file": spec.relpath,
            "sha256": spec.sha256,
            "entries": tc.checkpoint_layout(spec.path()),
        }
    _write(
        tc.LAYOUT_PIN,
        {
            "about": (
                "State-dict layout (name -> [shape, dtype]) of the registered checkpoints, "
                "read with torch.load(weights_only=True). The networks the app builds must "
                "load it strictly (tests/test_mask_model.py, tests/test_skeleton_generator.py); "
                "tests/test_checkpoints.py checks it against the files. Re-record only with a "
                "'Scientific output changes' entry in CHANGELOG.md."
            ),
            "recorded_by": SCRIPT,
            "recorded": _today(),
            "torch": torch.__version__,
            "models": models,
        },
    )


def record_outputs() -> None:
    rt = runtime.get_runtime()
    assert rt.device.type == "cpu", rt.device
    runs = {}
    with tempfile.TemporaryDirectory() as folder:
        stems = tc.copy_photos(folder)
        for run in tc.RUNS:
            out = tc.run_models(folder, run)
            entry = {
                "entry_point": "mask_inference.generate_masks"
                if run == "mask"
                else "skeleton_inference.run_skeleton_generation",
                "batch_policy": "fixed",
                "batch_size": config.MASK_BATCH_SIZE_CPU
                if run == "mask"
                else config.SKELETON_BATCH_SIZE_CPU,
            }
            if run != "mask":
                entry["norm_stats"] = run.removeprefix("skeleton_")
            entry["files"] = tc.describe_run(out, stems, run)
            runs[run] = entry
    _write(
        tc.OUTPUTS_PIN,
        {
            "about": (
                "CPU outputs of the real checkpoints on the three docs/examples photos, through "
                "the app's entry points (fixed batches). Per file: shape, SHA-256 of the pixels, "
                "foreground pixels (mask: 255; skeleton: after Measure's Otsu + thinning) and "
                "the value Measure computes (18 x 13.5 mm field of view, 2 dp). Compared exactly, "
                "only on the machine kind of 'recorded_on'. These are not the committed "
                "docs/examples files (GPU production outputs). Re-record only with a "
                "'Scientific output changes' entry in CHANGELOG.md."
            ),
            "recorded_by": SCRIPT,
            "recorded": _today(),
            "recorded_on": {
                **tc.this_machine(),
                "torch": torch.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch_threads": torch.get_num_threads(),
            },
            "also_reproduced_with": (
                "torch 2.8.0.dev20250325+cu128 on the CPU (the venv of the published results), "
                "1 / 2 / 16 torch threads, batch 1 / 4 / 8 (probe of 2026-11-10)"
            ),
            "checkpoints": {key: spec.sha256 for key, spec in MODELS.items()},
            "photos": [f"docs/examples/{stem}.jpg" for stem in stems],
            "runs": runs,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", action="store_true")
    parser.add_argument("--outputs", action="store_true")
    args = parser.parse_args()
    both = not (args.layout or args.outputs)
    if args.layout or both:
        record_layout()
    if args.outputs or both:
        record_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
