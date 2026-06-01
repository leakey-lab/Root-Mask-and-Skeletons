"""Root-area calculation thread (thin wrapper over app.inference.metrics)."""
import logging
import os
import time

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

from app.config import AREA_CSV_HEADERS, ROOT_AREAS_CSV
from app.inference.metrics import (
    calculate_root_area,
    load_mask,
    parse_image_name,
    root_area_mm2,
    run_metric_pool,
    write_metric_csv,
)

__all__ = [
    "calculate_root_area",
    "load_mask",
    "parse_image_name",
    "process_single_mask",
    "RootAreaCalculatorThread",
]


def process_single_mask(name, path):
    """Compute the area row for a single mask image."""
    try:
        total_area = root_area_mm2(load_mask(path))
        info = parse_image_name(name)
        return {
            "Image": info["original_name"],
            "Tube": info["tube_number"],
            "Position": info["length_position"],
            "Date": info["date"],
            "Time": info["time"],
            "Area (mm²)": round(total_area, 2),
        }
    except Exception as e:
        logger.error("Area metric failed for %r: %s", name, e)
        return {
            "Image": name, "Tube": None, "Position": None, "Date": None,
            "Time": None, "Area (mm²)": 0, "Error": str(e),
        }


class RootAreaCalculatorThread(QThread):
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, mask_images, output_dir):
        super().__init__()
        self.mask_images = mask_images
        self.output_dir = output_dir

    def run(self):
        if not self.mask_images:
            logger.debug("RootAreaCalculatorThread: no images, skipping")
            self.finished.emit("")
            return
        logger.info(
            "RootAreaCalculatorThread: processing %d images → %s",
            len(self.mask_images), self.output_dir,
        )
        t0 = time.monotonic()
        results = run_metric_pool(self.mask_images, process_single_mask, self.progress.emit)
        csv_path = os.path.join(self.output_dir, ROOT_AREAS_CSV)
        write_metric_csv(results, csv_path, AREA_CSV_HEADERS)
        logger.info(
            "RootAreaCalculatorThread: done in %.1fs, wrote %s",
            time.monotonic() - t0, csv_path,
        )
        self.finished.emit(csv_path)
