"""Root-length calculation thread (thin wrapper over app.inference.metrics)."""
import os

from PyQt6.QtCore import QThread, pyqtSignal

from app.config import LENGTH_CSV_HEADERS, ROOT_LENGTHS_CSV
from app.inference.metrics import (
    calculate_root_length,
    parse_image_name,
    preprocess_image,
    root_length_mm,
    run_metric_pool,
    write_metric_csv,
)

__all__ = [
    "calculate_root_length",
    "parse_image_name",
    "preprocess_image",
    "process_single_image",
    "RootLengthCalculatorThread",
]


def process_single_image(name, path):
    """Compute the length row for a single skeleton image."""
    try:
        total_length = root_length_mm(preprocess_image(path))
        info = parse_image_name(name)
        return {
            "Image": info["original_name"],
            "Tube": info["tube_number"],
            "Position": info["length_position"],
            "Date": info["date"],
            "Time": info["time"],
            "Length (mm)": round(total_length, 2),
        }
    except Exception as e:
        return {
            "Image": name, "Tube": None, "Position": None, "Date": None,
            "Time": None, "Length (mm)": 0, "Error": str(e),
        }


class RootLengthCalculatorThread(QThread):
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, fake_images, output_dir):
        super().__init__()
        self.fake_images = fake_images
        self.output_dir = output_dir

    def run(self):
        if not self.fake_images:
            self.finished.emit("")
            return
        results = run_metric_pool(self.fake_images, process_single_image, self.progress.emit)
        csv_path = os.path.join(self.output_dir, ROOT_LENGTHS_CSV)
        write_metric_csv(results, csv_path, LENGTH_CSV_HEADERS)
        self.finished.emit(csv_path)
