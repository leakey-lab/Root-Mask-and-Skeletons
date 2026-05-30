import logging
import os

from PyQt6.QtWidgets import QMessageBox

from .generate_skeleton_handler import GenerateSkeletonHandler
from app.inference.root_area_inference_handler import RootAreaCalculatorThread
from app.inference.root_length_inference_handler import RootLengthCalculatorThread

logger = logging.getLogger(__name__)


class SkeletonHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        self.skeleton_handler = GenerateSkeletonHandler(self.main_window)
        # Keep strong references to calculator threads so they are not GC'd
        # before they finish (F-024: overwriting the ref before previous thread
        # finishes causes a thread leak — the new ref at least keeps the current
        # thread alive until replaced or finished).
        self.calculator_thread: RootLengthCalculatorThread | None = None
        self.area_calculator_thread: RootAreaCalculatorThread | None = None

    def generate_skeleton(self):
        self.skeleton_handler.generate_skeleton()

    def calculate_root_length(self):
        """Calculate root length for all skeleton images in the output directory."""
        # First find the base folder
        if self.main_window.image_manager.original_folder:
            base_folder = self.main_window.image_manager.original_folder
        else:
            if not self.main_window.image_manager.images:
                QMessageBox.warning(
                    self.main_window,
                    "Warning",
                    "No images loaded. Please load images first.",
                )
                return
            base_folder = os.path.dirname(
                next(iter(self.main_window.image_manager.images.values()))
            )

        # Find skeletons output directory
        potential_paths = [
            os.path.join(base_folder, "skeletons"),
            os.path.join(base_folder, "output", "skeletonizer", "test_latest", "images"),  # Legacy path support
            os.path.join(base_folder, "skeletonizer", "test_latest", "images"),  # Legacy path support
        ]

        output_dir = None
        for path in potential_paths:
            if os.path.exists(path):
                output_dir = path
                break

        if not output_dir:
            # Try finding test_latest directory recursively (legacy support)
            test_latest = self.main_window.find_test_latest_dir(base_folder)
            if test_latest:
                output_dir = os.path.join(test_latest, "images")

        if not output_dir or not os.path.exists(output_dir):
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "No skeleton images found. Please generate skeletons first.",
            )
            return

        # Collect ALL fake images; handle listing errors gracefully.
        fake_images: dict[str, str] = {}
        try:
            for filename in os.listdir(output_dir):
                if filename.endswith("_fake.png"):
                    base_name = filename.replace("_fake.png", "")
                    fake_images[base_name] = os.path.join(output_dir, filename)
        except OSError as exc:
            logger.exception("Cannot list skeleton directory %s", output_dir)
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Cannot read skeleton directory:\n{output_dir}\n\n{exc}",
            )
            return

        if not fake_images:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "No skeleton images found in the output directory.",
            )
            return

        # Start calculation thread - save CSV in the skeletons directory
        self.calculator_thread = RootLengthCalculatorThread(
            fake_images, output_dir
        )
        self.calculator_thread.finished.connect(self.on_calculation_finished)
        self.calculator_thread.progress.connect(self.update_progress)
        self.calculator_thread.start()

        self.main_window.status_bar.showMessage("Calculating root lengths...")

    def on_calculation_finished(self, csv_path):
        self.main_window.status_bar.showMessage(
            "Root length calculation completed.", 5000
        )
        QMessageBox.information(
            self.main_window, "Calculation Complete", f"Results saved to {csv_path}"
        )

    def update_progress(self, value):
        self.main_window.status_bar.showMessage(f"Calculating root lengths... {value}%")

    def calculate_root_area(self):
        """Calculate root area for all mask images in the masks directory."""
        # First find the base folder
        if self.main_window.image_manager.original_folder:
            base_folder = self.main_window.image_manager.original_folder
        else:
            if not self.main_window.image_manager.images:
                QMessageBox.warning(
                    self.main_window,
                    "Warning",
                    "No images loaded. Please load images first.",
                )
                return
            base_folder = os.path.dirname(
                next(iter(self.main_window.image_manager.images.values()))
            )

        # Find masks output directory
        potential_paths = [
            os.path.join(base_folder, "output", "mask"),
            os.path.join(base_folder, "mask"),
        ]

        masks_dir = None
        for path in potential_paths:
            if os.path.exists(path):
                masks_dir = path
                break

        if not masks_dir or not os.path.exists(masks_dir):
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "No mask images found. Please generate masks first.",
            )
            return

        # Collect ALL mask images; handle listing errors gracefully.
        mask_images: dict[str, str] = {}
        try:
            for filename in os.listdir(masks_dir):
                if filename.lower().endswith(".png"):
                    base_name = os.path.splitext(filename)[0]
                    mask_images[base_name] = os.path.join(masks_dir, filename)
        except OSError as exc:
            logger.exception("Cannot list mask directory %s", masks_dir)
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Cannot read mask directory:\n{masks_dir}\n\n{exc}",
            )
            return

        if not mask_images:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "No mask images found in the masks directory.",
            )
            return

        # Start calculation thread
        self.area_calculator_thread = RootAreaCalculatorThread(mask_images, masks_dir)
        self.area_calculator_thread.finished.connect(self.on_area_calculation_finished)
        self.area_calculator_thread.progress.connect(self.update_area_progress)
        self.area_calculator_thread.start()

        self.main_window.status_bar.showMessage("Calculating root areas...")

    def on_area_calculation_finished(self, csv_path):
        self.main_window.status_bar.showMessage(
            "Root area calculation completed.", 5000
        )
        QMessageBox.information(
            self.main_window, "Calculation Complete", f"Results saved to {csv_path}"
        )

    def update_area_progress(self, value):
        self.main_window.status_bar.showMessage(f"Calculating root areas... {value}%")
