import logging
import os

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from app.config import SKELETON_BATCH_SIZE
from app.inference.skeleton_inference import run_inference

logger = logging.getLogger(__name__)


class SkeletonGeneratorThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, input_dir, output_dir):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.total_images = 0
        self.processed_images = 0

    def run(self):
        try:
            # Progress callback function
            def progress_callback(current, total):
                if total > 0:
                    progress_percentage = int((current / total) * 100)
                    self.progress.emit(progress_percentage)

            # Run inference using the batch size from config (F-005 / config adoption).
            results_dir = run_inference(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                progress_callback=progress_callback,
                batch_size=SKELETON_BATCH_SIZE,
            )

            self.progress.emit(100)  # Ensure we reach 100% at the end
            self.finished.emit(results_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("Skeleton generation failed for input_dir=%s", self.input_dir)
            self.error.emit(f"Unexpected error: {exc}")


class GenerateSkeletonHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        # Keep a strong reference to the thread so it cannot be GC'd before
        # completion (F-025: thread stored as self.thread may be collected early).
        self._thread: SkeletonGeneratorThread | None = None

    def generate_skeleton(self):
        # Check if images are already loaded
        if self.main_window.image_manager.images:
            # Use the directory of already loaded images
            if self.main_window.image_manager.original_folder:
                input_dir = self.main_window.image_manager.original_folder
            else:
                # Fallback: get directory from first image
                try:
                    first_image_path = next(
                        iter(self.main_window.image_manager.images.values())
                    )
                except StopIteration:
                    QMessageBox.warning(
                        self.main_window,
                        "Warning",
                        "No images loaded. Please load images first.",
                    )
                    return
                input_dir = os.path.dirname(first_image_path)
        else:
            # No images loaded, ask user to select directory
            input_dir = QFileDialog.getExistingDirectory(
                self.main_window, "Select Input Folder"
            )
            if not input_dir:
                return

        output_dir = os.path.normpath(os.path.join(input_dir, "skeletons"))
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as exc:
            logger.exception("Could not create skeleton output directory %s", output_dir)
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Could not create output directory:\n{output_dir}\n\n{exc}",
            )
            return

        # Use the existing loading progress bar instead of creating a new one
        self.main_window.loading_progress_bar.setValue(0)
        self.main_window.loading_progress_bar.show()
        self.main_window.loading_progress_bar.setTextVisible(True)
        self.main_window.loading_progress_bar.setFormat("Generating skeletons: %p%")

        # Store thread on the instance so it is not GC'd before finished (F-025).
        self._thread = SkeletonGeneratorThread(input_dir, output_dir)
        self._thread.finished.connect(self.on_generation_finished)
        self._thread.error.connect(self.on_generation_error)
        self._thread.progress.connect(self.update_progress)
        self._thread.start()

        self.main_window.status_bar.showMessage("Generating skeletons...")

    def update_progress(self, value):
        self.main_window.loading_progress_bar.setValue(value)
        # Update status message with progress
        self.main_window.status_bar.showMessage(f"Generating skeletons... {value}%")

    def on_generation_finished(self, results_dir):
        self.main_window.loading_progress_bar.hide()
        self.main_window.status_bar.showMessage("Skeleton generation completed.", 5000)
        self.main_window.load_results(results_dir)

    def on_generation_error(self, error_message):
        self.main_window.loading_progress_bar.hide()
        self.main_window.status_bar.showMessage(
            "Error occurred during skeleton generation.", 5000
        )
        logger.error("Skeleton generation error: %s", error_message)
        QMessageBox.critical(self.main_window, "Error", error_message)
