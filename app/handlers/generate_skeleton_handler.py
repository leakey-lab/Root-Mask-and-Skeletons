import os
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal
from app.inference.skeleton_inference import run_inference

BATCH_SIZE = 64


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

            # Run inference directly
            results_dir = run_inference(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                progress_callback=progress_callback,
                batch_size=BATCH_SIZE,
            )

            self.progress.emit(100)  # Ensure we reach 100% at the end
            self.finished.emit(results_dir)
        except Exception as e:
            self.error.emit(f"Unexpected error: {str(e)}")


class GenerateSkeletonHandler:
    def __init__(self, main_window):
        self.main_window = main_window

    def generate_skeleton(self):
        # Check if images are already loaded
        if self.main_window.image_manager.images:
            # Use the directory of already loaded images
            if self.main_window.image_manager.original_folder:
                input_dir = self.main_window.image_manager.original_folder
            else:
                # Fallback: get directory from first image
                first_image_path = next(
                    iter(self.main_window.image_manager.images.values())
                )
                input_dir = os.path.dirname(first_image_path)
        else:
            # No images loaded, ask user to select directory
            input_dir = QFileDialog.getExistingDirectory(
                self.main_window, "Select Input Folder"
            )
            if not input_dir:
                return

        output_dir = os.path.normpath(os.path.join(input_dir, "skeletons"))
        os.makedirs(output_dir, exist_ok=True)

        # Use the existing loading progress bar instead of creating a new one
        self.main_window.loading_progress_bar.setValue(0)
        self.main_window.loading_progress_bar.show()
        self.main_window.loading_progress_bar.setTextVisible(True)
        self.main_window.loading_progress_bar.setFormat("Generating skeletons: %p%")

        self.thread = SkeletonGeneratorThread(input_dir, output_dir)
        self.thread.finished.connect(self.on_generation_finished)
        self.thread.error.connect(self.on_generation_error)
        self.thread.progress.connect(self.update_progress)
        self.thread.start()

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
        QMessageBox.critical(self.main_window, "Error", error_message)
