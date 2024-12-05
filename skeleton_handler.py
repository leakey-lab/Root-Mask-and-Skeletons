from PyQt6.QtWidgets import QMessageBox
from generate_skeleton_handler import GenerateSkeletonHandler
from root_length_inference_handler import RootLengthCalculatorThread
import os


class SkeletonHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        self.skeleton_handler = GenerateSkeletonHandler(self.main_window)
        self.calculator_thread = None

    def generate_skeleton(self):
        print("DEBUG: Generate skeleton initiated")
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

        # Find skeletonizer output directory
        potential_paths = [
            os.path.join(
                base_folder, "output", "skeletonizer", "test_latest", "images"
            ),
            os.path.join(base_folder, "skeletonizer", "test_latest", "images"),
            os.path.join(base_folder, "test_latest", "images"),
        ]

        output_dir = None
        for path in potential_paths:
            if os.path.exists(path):
                output_dir = path
                break

        if not output_dir:
            # Try finding test_latest directory recursively
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

        # Collect ALL fake images
        fake_images = {}
        for filename in os.listdir(output_dir):
            if filename.endswith("_fake.png"):
                base_name = filename.replace("_fake.png", "")
                fake_images[base_name] = os.path.join(output_dir, filename)

        if not fake_images:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "No skeleton images found in the output directory.",
            )
            return

        print(f"Found {len(fake_images)} skeleton images in {output_dir}")

        # Start calculation thread
        self.calculator_thread = RootLengthCalculatorThread(
            fake_images, os.path.dirname(output_dir)
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
