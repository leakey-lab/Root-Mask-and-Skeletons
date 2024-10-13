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
        if not self.main_window.image_manager.fake_images:
            QMessageBox.warning(
                self.main_window, "Warning", "No skeleton images loaded."
            )
            return

        output_dir = os.path.dirname(
            next(iter(self.main_window.image_manager.fake_images.values()))
        )
        self.calculator_thread = RootLengthCalculatorThread(
            self.main_window.image_manager.fake_images, output_dir
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
