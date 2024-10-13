from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import QObject
from root_length_inference_handler import RootLengthCalculatorThread
import os


class RootLengthCalculator(QObject):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

    def calculate_root_length(self):
        if not self.main_window.file_handler.fake_images:
            QMessageBox.warning(
                self.main_window, "Warning", "No skeleton images loaded."
            )
            return

        output_dir = os.path.dirname(
            next(iter(self.main_window.file_handler.fake_images.values()))
        )
        self.calculator_thread = RootLengthCalculatorThread(
            self.main_window.file_handler.fake_images, output_dir
        )
        self.calculator_thread.finished.connect(self.on_calculation_finished)
        self.calculator_thread.progress.connect(self.update_progress)
        self.calculator_thread.start()

        self.main_window.show_status_message("Calculating root lengths...")

    def on_calculation_finished(self, csv_path):
        self.main_window.show_status_message("Root length calculation completed.", 5000)
        QMessageBox.information(
            self.main_window, "Calculation Complete", f"Results saved to {csv_path}"
        )

    def update_progress(self, value):
        self.main_window.show_status_message(f"Calculating root lengths... {value}%")
