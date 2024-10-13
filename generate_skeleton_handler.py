import os
import subprocess
import sys
import re
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressBar
from PyQt6.QtCore import QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QImage, QPainter, QColor

print("Debug: Modules imported successfully")


class SkeletonGeneratorThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, input_dir, output_dir):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        print(
            f"Debug: SkeletonGeneratorThread initialized with input_dir: {input_dir}, output_dir: {output_dir}"
        )

    def run(self):
        print("Debug: SkeletonGeneratorThread run method started")
        try:
            script_path = "pix2pix_inference_script.py"
            command = [
                sys.executable,
                script_path,
                "--dataroot",
                self.input_dir,
                "--results_dir",
                self.output_dir,
            ]

            print(f"Debug: Executing command: {' '.join(command)}")

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
            print("Debug: Subprocess started")

            progress_regex = re.compile(r"Progress: (\d+)%")

            for line in iter(process.stdout.readline, ""):
                print(f"Debug: Script output: {line.strip()}")

                match = progress_regex.search(line)
                if match:
                    progress_percentage = int(match.group(1))
                    print(f"Debug: Progress: {progress_percentage}%")
                    self.progress.emit(progress_percentage)

            process.wait()
            print(f"Debug: Subprocess completed with return code: {process.returncode}")

            if process.returncode != 0:
                print(
                    f"Debug: Process returned non-zero exit status: {process.returncode}"
                )
                raise subprocess.CalledProcessError(process.returncode, command)

            model_name = "skeletonizer"
            results_dir = os.path.join(self.output_dir, model_name, "test_latest")
            html_path = os.path.join(results_dir, "index.html")

            print(f"Debug: Checking for HTML file at: {html_path}")
            if os.path.exists(html_path):
                print(
                    f"Debug: HTML file found. Emitting results directory: {results_dir}"
                )
                self.progress.emit(100)  # Ensure we reach 100% at the end
                self.finished.emit(results_dir)
            else:
                print("Debug: HTML file not found")
                self.error.emit("HTML result file not found.")
        except subprocess.CalledProcessError as e:
            print(f"Debug: CalledProcessError: {str(e)}")
            self.error.emit(f"Error running skeleton generation script: {str(e)}")
        except Exception as e:
            print(f"Debug: Unexpected error: {str(e)}")
            self.error.emit(f"Unexpected error: {str(e)}")
        print("Debug: SkeletonGeneratorThread run method completed")


class GenerateSkeletonHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        print("Debug: GenerateSkeletonHandler initialized")

    def generate_skeleton(self):
        print("Debug: generate_skeleton method called")
        input_dir = QFileDialog.getExistingDirectory(
            self.main_window, "Select Input Folder"
        )
        print(f"Debug: Input directory selected: {input_dir}")
        if not input_dir:
            print("Debug: No input directory selected, returning")
            return

        output_dir = os.path.normpath(os.path.join(input_dir, "output"))
        os.makedirs(output_dir, exist_ok=True)
        print(f"Debug: Output directory created: {output_dir}")

        self.progress_bar = QProgressBar(self.main_window)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.main_window.status_bar.addWidget(self.progress_bar)
        print("Debug: Progress bar added to status bar")

        self.thread = SkeletonGeneratorThread(input_dir, output_dir)
        self.thread.finished.connect(self.on_generation_finished)
        self.thread.error.connect(self.on_generation_error)
        self.thread.progress.connect(self.update_progress)
        self.thread.start()
        print("Debug: SkeletonGeneratorThread started")

        self.main_window.status_bar.showMessage("Generating skeletons...")
        print("Debug: Status bar message updated")

    def update_progress(self, value):
        print(f"Debug: Updating progress bar to {value}%")
        self.progress_bar.setValue(value)

    def on_generation_finished(self, results_dir):
        print(f"Debug: on_generation_finished called with results_dir: {results_dir}")
        self.main_window.status_bar.removeWidget(self.progress_bar)
        self.main_window.status_bar.showMessage("Skeleton generation completed.", 5000)
        print(f"Debug: Loading results from {results_dir}")
        self.main_window.load_results(results_dir)

    def on_generation_error(self, error_message):
        print(f"Debug: on_generation_error called with message: {error_message}")
        self.main_window.status_bar.removeWidget(self.progress_bar)
        self.main_window.status_bar.showMessage(
            "Error occurred during skeleton generation.", 5000
        )
        QMessageBox.critical(self.main_window, "Error", error_message)
        print("Debug: Error message displayed to user")


print("Debug: Script completed")
