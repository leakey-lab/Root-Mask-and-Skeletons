from skimage.morphology import skeletonize
import csv
import cv2
import numpy as np
import os
import re
from datetime import datetime
from PyQt6.QtCore import QThread, pyqtSignal


class RootLengthCalculatorThread(QThread):
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, fake_images, output_dir):
        super().__init__()
        self.fake_images = fake_images
        self.output_dir = output_dir

    def parse_image_name(self, name):
        """
        Parse image name to extract tube number, length position, date, and time.
        Expected format: anything_T{XXX}_L{XXX}_YYYY.MM.DD_HHMMSS_anything
        Returns a dictionary with the parsed information.
        """
        try:
            # Initialize default values
            info = {
                "original_name": name,
                "tube_number": None,
                "length_position": None,
                "date": None,
                "time": None,
            }

            # Extract tube number (T followed by numbers)
            tube_match = re.search(r"T(\d+)", name)
            if tube_match:
                info["tube_number"] = int(tube_match.group(1))

            # Extract length position (L followed by numbers)
            length_match = re.search(r"L(\d+)", name)
            if length_match:
                info["length_position"] = int(length_match.group(1))

            # Extract date (YYYY.MM.DD format)
            date_match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", name)
            if date_match:
                year, month, day = date_match.groups()
                info["date"] = f"{year}.{month}.{day}"

            # Extract time (HHMMSS format)
            time_match = re.search(r"_(\d{6})(?:_|$)", name)
            if time_match:
                time_str = time_match.group(1)
                formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                info["time"] = formatted_time

            return info

        except Exception as e:
            print(f"Error parsing image name '{name}': {str(e)}")
            return {
                "original_name": name,
                "tube_number": None,
                "length_position": None,
                "date": None,
                "time": None,
            }

    def run(self):
        results = []
        total_images = len(self.fake_images)

        for i, (name, path) in enumerate(self.fake_images.items()):
            try:
                # Calculate root length
                skeleton = self.preprocess_image(path)
                total_length = self.calculate_root_length(skeleton)

                # Parse image information
                info = self.parse_image_name(name)

                # Create result dictionary
                result = {
                    "Image": info["original_name"],
                    "Tube": info["tube_number"],
                    "Position": info["length_position"],
                    "Date": info["date"],
                    "Time": info["time"],
                    "Length (mm)": round(total_length, 2),
                }

                results.append(result)

                # Update progress
                self.progress.emit(int((i + 1) / total_images * 100))

            except Exception as e:
                print(f"Error processing image '{name}': {str(e)}")
                # Add error entry
                results.append(
                    {
                        "Image": name,
                        "Tube": None,
                        "Position": None,
                        "Date": None,
                        "Time": None,
                        "Length (mm)": 0,
                        "Error": str(e),
                    }
                )

        # Sort results by Tube, Date, Time, and Position
        sorted_results = sorted(
            results,
            key=lambda x: (
                x["Tube"] or float("inf"),  # Handle None values in sorting
                x["Date"] or "",
                x["Time"] or "",
                x["Position"] or float("inf"),
            ),
        )

        # Save to CSV
        csv_path = os.path.join(os.path.dirname(self.output_dir), "root_lengths.csv")
        self.save_to_csv(sorted_results, csv_path)
        self.finished.emit(csv_path)

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary) > 127:
            binary = 255 - binary
        skeleton = skeletonize(binary / 255)
        return (skeleton * 255).astype(np.uint8)

    def calculate_root_length(self, skeleton):
        pixel_length = np.sum(skeleton) / 255  # Number of skeleton pixels

        # Original and processed image dimensions
        original_width_px = 640
        original_height_px = 480
        processed_width_px = 341  # New processed width
        processed_height_px = 256  # New processed height

        # Physical dimensions of the original image
        original_width_mm = 18  # mm
        original_height_mm = 13  # mm

        # Scaling factors
        scaling_factor_x = processed_width_px / original_width_px
        scaling_factor_y = processed_height_px / original_height_px
        scaling_factor = (scaling_factor_x + scaling_factor_y) / 2

        inverse_length_scaling_factor = 1 / scaling_factor

        # Physical pixel sizes in the original image
        pixel_size_x = original_width_mm / original_width_px
        pixel_size_y = original_height_mm / original_height_px
        average_pixel_size = (pixel_size_x + pixel_size_y) / 2

        # Adjusted pixel length to reflect original dimensions
        adjusted_pixel_length = pixel_length * inverse_length_scaling_factor

        # Total root length in mm
        total_length = adjusted_pixel_length * average_pixel_size
        return total_length

    def save_to_csv(self, results, filename):
        """Save results to CSV with headers for all fields."""
        headers = ["Image", "Tube", "Position", "Date", "Time", "Length (mm)"]

        with open(filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for result in results:
                # Only write the fields we want (exclude Error if present)
                row = {header: result.get(header, "") for header in headers}
                writer.writerow(row)
