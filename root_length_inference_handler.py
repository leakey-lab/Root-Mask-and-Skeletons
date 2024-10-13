from skimage.morphology import skeletonize
import csv
import cv2
import numpy as np
import os
from PyQt6.QtCore import QThread, pyqtSignal
from math import sqrt


class RootLengthCalculatorThread(QThread):
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, fake_images, output_dir):
        super().__init__()
        self.fake_images = fake_images
        self.output_dir = output_dir

    def run(self):
        results = {}
        total_images = len(self.fake_images)

        for i, (name, path) in enumerate(self.fake_images.items()):
            skeleton = self.preprocess_image(path)
            total_length = self.calculate_root_length(skeleton)
            results[name] = total_length
            self.progress.emit(int((i + 1) / total_images * 100))

        csv_path = os.path.join(self.output_dir, "root_lengths.csv")
        self.save_to_csv(results, csv_path)
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

        # Scaling factors (while maintaining aspect ratio)
        scaling_factor_x = processed_width_px / original_width_px  # 341 / 640
        scaling_factor_y = processed_height_px / original_height_px  # 256 / 480

        # Since aspect ratio is maintained, the scaling factors should be similar
        # You can take an average scaling factor to simplify calculations
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
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Image", "Length (mm)"])
            for name, length in results.items():
                writer.writerow([name, f"{length:.2f}"])
