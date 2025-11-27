from skimage.morphology import skeletonize
import csv
import cv2
import numpy as np
import os
import re
from PyQt6.QtCore import QThread, pyqtSignal
from concurrent.futures import ThreadPoolExecutor, as_completed


# Standalone functions for parallel processing with thread pool

def preprocess_image(image_path):
    """Preprocess image to skeleton (standalone for thread pool execution)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = 255 - binary
    skeleton = skeletonize(binary / 255)
    return (skeleton * 255).astype(np.uint8)


def calculate_root_length(skeleton):
    """Calculate root length from skeleton (standalone for thread pool execution)."""
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


def parse_image_name(name):
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

    except Exception:
        return {
            "original_name": name,
            "tube_number": None,
            "length_position": None,
            "date": None,
            "time": None,
        }


def process_single_image(name, path):
    """
    Process a single image to calculate root length (standalone for thread pool execution).
    
    Args:
        name: Image name
        path: Path to image file
        
    Returns:
        Dictionary with result information
    """
    try:
        # Calculate root length
        skeleton = preprocess_image(path)
        total_length = calculate_root_length(skeleton)

        # Parse image information
        info = parse_image_name(name)

        # Create result dictionary
        result = {
            "Image": info["original_name"],
            "Tube": info["tube_number"],
            "Position": info["length_position"],
            "Date": info["date"],
            "Time": info["time"],
            "Length (mm)": round(total_length, 2),
        }

        return result

    except Exception as e:
        return {
            "Image": name,
            "Tube": None,
            "Position": None,
            "Date": None,
            "Time": None,
            "Length (mm)": 0,
            "Error": str(e),
        }


class RootLengthCalculatorThread(QThread):
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, fake_images, output_dir):
        super().__init__()
        self.fake_images = fake_images
        self.output_dir = output_dir

    def run(self):
        results = []
        total_images = len(self.fake_images)
        
        if total_images == 0:
            self.finished.emit("")
            return
        
        # Determine optimal number of workers for thread pool
        # For I/O-bound tasks (image reading) and NumPy operations (GIL-releasing),
        # we can use more workers than CPU count
        cpu_count = os.cpu_count() or 4
        # Use 2x CPU count for I/O-bound operations, but cap at reasonable limit
        max_workers = min(cpu_count * 2, 32, total_images)

        # Use ThreadPoolExecutor for better performance with I/O-bound operations
        # Threads have less overhead than processes and work well for image I/O
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_name = {
                executor.submit(process_single_image, name, path): name
                for name, path in self.fake_images.items()
            }
            
            # Process completed tasks as they finish
            completed = 0
            for future in as_completed(future_to_name):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    name = future_to_name[future]
                    results.append({
                        "Image": name,
                        "Tube": None,
                        "Position": None,
                        "Date": None,
                        "Time": None,
                        "Length (mm)": 0,
                        "Error": str(e),
                    })
                
                # Update progress
                completed += 1
                progress_pct = int((completed / total_images) * 100)
                self.progress.emit(progress_pct)

        # Sort results
        sorted_results = sorted(
            results,
            key=lambda x: (
                x["Tube"] or float("inf"),
                x["Date"] or "",
                x["Time"] or "",
                x["Position"] or float("inf"),
            ),
        )

        # Save to CSV
        csv_path = os.path.join(self.output_dir, "root_lengths.csv")
        self.save_to_csv(sorted_results, csv_path)
        self.finished.emit(csv_path)

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
