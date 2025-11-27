import csv
import cv2
import numpy as np
import os
import re
from PyQt6.QtCore import QThread, pyqtSignal
from concurrent.futures import ThreadPoolExecutor, as_completed


# Standalone functions for parallel processing with thread pool

def load_mask(image_path):
    """Load mask image as binary (standalone for thread pool execution)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary


def calculate_root_area(mask):
    """Calculate the total root area from mask image (standalone for thread pool execution)."""
    # Count white pixels (root pixels)
    root_pixels = np.sum(mask == 255)

    # Mask image dimensions (same as original)
    image_width_px = 640
    image_height_px = 480

    # Physical dimensions of the image
    physical_width_mm = 18  # mm
    physical_height_mm = 13  # mm

    # Physical pixel sizes
    pixel_size_x = physical_width_mm / image_width_px
    pixel_size_y = physical_height_mm / image_height_px

    # Calculate area per pixel (mm²)
    area_per_pixel = pixel_size_x * pixel_size_y

    # Total root area in mm²
    total_area = root_pixels * area_per_pixel
    return total_area


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


def process_single_mask(name, path):
    """
    Process a single mask to calculate root area (standalone for thread pool execution).
    
    Args:
        name: Image name
        path: Path to mask file
        
    Returns:
        Dictionary with result information
    """
    try:
        # Calculate root area
        mask = load_mask(path)
        total_area = calculate_root_area(mask)

        # Parse image information
        info = parse_image_name(name)

        # Create result dictionary
        result = {
            "Image": info["original_name"],
            "Tube": info["tube_number"],
            "Position": info["length_position"],
            "Date": info["date"],
            "Time": info["time"],
            "Area (mm²)": round(total_area, 2),
        }

        return result

    except Exception as e:
        return {
            "Image": name,
            "Tube": None,
            "Position": None,
            "Date": None,
            "Time": None,
            "Area (mm²)": 0,
            "Error": str(e),
        }


class RootAreaCalculatorThread(QThread):
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, mask_images, output_dir):
        super().__init__()
        self.mask_images = mask_images
        self.output_dir = output_dir

    def run(self):
        results = []
        total_images = len(self.mask_images)
        
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
                executor.submit(process_single_mask, name, path): name
                for name, path in self.mask_images.items()
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
                        "Area (mm²)": 0,
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
        csv_path = os.path.join(self.output_dir, "root_areas.csv")
        self.save_to_csv(sorted_results, csv_path)
        self.finished.emit(csv_path)

    def save_to_csv(self, results, filename):
        """Save results to CSV with headers for all fields."""
        headers = ["Image", "Tube", "Position", "Date", "Time", "Area (mm²)"]

        with open(filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for result in results:
                # Only write the fields we want (exclude Error if present)
                row = {header: result.get(header, "") for header in headers}
                writer.writerow(row)
