from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtCore import QThread, pyqtSignal
import os
import re


class ImageLoaderThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict, dict, dict, bool)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        images = {}
        masks = {}

        try:
            # Load regular images using os.scandir (faster than os.listdir)
            with os.scandir(self.folder_path) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith((".png", ".jpg", ".jpeg")):
                        base_name = os.path.splitext(entry.name)[0]
                        images[base_name] = entry.path

            # Load mask images if they exist
            mask_dir = os.path.join(self.folder_path, "mask")
            if os.path.exists(mask_dir):
                with os.scandir(mask_dir) as entries:
                    for entry in entries:
                        if entry.is_file() and entry.name.lower().endswith((".png", ".jpg", ".jpeg")):
                            base_name = os.path.splitext(entry.name)[0]
                            masks[base_name] = entry.path

            self.progress.emit(100)
            self.finished.emit(images, {}, masks, False)

        except Exception:
            self.progress.emit(100)
            self.finished.emit({}, {}, {}, False)


class ImageManager:
    def __init__(self, main_window=None):
        self.images = {}
        self.fake_images = {}
        self.masks = {}
        self.has_fake_real_pairs = False
        self.original_folder = None
        self.loader_thread = None
        self.main_window = main_window
        self.current_view_mode = None
        self.processed_base_path = None
        # Cache for hierarchical structure
        self._hierarchy_cache = None
        self._hierarchy_cache_valid = False

    def load_images(self, folder_path=None):
        if folder_path is None:
            folder_path = QFileDialog.getExistingDirectory(
                None, "Select Image Directory"
            )

        if folder_path:
            self.original_folder = folder_path
            self._clear_images()
            self._find_processed_base_path()

            if self.loader_thread and self.loader_thread.isRunning():
                self.loader_thread.terminate()
                self.loader_thread.wait()

            self.loader_thread = ImageLoaderThread(folder_path)
            if self.main_window:
                self.loader_thread.progress.connect(
                    self.main_window.update_loading_progress
                )
                self.loader_thread.finished.connect(self.on_loading_finished)
            self.loader_thread.start()

    def _find_processed_base_path(self):
        if not self.original_folder:
            return

        # Look for skeletons folder directly
        potential_path = os.path.join(self.original_folder, "skeletons")
        if os.path.exists(potential_path):
            self.processed_base_path = potential_path

    def get_fake_image_path(self, name):
        if name in self.fake_images:
            return self.fake_images[name]

        if (
            self.current_view_mode in ["Overlay", "Side by Side"]
            and self.processed_base_path
        ):
            fake_path = os.path.join(self.processed_base_path, f"{name}_fake.png")
            if os.path.exists(fake_path):
                self.fake_images[name] = fake_path
                self.has_fake_real_pairs = True
                return fake_path

        return None

    def reload_for_view_mode(self, view_mode):
        self.current_view_mode = view_mode
        if view_mode == "Single Image":
            self.fake_images.clear()
            self.has_fake_real_pairs = False

        if self.main_window:
            self.main_window.populate_file_list()
            self.main_window.display_controller.update_display()

    def on_loading_finished(self, images, fake_images, masks, has_fake_real_pairs):
        self.images = images
        self.masks = masks
        # Invalidate cache when images change
        self._hierarchy_cache = None
        self._hierarchy_cache_valid = False
        if self.main_window:
            self.main_window.on_loading_finished(
                images,
                self.fake_images,
                masks,
                self.has_fake_real_pairs,
            )

    def _clear_images(self):
        self.images.clear()
        self.fake_images.clear()
        self.has_fake_real_pairs = False
        self.masks.clear()
        self.processed_base_path = None
        # Invalidate cache
        self._hierarchy_cache = None
        self._hierarchy_cache_valid = False

    def get_image_names(self):
        return list(self.images.keys())

    def get_image_path(self, name):
        return self.images.get(name)

    def has_mask(self, name):
        return name in self.masks

    def parse_image_name(self, name):
        """
        Parse image name to extract field, tube number, length position (depth), date, and time.
        Expected format: Field_T{XXX}_L{XXX}_YYYY.MM.DD_HHMMSS_anything
        Returns a dictionary with the parsed information.
        """
        try:
            # Initialize default values
            info = {
                "original_name": name,
                "field": None,
                "tube_number": None,
                "length_position": None,
                "date": None,
                "time": None,
            }

            # Extract field name (everything before first underscore, but remove camera suffixes)
            parts = name.split("_", 1)
            if parts:
                field_name = parts[0]
                # Remove camera suffixes like cam1, CAM1, cam2, etc. from the end of field name
                # This handles cases like xyzcam1, xyzCAM1, etc.
                field_name = re.sub(r'cam\d+$', '', field_name, flags=re.IGNORECASE)
                info["field"] = field_name

            # Extract tube number (T followed by numbers)
            tube_match = re.search(r"T(\d+)", name)
            if tube_match:
                info["tube_number"] = int(tube_match.group(1))

            # Extract length position (L followed by numbers) - this is the depth
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
                "field": name,  # Fallback to using full name as field
                "tube_number": None,
                "length_position": None,
                "date": None,
                "time": None,
            }

    def get_hierarchical_structure(self):
        """
        Organize images into a hierarchical structure:
        Field → Tube → Date → Images (with L prefix)
        
        Cached for performance - only recalculated when images change.

        Returns a nested dictionary structure.
        """
        # Return cached result if valid
        if self._hierarchy_cache_valid and self._hierarchy_cache is not None:
            return self._hierarchy_cache
        
        hierarchy = {}

        for image_name in self.images.keys():
            info = self.parse_image_name(image_name)

            # Use parsed values or fallbacks
            field = info["field"] or "Unknown"
            tube = (
                f"T{info['tube_number']}"
                if info["tube_number"] is not None
                else "No Tube"
            )
            date = info["date"] or "No Date"

            # Build hierarchy (no separate depth level)
            if field not in hierarchy:
                hierarchy[field] = {}
            if tube not in hierarchy[field]:
                hierarchy[field][tube] = {}
            if date not in hierarchy[field][tube]:
                hierarchy[field][tube][date] = []

            # Store image with its depth info for sorting
            depth = (
                info["length_position"]
                if info["length_position"] is not None
                else float("inf")
            )
            hierarchy[field][tube][date].append((depth, image_name))

        # Sort images by depth within each date
        for field in hierarchy:
            for tube in hierarchy[field]:
                for date in hierarchy[field][tube]:
                    hierarchy[field][tube][date].sort(key=lambda x: x[0])

        # Cache the result
        self._hierarchy_cache = hierarchy
        self._hierarchy_cache_valid = True

        return hierarchy
