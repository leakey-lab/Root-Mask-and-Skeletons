from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal
import os


from PyQt6.QtCore import QThread, pyqtSignal


class ImageLoaderThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict, dict, dict, str, bool)

    def __init__(self, folder_path, load_skeletons=False):
        super().__init__()
        self.folder_path = folder_path
        self.load_skeletons = load_skeletons
        print(
            f"Debug: ImageLoaderThread initialized with load_skeletons={load_skeletons}"
        )

    def run(self):
        """Load images based on the load_skeletons flag"""
        images = {}
        fake_images = {}
        masks = {}
        html_path = None
        has_fake_real_pairs = False

        try:
            # Always load regular images first
            print("Debug: Loading regular images")
            self._load_from_folder(self.folder_path, images, fake_images, "single")

            # Load mask images if they exist
            mask_dir = os.path.join(self.folder_path, "mask")
            if os.path.exists(mask_dir):
                print("Debug: Loading mask images")
                for file_name in os.listdir(mask_dir):
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        base_name = os.path.splitext(file_name)[0]
                        masks[base_name] = os.path.join(mask_dir, file_name)
                        print(f"DEBUG: Found mask for {base_name}")

            # Only load skeleton/processed images if specifically requested
            if self.load_skeletons:
                output_path = os.path.join(
                    self.folder_path, "output", "skeletonizer", "test_latest", "images"
                )
                if os.path.exists(output_path):
                    print(f"DEBUG: Loading skeleton images from {output_path}")
                    self._load_from_folder(
                        output_path, images, fake_images, "processed"
                    )
                    has_fake_real_pairs = (
                        True  # Set to True if processed images are loaded
                    )

                    # Check for HTML file
                    test_latest_dir = os.path.dirname(output_path)
                    potential_html_path = os.path.join(test_latest_dir, "index.html")
                    if os.path.exists(potential_html_path):
                        html_path = potential_html_path
                        print(f"DEBUG: Found HTML file at {html_path}")
                else:
                    print(f"DEBUG: No skeleton images found in {output_path}")

            print("Debug: Image loading completed successfully")
            self.progress.emit(100)  # Ensure we show 100% completion
            self.finished.emit(
                images, fake_images, masks, html_path, has_fake_real_pairs
            )

        except Exception as e:
            print(f"Error loading images: {str(e)}")
            self.progress.emit(100)  # Ensure progress bar completes even on error
            self.finished.emit({}, {}, {}, None, False)

    def _load_from_folder(self, folder_path, images, fake_images, mode):
        """Load images from a folder with progress updates"""
        print(f"DEBUG: Loading images from folder: {folder_path}, mode: {mode}")
        is_processed = mode == "processed"

        # Count total files for progress
        total_files = len(
            [
                f
                for f in os.listdir(folder_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        processed_files = 0

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(folder_path, file_name)

                if is_processed:
                    if file_name.endswith("_fake.png"):
                        base_name = file_name.replace("_fake.png", "")
                        fake_images[base_name] = full_path
                        print(f"DEBUG: Added fake image: {full_path}")
                    elif file_name.endswith("_real.png"):
                        base_name = file_name.replace("_real.png", "")
                        images[base_name] = full_path
                        print(f"DEBUG: Added real image: {full_path}")
                else:
                    base_name = os.path.splitext(file_name)[0]
                    images[base_name] = full_path
                    print(f"DEBUG: Added regular image: {full_path}")

                processed_files += 1
                progress = int((processed_files / total_files) * 100)
                self.progress.emit(progress)


class ImageManager:
    def __init__(self, main_window=None):
        self.images = {}
        self.fake_images = {}
        self.has_fake_real_pairs = False
        self.html_path = None
        self.masks = {}
        self.original_folder = None
        self.loader_thread = None
        self.main_window = main_window
        self.current_view_mode = None
        self.processed_images_loaded = False

    def load_images(self, folder_path=None):
        """Load initial images without processing fake/real pairs"""
        if folder_path is None:
            folder_path = QFileDialog.getExistingDirectory(
                None, "Select Image Directory"
            )

        if folder_path:
            self.original_folder = folder_path
            self._clear_images()
            self.processed_images_loaded = False  # Reset the processed images flag

            # Cancel any existing loading thread
            if self.loader_thread is not None and self.loader_thread.isRunning():
                self.loader_thread.terminate()
                self.loader_thread.wait()

            # Start new loading thread with load_skeletons=False for initial load
            self.loader_thread = ImageLoaderThread(folder_path, load_skeletons=False)

            # Connect signals to main window methods
            if self.main_window:
                self.loader_thread.progress.connect(
                    self.main_window.update_loading_progress
                )
                self.loader_thread.finished.connect(self.on_loading_finished)

            self.loader_thread.start()

    def reload_for_view_mode(self, view_mode):
        """
        Reload and reorganize images based on the selected view mode
        Only loads processed images if they haven't been loaded before
        """
        print(f"DEBUG: Reloading images for view mode: {view_mode}")
        self.current_view_mode = view_mode

        if self.original_folder is None:
            print("DEBUG: No folder loaded yet")
            return

        needs_processed_images = view_mode in ["Basic View", "Overlay", "Side by Side"]

        if needs_processed_images and not self.processed_images_loaded:
            # Only load processed images if they haven't been loaded before
            if self.loader_thread is not None and self.loader_thread.isRunning():
                self.loader_thread.terminate()
                self.loader_thread.wait()

            # Start new loading thread for processed images
            self.loader_thread = ImageLoaderThread(
                self.original_folder, load_skeletons=True
            )

            # Connect signals to main window methods
            if self.main_window:
                self.loader_thread.progress.connect(
                    self.main_window.update_loading_progress
                )
                self.loader_thread.finished.connect(self.on_loading_finished)

            self.loader_thread.start()
            self.processed_images_loaded = True
        else:
            # Just update the display without reloading
            if self.main_window:
                self.main_window.populate_file_list()
                self.main_window.display_controller.update_display()

    def on_loading_finished(
        self, images, fake_images, masks, html_path, has_fake_real_pairs
    ):
        """Handle completion of image loading"""
        self.images = images
        self.fake_images = fake_images
        self.masks = masks
        self.html_path = html_path
        self.has_fake_real_pairs = has_fake_real_pairs

        # Update UI through main window
        if self.main_window:
            self.main_window.on_loading_finished(
                images, fake_images, masks, html_path, has_fake_real_pairs
            )

    def _clear_images(self):
        """Clear all image data and reset flags"""
        self.images.clear()
        self.fake_images.clear()
        self.has_fake_real_pairs = False
        self.html_path = None
        self.masks.clear()
        self.processed_images_loaded = False

    def get_image_names(self):
        return list(self.images.keys())

    def get_image_path(self, name):
        return self.images.get(name)

    def get_fake_image_path(self, name):
        return self.fake_images.get(name)

    def has_mask(self, name):
        return name in self.masks
