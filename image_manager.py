from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal
import os


class ImageLoaderThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict, dict, dict, str, bool)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        images = {}
        masks = {}

        try:
            # Load regular images
            for file_name in os.listdir(self.folder_path):
                if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    base_name = os.path.splitext(file_name)[0]
                    images[base_name] = os.path.join(self.folder_path, file_name)

            # Load mask images if they exist
            mask_dir = os.path.join(self.folder_path, "mask")
            if os.path.exists(mask_dir):
                for file_name in os.listdir(mask_dir):
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        base_name = os.path.splitext(file_name)[0]
                        masks[base_name] = os.path.join(mask_dir, file_name)

            self.progress.emit(100)
            self.finished.emit(images, {}, masks, None, False)

        except Exception as e:
            print(f"Error loading images: {str(e)}")
            self.progress.emit(100)
            self.finished.emit({}, {}, {}, None, False)


class ImageManager:
    def __init__(self, main_window=None):
        self.images = {}
        self.fake_images = {}
        self.masks = {}
        self.has_fake_real_pairs = False
        self.html_path = None
        self.original_folder = None
        self.loader_thread = None
        self.main_window = main_window
        self.current_view_mode = None
        self.processed_base_path = None

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

        potential_path = os.path.join(
            self.original_folder, "output", "skeletonizer", "test_latest", "images"
        )
        if os.path.exists(potential_path):
            self.processed_base_path = potential_path
            test_latest_dir = os.path.dirname(potential_path)
            potential_html = os.path.join(test_latest_dir, "index.html")
            if os.path.exists(potential_html):
                self.html_path = potential_html

    def get_fake_image_path(self, name):
        if name in self.fake_images:
            return self.fake_images[name]

        if (
            self.current_view_mode in ["Basic View", "Overlay", "Side by Side"]
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

    def on_loading_finished(
        self, images, fake_images, masks, html_path, has_fake_real_pairs
    ):
        self.images = images
        self.masks = masks
        if self.main_window:
            self.main_window.on_loading_finished(
                images,
                self.fake_images,
                masks,
                self.html_path,
                self.has_fake_real_pairs,
            )

    def _clear_images(self):
        self.images.clear()
        self.fake_images.clear()
        self.has_fake_real_pairs = False
        self.html_path = None
        self.masks.clear()
        self.processed_base_path = None

    def get_image_names(self):
        return list(self.images.keys())

    def get_image_path(self, name):
        return self.images.get(name)

    def has_mask(self, name):
        return name in self.masks
