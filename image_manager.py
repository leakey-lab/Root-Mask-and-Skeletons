import os
from PyQt6.QtWidgets import QFileDialog, QMessageBox


class ImageManager:
    def __init__(self):
        self.images = {}
        self.fake_images = {}
        self.has_fake_real_pairs = False
        self.html_path = None
        self.masks = {}  # New dictionary to store mask information

    def load_images(self, folder_path=None):
        if folder_path is None:
            folder_path = QFileDialog.getExistingDirectory(None, "Select Image Folder")
        if folder_path:
            self._clear_images()
            self._load_from_folder(folder_path)
            self._check_for_masks(folder_path)  # New method to check for masks

    def _clear_images(self):
        self.images.clear()
        self.fake_images.clear()
        self.has_fake_real_pairs = False
        self.html_path = None
        self.masks.clear()  # Clear masks information

    def _load_from_folder(self, folder_path):
        print(f"DEBUG: Loading images from folder: {folder_path}")
        is_output_dir = any(
            file.endswith("_fake.png") for file in os.listdir(folder_path)
        )
        if is_output_dir:
            image_dir = os.path.join(folder_path, "images")
            if os.path.exists(image_dir):
                folder_path = image_dir

        for file_name in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file_name)
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                self._categorize_image(file_name, full_path, is_output_dir)

        if self.has_fake_real_pairs:
            parent_dir = os.path.dirname(folder_path)
            potential_html_path = os.path.join(parent_dir, "index.html")
            if os.path.exists(potential_html_path):
                self.html_path = potential_html_path
                print(f"DEBUG: Found HTML file at {self.html_path}")
            else:
                self.html_path = None
                print("DEBUG: No HTML file found")
        else:
            self.html_path = None
            print("DEBUG: No fake-real pairs found")

    def _categorize_image(self, file_name, full_path, is_output_dir):
        if is_output_dir and file_name.endswith("_fake.png"):
            base_name = file_name.replace("_fake.png", "")
            self.fake_images[base_name] = full_path
            if base_name not in self.images:
                self.has_fake_real_pairs = True
        elif is_output_dir and file_name.endswith("_real.png"):
            base_name = file_name.replace("_real.png", "")
            self.images[base_name] = full_path
            if base_name not in self.fake_images:
                self.has_fake_real_pairs = True
        else:
            base_name = os.path.splitext(file_name)[0]
            self.images[base_name] = full_path
        print(f"DEBUG: Added file: {full_path}")

    def _check_for_masks(self, folder_path):
        mask_dir = os.path.join(folder_path, "mask")
        if os.path.exists(mask_dir):
            for file_name in os.listdir(mask_dir):
                if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    base_name = os.path.splitext(file_name)[0]
                    self.masks[base_name] = os.path.join(mask_dir, file_name)
                    print(f"DEBUG: Found mask for {base_name}")

    def get_image_names(self):
        return list(self.images.keys())

    def get_image_path(self, name):
        return self.images.get(name)

    def get_fake_image_path(self, name):
        return self.fake_images.get(name)

    def has_mask(self, name):
        return name in self.masks
