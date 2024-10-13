import sys
import cv2
import numpy as np
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QSplitter,
    QStatusBar,
    QMessageBox,
    QFileDialog,
    QComboBox,
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
from generate_skeleton_handler import GenerateSkeletonHandler
from root_length_inference_handler import RootLengthCalculatorThread


class RootSkeletonViewerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        print("DEBUG: Initializing RootSkeletonViewerUI")
        self.setWindowTitle("Root Skeleton Viewer")
        self.setGeometry(100, 100, 1200, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left section
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.generate_button = QPushButton("Generate Skeleton")
        self.generate_button.clicked.connect(self.generate_skeleton)
        left_layout.addWidget(self.generate_button)

        self.load_images_button = QPushButton("Load Images")
        self.load_images_button.clicked.connect(self.load_images_button_clicked)
        left_layout.addWidget(self.load_images_button)

        self.calculate_length_button = QPushButton("Calculate Root Length")
        self.calculate_length_button.clicked.connect(self.calculate_root_length)
        left_layout.addWidget(self.calculate_length_button)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(
            ["Single Image", "Overlay", "Side by Side", "Basic View"]
        )
        self.view_mode_combo.currentIndexChanged.connect(self.update_image_display)
        left_layout.addWidget(self.view_mode_combo)

        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.display_selected_image)
        left_layout.addWidget(QLabel("Images:"))
        left_layout.addWidget(self.file_list)

        splitter.addWidget(left_widget)

        # Right section: Image display
        right_widget = QWidget()
        self.right_layout = QVBoxLayout(right_widget)
        self.image_label = QLabel("Select an image to display")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_layout.addWidget(self.image_label)

        # Add QWebEngineView for Basic View
        self.web_view = QWebEngineView()
        self.right_layout.addWidget(self.web_view)
        self.web_view.hide()  # Initially hide the web view

        splitter.addWidget(right_widget)

        # Set the initial sizes of the splitter
        splitter.setSizes([400, 800])

        # Add status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.current_image_path = None
        self.current_fake_image_path = None
        self.images = {}
        self.fake_images = {}
        self.html_path = None
        self.skeleton_handler = GenerateSkeletonHandler(self)
        print("DEBUG: RootSkeletonViewerUI initialized")

    def load_images_button_clicked(self):
        if self.images or self.fake_images:
            reply = QMessageBox.question(
                self,
                "Confirm Load",
                "Loading new images will clear the current ones. Are you sure you want to proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.load_images_from_folder()
        else:
            self.load_images_from_folder()

    def load_results(self, results_dir):
        # Append '/images' to the provided directory
        images_dir = f"{results_dir}/images"
        print(f"DEBUG: Loading results from {images_dir}")

        # Load images from the updated folder path
        self.load_images_from_folder(images_dir)

        # Store the path to the index.html file
        self.html_path = os.path.join(results_dir, "index.html")
        print(f"DEBUG: HTML path set to {self.html_path}")

    def load_images_from_folder(self, folder_path=None):
        if folder_path is None:
            folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder_path:
            print(f"DEBUG: Loading images from folder: {folder_path}")
            self.file_list.clear()
            self.images.clear()
            self.fake_images.clear()
            self.has_fake_real_pairs = False

            # Check if this is an output directory with real and fake images
            is_output_dir = any(
                file.endswith("_fake.png") for file in os.listdir(folder_path)
            )

            if is_output_dir:
                image_dir = os.path.join(folder_path, "images")
                if os.path.exists(image_dir):
                    folder_path = image_dir

            for file_name in os.listdir(folder_path):
                full_path = os.path.normpath(os.path.join(folder_path, file_name))
                if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                    if is_output_dir and file_name.endswith("_fake.png"):
                        base_name = file_name.replace("_fake.png", "")
                        self.fake_images[base_name] = full_path
                        if base_name not in self.images:
                            self.file_list.addItem(base_name)
                        self.has_fake_real_pairs = True
                    elif is_output_dir and file_name.endswith("_real.png"):
                        base_name = file_name.replace("_real.png", "")
                        self.images[base_name] = full_path
                        if base_name not in self.fake_images:
                            self.file_list.addItem(base_name)
                        self.has_fake_real_pairs = True
                    else:
                        base_name = os.path.splitext(file_name)[0]
                        self.images[base_name] = full_path
                        self.file_list.addItem(base_name)
                    print(f"DEBUG: Added file to list: {full_path}")

            # Look for index.html file only if we have fake-real pairs
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
                print("DEBUG: No fake-real pairs found, not looking for HTML file")

            print(f"DEBUG: Loaded {self.file_list.count()} images")
            if self.file_list.count() == 0:
                QMessageBox.information(
                    self, "Information", "No images found in the selected folder."
                )
            else:
                self.status_bar.showMessage(
                    f"Loaded {self.file_list.count()} images from folder", 5000
                )

    def display_selected_image(self, item):
        base_name = item.text()
        self.current_image_path = self.images.get(base_name)
        self.current_fake_image_path = self.fake_images.get(base_name)
        self.update_image_display()

    def display_single_image(self):
        pixmap = QPixmap(self.current_image_path)
        scaled_pixmap = pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
        print(f"DEBUG: Single image displayed. Size: {scaled_pixmap.size()}")

    def display_overlay_image(self):
        if not self.current_image_path or not self.current_fake_image_path:
            print("DEBUG: Missing real or fake image path")
            return

        real_pixmap = QPixmap(self.current_image_path)
        fake_image_path = self.current_fake_image_path

        if real_pixmap.isNull():
            print("DEBUG: Failed to load real image")
            return

        # Load the fake image in grayscale
        fake_image_gray = cv2.imread(fake_image_path, cv2.IMREAD_GRAYSCALE)
        if fake_image_gray is None:
            print("DEBUG: Failed to load fake image")
            return

        # Apply Otsu's threshold to convert to binary
        _, binary_mask = cv2.threshold(
            fake_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Convert real image to QImage for painting
        real_image = real_pixmap.toImage()

        # Ensure both images are the same size
        if real_image.size() != (binary_mask.shape[1], binary_mask.shape[0]):
            print(
                "DEBUG: Image sizes do not match. Resizing binary mask to match real image."
            )
            binary_mask = cv2.resize(
                binary_mask,
                (real_image.width(), real_image.height()),
                interpolation=cv2.INTER_NEAREST,
            )

        result_image = QImage(
            real_image.size(), QImage.Format.Format_ARGB32_Premultiplied
        )
        result_image.fill(Qt.GlobalColor.transparent)

        # Start painting the overlay
        painter = QPainter(result_image)
        painter.drawImage(0, 0, real_image)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # Overlay the binary mask (where mask is 0, keep real image; where mask is >=50, set black)
        for x in range(real_image.width()):
            for y in range(real_image.height()):
                if (
                    binary_mask[y, x] >= 50
                ):  # Binary mask, 255 means foreground (white pixels in the fake image)
                    result_image.setPixelColor(
                        x, y, QColor(Qt.GlobalColor.black)
                    )  # Overlay black color on the real image

        painter.end()

        # Convert result to QPixmap and display
        result_pixmap = QPixmap.fromImage(result_image)
        self.image_label.setPixmap(result_pixmap)
        print(f"DEBUG: Overlay image displayed. Size: {result_pixmap.size()}")

    def display_side_by_side_images(self):
        real_pixmap = QPixmap(self.current_image_path)
        if self.current_fake_image_path:
            fake_pixmap = QPixmap(self.current_fake_image_path)
            combined_pixmap = QPixmap(real_pixmap.width() * 2, real_pixmap.height())
            combined_pixmap.fill(Qt.GlobalColor.white)
            painter = QPainter(combined_pixmap)
            painter.drawPixmap(0, 0, real_pixmap)
            painter.drawPixmap(real_pixmap.width(), 0, fake_pixmap)
            painter.end()
        else:
            combined_pixmap = real_pixmap

        scaled_pixmap = combined_pixmap.scaled(
            800, 600, Qt.AspectRatioMode.KeepAspectRatio
        )
        self.image_label.setPixmap(scaled_pixmap)
        print(f"DEBUG: Side-by-side images displayed. Size: {scaled_pixmap.size()}")

    def generate_skeleton(self):
        print("DEBUG: Generate skeleton button clicked")
        self.skeleton_handler.generate_skeleton()

    def calculate_root_length(self):
        if not self.fake_images:
            QMessageBox.warning(self, "Warning", "No skeleton images loaded.")
            return

        output_dir = os.path.dirname(next(iter(self.fake_images.values())))
        self.calculator_thread = RootLengthCalculatorThread(
            self.fake_images, output_dir
        )
        self.calculator_thread.finished.connect(self.on_calculation_finished)
        self.calculator_thread.progress.connect(self.update_progress)
        self.calculator_thread.start()

        self.status_bar.showMessage("Calculating root lengths...")

    def on_calculation_finished(self, csv_path):
        self.status_bar.showMessage("Root length calculation completed.", 5000)
        QMessageBox.information(
            self, "Calculation Complete", f"Results saved to {csv_path}"
        )

    def update_progress(self, value):
        self.status_bar.showMessage(f"Calculating root lengths... {value}%")

    def update_image_display(self):
        view_mode = self.view_mode_combo.currentText()

        # Hide both image label and web view initially
        self.image_label.hide()
        self.web_view.hide()

        if view_mode == "Basic View":
            if (
                self.has_fake_real_pairs
                and self.html_path
                and os.path.exists(self.html_path)
            ):
                self.web_view.show()
                self.web_view.load(QUrl.fromLocalFile(self.html_path))
                print(f"DEBUG: Basic View displayed using HTML file: {self.html_path}")
            else:
                print(
                    "DEBUG: Basic View not available. Displaying single image instead."
                )
                self.display_single_image()
                QMessageBox.information(
                    self,
                    "Information",
                    "Basic View is only available for fake-real image pairs.",
                )
        else:
            self.image_label.show()
            if self.current_image_path:
                if view_mode == "Single Image":
                    self.display_single_image()
                elif view_mode == "Overlay":
                    self.display_overlay_image()
                elif view_mode == "Side by Side":
                    self.display_side_by_side_images()
            else:
                self.image_label.setText("Select an image to display")


if __name__ == "__main__":
    print("DEBUG: Starting application")
    app = QApplication(sys.argv)
    window = RootSkeletonViewerUI()
    window.show()
    print("DEBUG: Window displayed")
    sys.exit(app.exec())
