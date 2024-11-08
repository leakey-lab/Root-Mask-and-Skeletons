from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QComboBox,
    QGroupBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
import cv2
import numpy as np


class ImageNormalization:
    """Handles image normalization techniques"""

    @staticmethod
    def apply_clahe(img, clip_limit=2.0, tile_size=(8, 8)):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        if len(img.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            cl = clahe.apply(l)

            # Merge channels and convert back to BGR
            enhanced_lab = cv2.merge((cl, a, b))
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            return clahe.apply(img)

    @staticmethod
    def apply_contrast_stretching(img, lower_percentile=2, upper_percentile=98):
        """Apply contrast stretching using percentile-based normalization"""
        if len(img.shape) == 3:
            channels = cv2.split(img)
            normalized_channels = []

            for channel in channels:
                lower = np.percentile(channel, lower_percentile)
                upper = np.percentile(channel, upper_percentile)
                normalized = np.clip(
                    (channel - lower) * 255.0 / (upper - lower), 0, 255
                )
                normalized_channels.append(normalized.astype(np.uint8))

            return cv2.merge(normalized_channels)
        else:
            lower = np.percentile(img, lower_percentile)
            upper = np.percentile(img, upper_percentile)
            normalized = np.clip((img - lower) * 255.0 / (upper - lower), 0, 255)
            return normalized.astype(np.uint8)


class NormalizationControls(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Image Enhancement Controls", parent)

        # Default values that can be saved
        self.default_values = {
            "clahe_clip": 20,  # 2.0 after division by 10
            "clahe_tile": 8,
            "contrast_lower": 2,
            "contrast_upper": 98,
            "last_method": "CLAHE",  # Remember last used method
        }

        # Try to load saved defaults
        self.load_defaults()
        self.init_ui()

    def save_current_as_defaults(self):
        """Save current settings as defaults"""
        self.default_values.update(
            {
                "clahe_clip": self.clip_slider.value(),
                "clahe_tile": self.tile_slider.value(),
                "contrast_lower": self.lower_slider.value(),
                "contrast_upper": self.upper_slider.value(),
                "last_method": self.method_combo.currentText(),
            }
        )

        # Save to file
        import json
        import os

        save_path = os.path.join(
            os.path.dirname(__file__), "normalization_defaults.json"
        )
        try:
            with open(save_path, "w") as f:
                json.dump(self.default_values, f)
        except Exception as e:
            print(f"Failed to save defaults: {e}")

    def load_defaults(self):
        """Load saved defaults if they exist"""
        import json
        import os

        save_path = os.path.join(
            os.path.dirname(__file__), "normalization_defaults.json"
        )
        try:
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    loaded_defaults = json.load(f)
                    self.default_values.update(loaded_defaults)
        except Exception as e:
            print(f"Failed to load defaults: {e}")

    def init_ui(self):
        layout = QVBoxLayout()

        # Method selector
        method_layout = QHBoxLayout()
        self.method_label = QLabel("Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["CLAHE", "Contrast Stretching"])
        # Set last used method
        self.method_combo.setCurrentText(self.default_values["last_method"])
        method_layout.addWidget(self.method_label)
        method_layout.addWidget(self.method_combo)
        layout.addLayout(method_layout)

        # CLAHE controls
        self.clahe_group = QGroupBox("CLAHE Parameters")
        clahe_layout = QVBoxLayout()

        # Clip limit slider
        clip_layout = QHBoxLayout()
        self.clip_label = QLabel(
            f"Clip Limit: {self.default_values['clahe_clip']/10.0:.1f}"
        )
        self.clip_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_slider.setRange(1, 50)
        self.clip_slider.setValue(self.default_values["clahe_clip"])
        clip_layout.addWidget(self.clip_label)
        clip_layout.addWidget(self.clip_slider)
        clahe_layout.addLayout(clip_layout)

        # Tile size slider
        tile_layout = QHBoxLayout()
        self.tile_label = QLabel(
            f"Tile Size: {self.default_values['clahe_tile']}x{self.default_values['clahe_tile']}"
        )
        self.tile_slider = QSlider(Qt.Orientation.Horizontal)
        self.tile_slider.setRange(2, 16)
        self.tile_slider.setValue(self.default_values["clahe_tile"])
        tile_layout.addWidget(self.tile_label)
        tile_layout.addWidget(self.tile_slider)
        clahe_layout.addLayout(tile_layout)

        self.clahe_group.setLayout(clahe_layout)
        layout.addWidget(self.clahe_group)

        # Contrast stretching controls
        self.contrast_group = QGroupBox("Contrast Stretching Parameters")
        contrast_layout = QVBoxLayout()

        # Percentile sliders
        lower_layout = QHBoxLayout()
        self.lower_label = QLabel(
            f"Lower Percentile: {self.default_values['contrast_lower']}%"
        )
        self.lower_slider = QSlider(Qt.Orientation.Horizontal)
        self.lower_slider.setRange(0, 49)
        self.lower_slider.setValue(self.default_values["contrast_lower"])
        lower_layout.addWidget(self.lower_label)
        lower_layout.addWidget(self.lower_slider)
        contrast_layout.addLayout(lower_layout)

        upper_layout = QHBoxLayout()
        self.upper_label = QLabel(
            f"Upper Percentile: {self.default_values['contrast_upper']}%"
        )
        self.upper_slider = QSlider(Qt.Orientation.Horizontal)
        self.upper_slider.setRange(51, 100)
        self.upper_slider.setValue(self.default_values["contrast_upper"])
        upper_layout.addWidget(self.upper_label)
        upper_layout.addWidget(self.upper_slider)
        contrast_layout.addLayout(upper_layout)

        self.contrast_group.setLayout(contrast_layout)
        layout.addWidget(self.contrast_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.apply_button = QPushButton("Apply Enhancement")
        button_layout.addWidget(self.apply_button)

        self.save_defaults_button = QPushButton("Save as Defaults")
        self.save_defaults_button.clicked.connect(self.save_current_as_defaults)
        button_layout.addWidget(self.save_defaults_button)

        # Restore defaults button
        self.restore_defaults_button = QPushButton("Restore Defaults")
        self.restore_defaults_button.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.restore_defaults_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect signals
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        self.clip_slider.valueChanged.connect(self.on_clip_changed)
        self.tile_slider.valueChanged.connect(self.on_tile_changed)
        self.lower_slider.valueChanged.connect(self.on_percentile_changed)
        self.upper_slider.valueChanged.connect(self.on_percentile_changed)

        # Initial state
        self.on_method_changed(self.method_combo.currentText())

    def set_image(self, image):
        """Set the current image for enhancement"""
        self.current_image = image
        self.apply_button.setEnabled(image is not None)

    def reset_to_defaults(self):
        """Reset all controls to their original default values"""
        # Reset method
        self.method_combo.setCurrentText("CLAHE")

        # Reset CLAHE parameters
        self.clip_slider.setValue(20)  # 2.0 after division by 10
        self.tile_slider.setValue(8)

        # Reset Contrast Stretching parameters
        self.lower_slider.setValue(2)
        self.upper_slider.setValue(98)

        # Update labels
        self.on_clip_changed(20)
        self.on_tile_changed(8)
        self.on_percentile_changed()

    def on_method_changed(self, method):
        """Handle visibility of parameter groups based on selected method"""
        self.clahe_group.setVisible(method == "CLAHE")
        self.contrast_group.setVisible(method == "Contrast Stretching")

    def on_clip_changed(self, value):
        """Update clip limit label"""
        self.clip_label.setText(f"Clip Limit: {value/10.0:.1f}")

    def on_tile_changed(self, value):
        """Update tile size label"""
        self.tile_label.setText(f"Tile Size: {value}x{value}")

    def on_percentile_changed(self):
        """Handle percentile slider changes and ensure valid range"""
        lower = self.lower_slider.value()
        upper = self.upper_slider.value()

        if lower >= upper:
            if self.sender() == self.lower_slider:
                self.lower_slider.setValue(upper - 1)
            else:
                self.upper_slider.setValue(lower + 1)

        self.lower_label.setText(f"Lower Percentile: {self.lower_slider.value()}%")
        self.upper_label.setText(f"Upper Percentile: {self.upper_slider.value()}%")
