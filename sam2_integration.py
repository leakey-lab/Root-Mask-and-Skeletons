import numpy as np
import torch
import cv2
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QComboBox,
    QCheckBox,
    QMessageBox,
    QGraphicsEllipseItem,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPen, QBrush
import os
from typing import List, Tuple, Optional, Dict, Any


try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print(
        "Warning: SAM2 not available. Please install sam2 package for advanced segmentation."
    )


class SAM2LoadingThread(QThread):
    """Thread for loading SAM2 model to prevent UI freezing"""

    model_loaded = pyqtSignal(object, object)  # predictor, mask_generator
    error_occurred = pyqtSignal(str)
    loading_progress = pyqtSignal(str)  # Status message

    def __init__(self, model_path: str, config_file: str, device: str):
        super().__init__()
        self.model_path = model_path
        self.config_file = config_file
        self.device = device

    def run(self):
        try:
            if not SAM2_AVAILABLE:
                raise ImportError("SAM2 package not available")

            self.loading_progress.emit("Loading SAM2 model...")

            # Load model
            model = build_sam2(config_file=self.config_file, device=self.device)

            if os.path.exists(self.model_path):
                self.loading_progress.emit("Loading checkpoint...")
                checkpoint = torch.load(
                    self.model_path, map_location="cpu", weights_only=False
                )
                state_dict = checkpoint.get(
                    "state_dict", checkpoint.get("model", checkpoint)
                )
                model.load_state_dict(state_dict, strict=False)

            self.loading_progress.emit("Initializing predictor...")
            predictor = SAM2ImagePredictor(model)

            self.loading_progress.emit("Initializing mask generator...")
            mask_generator = SAM2AutomaticMaskGenerator(
                model=model,
                points_per_side=32,
                points_per_batch=64,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.92,
                min_mask_region_area=15,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                box_nms_thresh=0.7,
                use_m2m=True,
            )

            self.loading_progress.emit("SAM2 ready!")
            self.model_loaded.emit(predictor, mask_generator)

        except Exception as e:
            self.error_occurred.emit(str(e))


class SAM2PredictionThread(QThread):
    """Thread for SAM2 predictions to prevent UI freezing"""

    prediction_ready = pyqtSignal(np.ndarray)  # mask
    error_occurred = pyqtSignal(str)

    def __init__(
        self, predictor, image, points, labels, bboxes=None, multimask_output=True
    ):
        super().__init__()
        self.predictor = predictor
        self.image = image
        self.points = points
        self.labels = labels
        self.bboxes = bboxes
        self.multimask_output = multimask_output

    def run(self):
        try:
            point_coords = np.array(self.points) if self.points else None
            point_labels = np.array(self.labels) if self.labels else None
            box = np.array(self.bboxes[0]) if self.bboxes else None

            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=self.multimask_output,
            )

            if masks is None or len(masks) == 0:
                mask = np.zeros(self.image.shape[:2], dtype=bool)
            else:
                # Return best mask
                best_idx = np.argmax(scores)
                mask = masks[best_idx].astype(bool)

            self.prediction_ready.emit(mask)

        except Exception as e:
            self.error_occurred.emit(str(e))


class SAM2AutoMaskThread(QThread):
    """Thread for automatic mask generation"""

    masks_ready = pyqtSignal(list)  # List of mask dictionaries
    error_occurred = pyqtSignal(str)

    def __init__(self, mask_generator, image):
        super().__init__()
        self.mask_generator = mask_generator
        self.image = image

    def run(self):
        try:
            masks = self.mask_generator.generate(self.image)
            # Sort by area (smallest first for easier selection)
            masks.sort(key=lambda x: x.get("area", 0))
            self.masks_ready.emit(masks)
        except Exception as e:
            self.error_occurred.emit(str(e))


class SAM2Controls(QGroupBox):
    """SAM2 control panel widget"""

    def __init__(self, parent=None):
        super().__init__("SAM2 Controls", parent)
        self.predictor = None
        self.mask_generator = None
        self.sam_points = []  # [(x, y, label), ...]
        self.sam_bboxes = []
        self.generated_masks = []
        self.auto_select_mode = False

        # Default model paths - should be configurable
        self.model_path = "model_best.pt"
        self.config_file = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.init_ui()

        # Try to load model on initialization
        if SAM2_AVAILABLE:
            self.load_sam2_model()

    def init_ui(self):
        layout = QVBoxLayout()

        # Model status and loading
        status_layout = QHBoxLayout()
        self.status_label = QLabel("SAM2: Not loaded")
        self.load_button = QPushButton("Load SAM2")
        self.load_button.clicked.connect(self.load_sam2_model)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.load_button)
        layout.addLayout(status_layout)

        # SAM2 prediction controls
        pred_group = QGroupBox("Point Prompts")
        pred_layout = QVBoxLayout()

        # Instructions
        instructions = QLabel("Left click: Positive point\nRight click: Negative point")
        instructions.setStyleSheet("color: #999; font-size: 10px;")
        pred_layout.addWidget(instructions)

        # Predict button
        predict_layout = QHBoxLayout()
        self.predict_button = QPushButton("🎯 Predict Mask")
        self.predict_button.clicked.connect(self.predict_mask)
        self.predict_button.setEnabled(False)

        self.multimask_checkbox = QCheckBox("Multi-mask")
        self.multimask_checkbox.setChecked(True)

        predict_layout.addWidget(self.predict_button)
        predict_layout.addWidget(self.multimask_checkbox)
        pred_layout.addLayout(predict_layout)

        # Clear points button
        self.clear_points_button = QPushButton("Clear Points")
        self.clear_points_button.clicked.connect(self.clear_sam_points)
        pred_layout.addWidget(self.clear_points_button)

        pred_group.setLayout(pred_layout)
        layout.addWidget(pred_group)

        # Auto segmentation controls
        auto_group = QGroupBox("Auto Segmentation")
        auto_layout = QVBoxLayout()

        self.auto_segment_button = QPushButton("🤖 Generate All Masks")
        self.auto_segment_button.clicked.connect(self.generate_auto_masks)
        self.auto_segment_button.setEnabled(False)

        self.auto_select_button = QPushButton("Enter Auto-Select Mode")
        self.auto_select_button.clicked.connect(self.toggle_auto_select_mode)
        self.auto_select_button.setEnabled(False)

        auto_layout.addWidget(self.auto_segment_button)
        auto_layout.addWidget(self.auto_select_button)

        auto_group.setLayout(auto_layout)
        layout.addWidget(auto_group)

        # Model settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()

        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        self.device_combo.setCurrentText(self.device)
        device_layout.addWidget(self.device_combo)
        settings_layout.addLayout(device_layout)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Style the buttons
        button_style = """
            QPushButton {
                background-color: #2d2d2d;
                border: none;
                border-radius: 4px;
                padding: 6px;
                color: white;
                font-size: 12px;
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #404040;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                color: #666;
            }
        """

        for button in [
            self.load_button,
            self.predict_button,
            self.clear_points_button,
            self.auto_segment_button,
            self.auto_select_button,
        ]:
            button.setStyleSheet(button_style)

        self.setLayout(layout)

    def load_sam2_model(self):
        """Load SAM2 model in background thread"""
        if not SAM2_AVAILABLE:
            QMessageBox.warning(
                self,
                "SAM2 Not Available",
                "SAM2 package is not installed. Please install it first.",
            )
            return

        self.status_label.setText("SAM2: Loading...")
        self.load_button.setEnabled(False)

        # Update device from combo
        self.device = self.device_combo.currentText()

        self.loading_thread = SAM2LoadingThread(
            self.model_path, self.config_file, self.device
        )
        self.loading_thread.model_loaded.connect(self.on_model_loaded)
        self.loading_thread.error_occurred.connect(self.on_loading_error)
        self.loading_thread.loading_progress.connect(self.status_label.setText)
        self.loading_thread.start()

    def on_model_loaded(self, predictor, mask_generator):
        """Handle successful model loading"""
        self.predictor = predictor
        self.mask_generator = mask_generator
        self.status_label.setText("SAM2: Ready")
        self.load_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.auto_segment_button.setEnabled(True)
        print("SAM2 model loaded successfully!")

    def on_loading_error(self, error_msg):
        """Handle model loading error"""
        self.status_label.setText("SAM2: Error")
        self.load_button.setEnabled(True)
        QMessageBox.critical(
            self, "SAM2 Loading Error", f"Failed to load SAM2: {error_msg}"
        )

    def set_image(self, image_array):
        """Set the current image for SAM2"""
        if self.predictor is not None:
            self.predictor.set_image(image_array)
            self.clear_sam_points()
            self.generated_masks = []
            self.auto_select_mode = False
            self.auto_select_button.setText("Enter Auto-Select Mode")
            self.auto_select_button.setEnabled(
                False if not self.mask_generator else True
            )

    def add_sam_point(self, x, y, is_positive=True):
        """Add a point prompt for SAM2"""
        label = 1 if is_positive else 0
        self.sam_points.append((x, y, label))
        print(
            f"Added SAM2 point: ({x}, {y}) - {'Positive' if is_positive else 'Negative'}"
        )

    def clear_sam_points(self):
        """Clear all SAM2 points"""
        self.sam_points = []
        self.sam_bboxes = []
        print("Cleared SAM2 points")

    def predict_mask(self):
        """Predict mask using current points"""
        if self.predictor is None:
            QMessageBox.warning(self, "SAM2 Not Ready", "Please load SAM2 model first.")
            return

        if not self.sam_points and not self.sam_bboxes:
            QMessageBox.information(
                self, "No Prompts", "Please add points or bounding boxes first."
            )
            return

        # Extract points and labels
        points = [(p[0], p[1]) for p in self.sam_points]
        labels = [p[2] for p in self.sam_points]

        # Get current image from parent
        parent = self.parent()
        if hasattr(parent, "image_pixmap") and parent.image_pixmap:
            # Convert QPixmap to numpy array
            qimage = parent.image_pixmap.toImage()
            qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
            width = qimage.width()
            height = qimage.height()
            ptr = qimage.bits()
            ptr.setsize(height * width * 3)
            image_array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))
        else:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        # Start prediction in background thread
        self.predict_button.setText("Predicting...")
        self.predict_button.setEnabled(False)

        self.prediction_thread = SAM2PredictionThread(
            self.predictor,
            image_array,
            points,
            labels,
            self.sam_bboxes,
            self.multimask_checkbox.isChecked(),
        )
        self.prediction_thread.prediction_ready.connect(self.on_prediction_ready)
        self.prediction_thread.error_occurred.connect(self.on_prediction_error)
        self.prediction_thread.start()

    def on_prediction_ready(self, mask):
        """Handle completed prediction"""
        self.predict_button.setText("🎯 Predict Mask")
        self.predict_button.setEnabled(True)

        # Apply mask to parent interface
        parent = self.parent()
        if hasattr(parent, "apply_sam2_mask"):
            parent.apply_sam2_mask(mask)

    def on_prediction_error(self, error_msg):
        """Handle prediction error"""
        self.predict_button.setText("🎯 Predict Mask")
        self.predict_button.setEnabled(True)
        QMessageBox.critical(
            self, "Prediction Error", f"SAM2 prediction failed: {error_msg}"
        )

    def generate_auto_masks(self):
        """Generate automatic masks for the entire image"""
        if self.mask_generator is None:
            QMessageBox.warning(self, "SAM2 Not Ready", "Please load SAM2 model first.")
            return

        parent = self.parent()
        if hasattr(parent, "image_pixmap") and parent.image_pixmap:
            # Convert QPixmap to numpy array
            qimage = parent.image_pixmap.toImage()
            qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
            width = qimage.width()
            height = qimage.height()
            ptr = qimage.bits()
            ptr.setsize(height * width * 3)
            image_array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))
        else:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        # Start auto mask generation in background thread
        self.auto_segment_button.setText("Generating...")
        self.auto_segment_button.setEnabled(False)

        self.auto_mask_thread = SAM2AutoMaskThread(self.mask_generator, image_array)
        self.auto_mask_thread.masks_ready.connect(self.on_auto_masks_ready)
        self.auto_mask_thread.error_occurred.connect(self.on_auto_masks_error)
        self.auto_mask_thread.start()

    def on_auto_masks_ready(self, masks):
        """Handle completed auto mask generation"""
        self.auto_segment_button.setText("🤖 Generate All Masks")
        self.auto_segment_button.setEnabled(True)

        self.generated_masks = masks
        self.auto_select_button.setEnabled(True)

        print(f"Generated {len(masks)} automatic masks")
        QMessageBox.information(
            self,
            "Auto Segmentation Complete",
            f"Generated {len(masks)} masks. Use Auto-Select mode to choose segments.",
        )

    def on_auto_masks_error(self, error_msg):
        """Handle auto mask generation error"""
        self.auto_segment_button.setText("🤖 Generate All Masks")
        self.auto_segment_button.setEnabled(True)
        QMessageBox.critical(
            self, "Auto Segmentation Error", f"Auto mask generation failed: {error_msg}"
        )

    def toggle_auto_select_mode(self):
        """Toggle auto-select mode for choosing from generated masks"""
        if not self.generated_masks:
            QMessageBox.information(
                self, "No Auto Masks", "Please generate automatic masks first."
            )
            return

        self.auto_select_mode = not self.auto_select_mode

        if self.auto_select_mode:
            self.auto_select_button.setText("Exit Auto-Select Mode")
            print(f"Entered auto-select mode with {len(self.generated_masks)} segments")
        else:
            self.auto_select_button.setText("Enter Auto-Select Mode")
            print("Exited auto-select mode")

        # Notify parent about mode change
        parent = self.parent()
        if hasattr(parent, "set_sam2_auto_select_mode"):
            parent.set_sam2_auto_select_mode(
                self.auto_select_mode, self.generated_masks
            )

    def handle_auto_select_click(self, x, y, add_to_mask=True):
        """Handle click in auto-select mode"""
        if not self.auto_select_mode or not self.generated_masks:
            return False

        # Find which mask contains this point
        for mask_data in self.generated_masks:
            mask = mask_data["segmentation"]
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                # Found a mask containing this point
                parent = self.parent()
                if hasattr(parent, "apply_sam2_auto_mask"):
                    parent.apply_sam2_auto_mask(mask.astype(bool), add_to_mask)
                return True

        return False


# Integration function to add SAM2 controls to existing mask tracing interface
def integrate_sam2_to_mask_interface(mask_tracing_interface):
    """
    Integrate SAM2 controls into the existing mask tracing interface
    """
    if not hasattr(mask_tracing_interface, "sam2_controls"):
        # Add SAM2 controls to the interface
        mask_tracing_interface.sam2_controls = SAM2Controls(mask_tracing_interface)

        # Add to the layout (find the control panel)
        control_panel = None
        for child in mask_tracing_interface.findChildren(QWidget):
            if child.styleSheet() and "background-color: #1e1e1e" in child.styleSheet():
                control_panel = child
                break

        if control_panel and hasattr(control_panel, "layout"):
            control_panel.layout().addWidget(mask_tracing_interface.sam2_controls)

        # Connect SAM2 to the interface
        _connect_sam2_to_interface(mask_tracing_interface)


def _connect_sam2_to_interface(interface):
    """Connect SAM2 functionality to the mask tracing interface"""

    # Store original methods
    interface._original_load_image = interface.load_image
    interface._original_mousePressEvent = interface.graphics_view.mousePressEvent

    def enhanced_load_image(image_path):
        """Enhanced load_image that also sets SAM2 image"""
        result = interface._original_load_image(image_path)

        # Set image for SAM2
        if hasattr(interface, "sam2_controls") and interface.image_pixmap:
            qimage = interface.image_pixmap.toImage()
            qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
            width = qimage.width()
            height = qimage.height()
            ptr = qimage.bits()
            ptr.setsize(height * width * 3)
            image_array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))
            interface.sam2_controls.set_image(image_array)

        return result

    def enhanced_mouse_press(event):
        """Enhanced mouse press that handles SAM2 prompts"""
        if not hasattr(interface, "sam2_controls"):
            return interface._original_mousePressEvent(event)

        # Get click position
        pos = interface.graphics_view.map_to_image(event.position())
        x, y = pos.x(), pos.y()

        # Check if we're in auto-select mode
        if interface.sam2_controls.auto_select_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                if interface.sam2_controls.handle_auto_select_click(
                    x, y, add_to_mask=True
                ):
                    return
            elif event.button() == Qt.MouseButton.RightButton:
                if interface.sam2_controls.handle_auto_select_click(
                    x, y, add_to_mask=False
                ):
                    return

        # Check if Ctrl is pressed for SAM2 point prompts
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.button() == Qt.MouseButton.LeftButton:
                interface.sam2_controls.add_sam_point(x, y, is_positive=True)
                interface.update_display()  # Refresh display to show point
                return
            elif event.button() == Qt.MouseButton.RightButton:
                interface.sam2_controls.add_sam_point(x, y, is_positive=False)
                interface.update_display()  # Refresh display to show point
                return

        # Fall back to original behavior
        return interface._original_mousePressEvent(event)

    def apply_sam2_mask(mask):
        """Apply SAM2 predicted mask to the interface"""
        if interface.mask_pixmap is None:
            interface.mask_pixmap = QPixmap(interface.image_pixmap.size())
            interface.mask_pixmap.fill(Qt.GlobalColor.transparent)

        # Save for undo
        interface.save_for_undo()

        # Convert numpy mask to QPixmap
        height, width = mask.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[mask] = [255, 255, 255, 255]  # White with full opacity

        mask_qimage = QImage(
            rgba.data, width, height, width * 4, QImage.Format.Format_RGBA8888
        )
        sam_mask_pixmap = QPixmap.fromImage(mask_qimage)

        # Combine with existing mask
        interface.mask_pixmap = sam_mask_pixmap
        interface.update_display()

        # Clear SAM2 points after successful prediction
        interface.sam2_controls.clear_sam_points()

    def apply_sam2_auto_mask(mask, add_to_mask=True):
        """Apply auto-selected mask to the interface"""
        if interface.mask_pixmap is None:
            interface.mask_pixmap = QPixmap(interface.image_pixmap.size())
            interface.mask_pixmap.fill(Qt.GlobalColor.transparent)

        # Save for undo
        interface.save_for_undo()

        # Convert mask to QImage
        height, width = mask.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[mask] = [255, 255, 255, 255]

        mask_qimage = QImage(
            rgba.data, width, height, width * 4, QImage.Format.Format_RGBA8888
        )
        new_mask_pixmap = QPixmap.fromImage(mask_qimage)

        if add_to_mask:
            # Add to existing mask
            from PyQt6.QtGui import QPainter

            painter = QPainter(interface.mask_pixmap)
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )
            painter.drawPixmap(0, 0, new_mask_pixmap)
            painter.end()
        else:
            # Remove from existing mask
            from PyQt6.QtGui import QPainter

            painter = QPainter(interface.mask_pixmap)
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_DestinationOut
            )
            painter.drawPixmap(0, 0, new_mask_pixmap)
            painter.end()

        interface.update_display()

    def set_sam2_auto_select_mode(mode, masks):
        """Set auto-select mode state"""
        # Could be used to change cursor or UI state
        pass

    # Store original update_display
    interface._original_update_display = interface.update_display

    def enhanced_update_display():
        """Enhanced update display that also shows SAM2 points"""
        interface._original_update_display()

        if hasattr(interface, "sam2_controls") and interface.sam2_controls.sam_points:
            for x, y, label in interface.sam2_controls.sam_points:
                color = Qt.GlobalColor.green if label == 1 else Qt.GlobalColor.red
                point_item = QGraphicsEllipseItem(x - 3, y - 3, 6, 6)
                point_item.setPen(QPen(color, 2))
                point_item.setBrush(QBrush(color))
                interface.scene.addItem(point_item)

    # Replace methods
    interface.load_image = enhanced_load_image
    interface.graphics_view.mousePressEvent = enhanced_mouse_press
    interface.apply_sam2_mask = apply_sam2_mask
    interface.apply_sam2_auto_mask = apply_sam2_auto_mask
    interface.set_sam2_auto_select_mode = set_sam2_auto_select_mode
    interface.update_display = enhanced_update_display
