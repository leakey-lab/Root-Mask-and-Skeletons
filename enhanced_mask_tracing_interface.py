# enhanced_mask_tracing_interface.py
# Updated mask tracing interface with integrated SAM2 support

from mask_tracing_interface import MaskTracingInterface
from sam2_integration import SAM2Controls, integrate_sam2_to_mask_interface
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGraphicsEllipseItem,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPen, QBrush, QPixmap, QImage, QPainter
import numpy as np


class EnhancedMaskTracingInterface(MaskTracingInterface):
    """
    Enhanced mask tracing interface with integrated SAM2 support
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_sam2_integration()

    def setup_sam2_integration(self):
        """Setup SAM2 integration"""
        # Add SAM2 controls to the control panel
        self.sam2_controls = SAM2Controls(self)

        # Find the control panel and add SAM2 controls
        control_widget = None
        for child in self.findChildren(QWidget):
            if child.styleSheet() and "background-color: #1e1e1e" in child.styleSheet():
                control_widget = child
                break

        if control_widget and hasattr(control_widget, "layout"):
            # Add SAM2 controls to the layout
            control_widget.layout().addWidget(self.sam2_controls)

        # Add instructions for SAM2
        self.add_sam2_instructions()

        # Connect SAM2 functionality
        self.connect_sam2_functionality()

    def add_sam2_instructions(self):
        """Add instructions for SAM2 usage"""
        # Create instructions widget
        instructions_widget = QWidget()
        instructions_widget.setMaximumHeight(80)
        instructions_widget.setStyleSheet("""
            QWidget {
                background-color: #2a2a2a;
                border-radius: 5px;
                margin: 2px;
            }
            QLabel {
                color: #cccccc;
                padding: 5px;
            }
        """)

        instructions_layout = QVBoxLayout(instructions_widget)

        title_label = QLabel("SAM2 Integration")
        title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        instructions_text = QLabel(
            "• Ctrl + Left Click: Add positive point\n"
            "• Ctrl + Right Click: Add negative point\n"
            "• Auto-Select: Click generated segments to add/remove"
        )
        instructions_text.setWordWrap(True)
        instructions_text.setStyleSheet("font-size: 9px; color: #aaa;")

        instructions_layout.addWidget(title_label)
        instructions_layout.addWidget(instructions_text)

        # Add to main layout at the top
        self.layout().insertWidget(0, instructions_widget)

    def connect_sam2_functionality(self):
        """Connect SAM2 functionality to the interface"""
        # Store original methods
        self._original_load_image = self.load_image
        self._original_mouse_press = self.graphics_view.mousePressEvent
        self._original_update_display = self.update_display

        # Replace with enhanced methods
        self.load_image = self.enhanced_load_image
        self.graphics_view.mousePressEvent = self.enhanced_mouse_press
        self.update_display = self.enhanced_update_display

    def enhanced_load_image(self, image_path):
        """Enhanced load_image that also sets SAM2 image - FIXED VERSION"""
        print(f"DEBUG: Enhanced load_image called with: {image_path}")

        # Call original load_image first
        result = self._original_load_image(image_path)

        # Set image for SAM2 with better error handling
        if hasattr(self, "sam2_controls") and self.image_pixmap:
            try:
                print("DEBUG: Setting SAM2 image...")

                # Method 1: Try using PIL to load the image directly
                try:
                    from PIL import Image
                    import numpy as np

                    # Load image directly from file path
                    pil_image = Image.open(image_path).convert("RGB")
                    image_array = np.array(pil_image)
                    print(f"DEBUG: Loaded image with shape: {image_array.shape}")

                    self.sam2_controls.set_image(image_array)
                    print("DEBUG: SAM2 image set successfully using PIL method")

                except Exception as e:
                    print(f"DEBUG: PIL method failed: {e}, trying QPixmap method...")

                    # Method 2: Convert from QPixmap (fallback)
                    qimage = self.image_pixmap.toImage()
                    qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
                    width = qimage.width()
                    height = qimage.height()
                    ptr = qimage.bits()
                    ptr.setsize(height * width * 3)
                    image_array = np.frombuffer(ptr, dtype=np.uint8).reshape(
                        (height, width, 3)
                    )
                    print(
                        f"DEBUG: Converted QPixmap to array with shape: {image_array.shape}"
                    )

                    self.sam2_controls.set_image(image_array)
                    print("DEBUG: SAM2 image set successfully using QPixmap method")

            except Exception as e:
                print(f"ERROR: Failed to set SAM2 image: {e}")
                import traceback

                traceback.print_exc()
        else:
            if not hasattr(self, "sam2_controls"):
                print("DEBUG: No sam2_controls found")
            if not self.image_pixmap:
                print("DEBUG: No image_pixmap found")

        return result

    def enhanced_mouse_press(self, event):
        """Enhanced mouse press that handles SAM2 prompts"""
        # Get click position
        pos = self.graphics_view.map_to_image(event.position())
        x, y = pos.x(), pos.y()

        # Check if we're in auto-select mode
        if hasattr(self, "sam2_controls") and self.sam2_controls.auto_select_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                if self.sam2_controls.handle_auto_select_click(x, y, add_to_mask=True):
                    return
            elif event.button() == Qt.MouseButton.RightButton:
                if self.sam2_controls.handle_auto_select_click(x, y, add_to_mask=False):
                    return

        # Check if Ctrl is pressed for SAM2 point prompts
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.button() == Qt.MouseButton.LeftButton:
                if hasattr(self, "sam2_controls"):
                    self.sam2_controls.add_sam_point(x, y, is_positive=True)
                    self.update_display()
                    return
            elif event.button() == Qt.MouseButton.RightButton:
                if hasattr(self, "sam2_controls"):
                    self.sam2_controls.add_sam_point(x, y, is_positive=False)
                    self.update_display()
                    return

        # Fall back to original behavior
        return self._original_mouse_press(event)

    def enhanced_update_display(self):
        """Enhanced update display that also shows SAM2 points"""
        self._original_update_display()

        if hasattr(self, "sam2_controls") and self.sam2_controls.sam_points:
            for x, y, label in self.sam2_controls.sam_points:
                color = Qt.GlobalColor.green if label == 1 else Qt.GlobalColor.red
                point_item = QGraphicsEllipseItem(x - 4, y - 4, 8, 8)
                point_item.setPen(QPen(color, 2))
                point_item.setBrush(QBrush(color))
                point_item.setZValue(10)  # Ensure points are visible on top
                self.scene.addItem(point_item)

    def apply_sam2_mask(self, mask):
        """Apply SAM2 predicted mask to the interface"""

        if self.mask_pixmap is None:
            self.mask_pixmap = QPixmap(self.image_pixmap.size())
            self.mask_pixmap.fill(Qt.GlobalColor.transparent)

        # Save for undo
        self.save_for_undo()

        # Convert numpy mask to QPixmap
        height, width = mask.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[mask] = [255, 255, 255, 255]  # White with full opacity

        mask_qimage = QImage(
            rgba.data, width, height, width * 4, QImage.Format.Format_RGBA8888
        )
        sam_mask_pixmap = QPixmap.fromImage(mask_qimage)

        # Resize if needed to match current mask
        if sam_mask_pixmap.size() != self.mask_pixmap.size():
            sam_mask_pixmap = sam_mask_pixmap.scaled(
                self.mask_pixmap.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # Combine with existing mask using OR operation
        painter = QPainter(self.mask_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.drawPixmap(0, 0, sam_mask_pixmap)
        painter.end()

        self.update_display()

        # Clear SAM2 points after successful prediction
        if hasattr(self, "sam2_controls"):
            self.sam2_controls.clear_sam_points()

        print("SAM2 mask applied successfully!")

    def apply_sam2_auto_mask(self, mask, add_to_mask=True):
        """Apply auto-selected mask to the interface"""

        if self.mask_pixmap is None:
            self.mask_pixmap = QPixmap(self.image_pixmap.size())
            self.mask_pixmap.fill(Qt.GlobalColor.transparent)

        # Save for undo
        self.save_for_undo()

        # Convert mask to QImage
        height, width = mask.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[mask] = [255, 255, 255, 255]

        mask_qimage = QImage(
            rgba.data, width, height, width * 4, QImage.Format.Format_RGBA8888
        )
        new_mask_pixmap = QPixmap.fromImage(mask_qimage)

        # Resize if needed
        if new_mask_pixmap.size() != self.mask_pixmap.size():
            new_mask_pixmap = new_mask_pixmap.scaled(
                self.mask_pixmap.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        painter = QPainter(self.mask_pixmap)
        if add_to_mask:
            # Add to existing mask
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )
            action_text = "Added"
        else:
            # Remove from existing mask
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_DestinationOut
            )
            action_text = "Removed"

        painter.drawPixmap(0, 0, new_mask_pixmap)
        painter.end()

        self.update_display()
        print(f"{action_text} auto-selected segment")

    def set_sam2_auto_select_mode(self, mode, masks):
        """Set auto-select mode state"""
        if mode:
            self.graphics_view.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.graphics_view.setCursor(self.brush_cursor)

    def keyPressEvent(self, event):
        """Enhanced key press event with SAM2 shortcuts"""
        # SAM2 shortcuts
        if event.key() == Qt.Key.Key_P:  # P for Predict
            if hasattr(self, "sam2_controls"):
                self.sam2_controls.predict_mask()
                return
        elif (
            event.key() == Qt.Key.Key_C
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):  # Shift+C for Clear SAM2 points
            if hasattr(self, "sam2_controls"):
                self.sam2_controls.clear_sam_points()
                self.update_display()
                return
        elif (
            event.key() == Qt.Key.Key_A
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):  # Shift+A for Auto segment
            if hasattr(self, "sam2_controls"):
                self.sam2_controls.generate_auto_masks()
                return

        # Call parent keyPressEvent for other keys
        super().keyPressEvent(event)


# Factory function to create enhanced interface
def create_enhanced_mask_tracing_interface(parent=None):
    """Create an enhanced mask tracing interface with SAM2 integration"""
    return EnhancedMaskTracingInterface(parent)


# Utility function to upgrade existing interface
def upgrade_mask_tracing_interface(existing_interface):
    """Upgrade an existing mask tracing interface with SAM2 capabilities"""
    if not hasattr(existing_interface, "sam2_controls"):
        integrate_sam2_to_mask_interface(existing_interface)
        print("Successfully upgraded mask tracing interface with SAM2!")
    else:
        print("Interface already has SAM2 integration!")

    return existing_interface
