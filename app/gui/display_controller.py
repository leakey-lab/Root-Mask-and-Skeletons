from PyQt6.QtWidgets import (
    QVBoxLayout,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QWheelEvent
from PyQt6.QtCore import Qt
import cv2
import numpy as np
import os


class MagnifyingGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        
        # Set up high-quality rendering
        self.setup_high_quality_rendering()
        
        # Zoom control
        self.zoom = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        
        # Set transformation anchor for better zoom quality
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def setup_high_quality_rendering(self):
        """Configure high-quality rendering settings to reduce noise during zoom"""
        # Enable all quality render hints
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.LosslessImageRendering
            | QPainter.RenderHint.TextAntialiasing
        )
        
        # Enable caching for better performance
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        
        # Optimize for interactive use
        self.setOptimizationFlags(
            QGraphicsView.OptimizationFlag.DontSavePainterState
            | QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing
        )

    def wheelEvent(self, event: QWheelEvent):
        """Enhanced wheel event with high-quality zooming"""
        # Check for Ctrl modifier for zoom
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            zoom_in = event.angleDelta().y() > 0
            
            if zoom_in and self.zoom < self.max_zoom:
                factor = 1.25
                self.zoom *= factor
            elif not zoom_in and self.zoom > self.min_zoom:
                factor = 0.8
                self.zoom *= factor
            else:
                event.ignore()
                return
            
            # Apply high-quality scaling
            self.scale_with_quality(factor)
            event.accept()
        else:
            # Allow normal scrolling when Ctrl is not pressed
            super().wheelEvent(event)

    def scale_with_quality(self, factor):
        """Apply scaling with high quality settings to reduce noise"""
        # Ensure smooth transformation for all items before scaling
        if self.scene():
            for item in self.scene().items():
                if isinstance(item, QGraphicsPixmapItem):
                    item.setTransformationMode(
                        Qt.TransformationMode.SmoothTransformation
                    )
                    item.setCacheMode(
                        QGraphicsPixmapItem.CacheMode.DeviceCoordinateCache
                    )
        
        # Apply the scaling
        self.scale(factor, factor)
        
        # Force a high-quality update
        self.viewport().update()
    
    def reset_zoom(self):
        """Reset zoom to fit the scene"""
        self.resetTransform()
        self.zoom = 1.0
        if self.scene():
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for zoom control"""
        if event.key() == Qt.Key.Key_0 and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl+0 to reset zoom
            self.reset_zoom()
            event.accept()
        elif event.key() == Qt.Key.Key_Plus and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl++ to zoom in
            if self.zoom < self.max_zoom:
                factor = 1.25
                self.zoom *= factor
                self.scale_with_quality(factor)
            event.accept()
        elif event.key() == Qt.Key.Key_Minus and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl+- to zoom out
            if self.zoom > self.min_zoom:
                factor = 0.8
                self.zoom *= factor
                self.scale_with_quality(factor)
            event.accept()
        else:
            super().keyPressEvent(event)


class DisplayController:
    def __init__(self, main_window):
        self.main_window = main_window
        self.magnifying_view = None
        self.current_image = None
        self.current_fake_image = None

    def setup_display_area(self, parent):
        layout = QVBoxLayout(parent)

        self.magnifying_view = MagnifyingGraphicsView()
        layout.addWidget(self.magnifying_view)
        
        # Show helpful message about zoom controls
        if self.main_window and hasattr(self.main_window, 'status_bar'):
            self.main_window.status_bar.showMessage(
                "Tip: Use Ctrl+Mouse Wheel to zoom, Ctrl+0 to reset, Ctrl+/- for keyboard zoom", 
                10000
            )

    def display_selected_image(self, item):
        """Handle image selection with lazy loading (for backward compatibility)"""
        name = item.text()
        self.display_selected_image_by_name(name)

    def display_selected_image_by_name(self, name):
        """Handle image selection by name with lazy loading"""
        self.current_image = self.main_window.image_manager.get_image_path(name)

        view_mode = self.main_window.view_mode_combo.currentText()
        if view_mode in ["Overlay", "Side by Side"]:
            # Lazy load the processed image when needed
            self.current_fake_image = (
                self.main_window.image_manager.get_fake_image_path(name)
            )

            if not self.current_fake_image:
                self.main_window.status_bar.showMessage(
                    f"No processed image found for {name}", 3000
                )

        self.update_display()

    def update_display_mode(self):
        """Handle view mode changes"""
        view_mode = self.main_window.view_mode_combo.currentText()

        # Reload images based on the new view mode
        self.main_window.image_manager.reload_for_view_mode(view_mode)

        # Refresh the file list in the main window
        self.main_window.populate_file_list()

        # Update the display
        self.update_display()

    def update_display(self):
        view_mode = self.main_window.view_mode_combo.currentText()

        if self.current_image:
            self.magnifying_view.show()
            if view_mode == "Single Image":
                self.display_single_image()
            elif view_mode == "Overlay":
                self.display_overlay_image()
            elif view_mode == "Side by Side":
                self.display_side_by_side_images()
        else:
            self.clear_magnifying_view()

    def display_single_image(self):
        if self.current_image:
            pixmap = QPixmap(self.current_image)
            self.set_magnifying_view_image(pixmap)
        else:
            self.clear_magnifying_view()

    def display_overlay_image(self):
        """Display overlay with lazy-loaded processed image"""
        if not self.current_image:
            return

        # Load processed image if needed
        if not self.current_fake_image:
            base_name = os.path.splitext(os.path.basename(self.current_image))[0]
            self.current_fake_image = (
                self.main_window.image_manager.get_fake_image_path(base_name)
            )

        if not self.current_fake_image:
            # Show single image with notification that fake image is missing
            self.display_single_image()
            print(f"⚠️ No fake image found for {os.path.basename(self.current_image)} - showing single image view")
            return

        # Rest of the overlay display code remains the same
        real_image = QImage(self.current_image)
        if real_image.isNull():
            return

        fake_image_gray = cv2.imread(self.current_fake_image, cv2.IMREAD_GRAYSCALE)
        if fake_image_gray is None:
            return

        # Binarize the fake image using OTSU thresholding
        _, binary_mask = cv2.threshold(
            fake_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Resize the binary mask to match the real image size if necessary
        if (real_image.width(), real_image.height()) != (
            binary_mask.shape[1],
            binary_mask.shape[0],
        ):
            binary_mask = cv2.resize(
                binary_mask,
                (real_image.width(), real_image.height()),
                interpolation=cv2.INTER_NEAREST,
            )

        # Ensure the QImage format is ARGB32_Premultiplied for consistent handling
        if real_image.format() != QImage.Format.Format_ARGB32_Premultiplied:
            real_image = real_image.convertToFormat(
                QImage.Format.Format_ARGB32_Premultiplied
            )

        # Convert QImage to NumPy array
        ptr = real_image.bits()
        ptr.setsize(real_image.sizeInBytes())
        real_array = np.array(ptr).reshape(
            real_image.height(), real_image.width(), 4
        )  # BGRA

        # Create a boolean mask where the binary mask is active
        mask = binary_mask >= 50  # Adjust threshold as needed

        # Define semi-transparent neon green in BGRA format
        neon_green = np.array([57, 255, 20, 128], dtype=np.uint8)  # B, G, R, A
        # Breakdown:
        # B (Blue)   = 57
        # G (Green)  = 255
        # R (Red)    = 20
        # A (Alpha)  = 128 (semi-transparent)

        # Ensure neon_green is broadcastable to the real_array
        neon_green = neon_green.reshape(1, 1, 4)

        # Extract the mask indices
        mask_indices = np.where(mask)

        # Perform alpha blending only on the RGB channels
        alpha = neon_green[0, 0, 3] / 255.0  # Normalize alpha to [0, 1]
        inv_alpha = 1.0 - alpha

        # Blend the neon green with the original image
        # Only modify the RGB channels; preserve the original alpha channel
        real_array[mask_indices[0], mask_indices[1], 0] = (
            neon_green[0, 0, 0] * alpha
            + real_array[mask_indices[0], mask_indices[1], 0] * inv_alpha
        ).astype(np.uint8)  # Blue channel

        real_array[mask_indices[0], mask_indices[1], 1] = (
            neon_green[0, 0, 1] * alpha
            + real_array[mask_indices[0], mask_indices[1], 1] * inv_alpha
        ).astype(np.uint8)  # Green channel

        real_array[mask_indices[0], mask_indices[1], 2] = (
            neon_green[0, 0, 2] * alpha
            + real_array[mask_indices[0], mask_indices[1], 2] * inv_alpha
        ).astype(np.uint8)  # Red channel

        # Optionally, adjust the alpha channel if you want to modify it
        # For example, keep it as is or set to maximum
        # real_array[mask_indices[0], mask_indices[1], 3] = 255  # Full opacity

        # Convert the modified NumPy array back to QImage
        result_image = QImage(
            real_array.data,
            real_image.width(),
            real_image.height(),
            real_image.bytesPerLine(),
            QImage.Format.Format_ARGB32_Premultiplied,
        ).copy()  # Use .copy() to ensure the data is owned by QImage

        # Convert QImage to QPixmap and set it to the view
        result_pixmap = QPixmap.fromImage(result_image)
        self.set_magnifying_view_image(result_pixmap)

    def display_side_by_side_images(self):
        """Display original and processed images side by side with basic lazy loading."""
        if not self.current_image:
            return

        # Load the original image
        real_pixmap = QPixmap(self.current_image)
        if real_pixmap.isNull():
            return

        # Try to lazy load the processed image if needed
        if not self.current_fake_image:
            base_name = os.path.splitext(os.path.basename(self.current_image))[0]
            self.current_fake_image = (
                self.main_window.image_manager.get_fake_image_path(base_name)
            )

        # Create the side-by-side display
        # Create the side-by-side display
        if self.current_fake_image:
            # Convert _fake.png to _real.png for the processed image
            real_processed_path = self.current_fake_image.replace(
                "_fake.png", "_real.png"
            )
            fake_pixmap = QPixmap(self.current_fake_image)
            real_pixmap = QPixmap(real_processed_path)
            if fake_pixmap.isNull():
                self.main_window.status_bar.showMessage(
                    "Failed to load processed image", 3000
                )
                return

            # Create combined pixmap
            combined_width = real_pixmap.width() + fake_pixmap.width()
            combined_height = max(real_pixmap.height(), fake_pixmap.height())
            combined_pixmap = QPixmap(combined_width, combined_height)
            combined_pixmap.fill(Qt.GlobalColor.white)

            # Draw the images side by side
            painter = QPainter(combined_pixmap)
            painter.drawPixmap(0, 0, real_pixmap)
            painter.drawPixmap(real_pixmap.width(), 0, fake_pixmap)
            painter.end()
        else:
            # If no processed image, just show original
            combined_pixmap = real_pixmap
            self.main_window.status_bar.showMessage(
                "Processed image not available", 3000
            )

        # Set up the scene and view with high-quality settings
        scene = QGraphicsScene()
        pixmap_item = QGraphicsPixmapItem(combined_pixmap)
        
        # Set high-quality transformation mode for the pixmap item
        pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        pixmap_item.setCacheMode(QGraphicsPixmapItem.CacheMode.DeviceCoordinateCache)
        
        scene.addItem(pixmap_item)
        self.magnifying_view.setScene(scene)

        # Fit view while maintaining aspect ratio
        self.magnifying_view.fitInView(
            scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

        # Reset zoom
        self.magnifying_view.zoom = 1.0

    def set_magnifying_view_image(self, pixmap):
        scene = QGraphicsScene()
        pixmap_item = QGraphicsPixmapItem(pixmap)
        
        # Set high-quality transformation mode for the pixmap item
        pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        pixmap_item.setCacheMode(QGraphicsPixmapItem.CacheMode.DeviceCoordinateCache)
        
        scene.addItem(pixmap_item)
        self.magnifying_view.setScene(scene)
        
        # Fit view while maintaining aspect ratio
        self.magnifying_view.fitInView(
            scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )
        
        # Reset zoom level when setting new image
        self.magnifying_view.zoom = 1.0

    def clear_magnifying_view(self):
        scene = QGraphicsScene()
        scene.addText("Select an image to display")
        self.magnifying_view.setScene(scene)
        self.magnifying_view.show()
