"""
Main window for the Root Viewer application.
Orchestrates the GUI components and handles core functionality.
"""

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSplitter,
    QStatusBar,
    QMessageBox,
    QStackedWidget,
    QFileDialog,
    QApplication,
)
from PyQt6.QtGui import QColor, QIcon, QCloseEvent, QShortcut, QKeySequence
from PyQt6.QtCore import Qt, pyqtSignal
from .image_manager import ImageManager
from .display_controller import DisplayController
from .mask_tracing_interface import MaskTracingInterface
from .skeleton_correction_interface import SkeletonCorrectionInterface
from . import ui_panels
from . import file_tree_manager
from . import visualization_manager
from .task_progress import TaskProgressWidget
from .empty_state import ShortcutsDialog
import logging
import os

logger = logging.getLogger(__name__)


class _ProgressBarShim:
    """Compat facade exposing the legacy QProgressBar API on TaskProgressWidget.

    The status-bar QProgressBar was replaced by TaskProgressWidget. The mask and
    skeleton *generation* handlers still call the old bar API
    (setValue/show/hide/setFormat/setTextVisible); this shim maps those onto the
    new widget so they keep working unchanged.
    """

    def __init__(self, tp):
        self._tp = tp

    def setValue(self, v):
        self._tp.set_progress(v)

    def show(self):
        # Legacy callers call show() then setValue() repeatedly; start with a
        # generic op name so the labelled widget actually appears.
        if not self._tp.isVisible():
            self._tp.start("Working")

    def hide(self):
        self._tp.finish()

    def setFormat(self, *_a, **_k):
        pass  # text-on-bar disabled; the label carries the text now

    def setTextVisible(self, *_a, **_k):
        pass

    def setRange(self, *_a, **_k):
        pass

    def setMinimumWidth(self, *_a, **_k):
        pass


class MainWindow(QMainWindow):
    mask_saved = pyqtSignal(str)
    mask_cleared = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Root Viewer")
        self.setGeometry(100, 100, 1200, 700)
        
        # Set window icon with multiple sizes for better visibility
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resources", "app_icon.ico")
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            self.setWindowIcon(icon)
        
        # Initialize components.
        #
        # Handlers are imported lazily here rather than at module top to break a
        # circular import (F: mask_handler -> app.gui (package init) ->
        # main_window -> mask_handler). By the time __init__ runs, the app.gui
        # package is fully initialized, so these imports resolve cleanly and
        # `import app.handlers.mask_handler` succeeds standalone.
        from app.handlers.skeleton_handler import SkeletonHandler
        from app.handlers.mask_handler import MaskHandler
        from app.handlers.mask_generation_handler import MaskGenerationHandler

        self.image_manager = ImageManager(self)
        self.display_controller = DisplayController(self)
        self.skeleton_handler = SkeletonHandler(self)
        self.mask_handler = MaskHandler(self)
        self.mask_generation_handler = MaskGenerationHandler(self)
        self.mask_tracing_interface = MaskTracingInterface()
        self.skeleton_correction_interface = SkeletonCorrectionInterface()
        self.root_length_viz = None
        self.root_area_viz = None

        # Tracks whether the mask-tracing interface signals are currently
        # connected, so repeated toggles do not stack duplicate connections
        # (F-015).
        self._mask_tracing_signals_connected = False

        # Connect the mask_saved and mask_cleared signals
        self.mask_saved.connect(self.highlight_saved_mask)
        self.mask_cleared.connect(self.unhighlight_cleared_mask)

        self.init_ui()

        # Keyboard-shortcut help dialog (F1 / ?).
        self._help_shortcut_f1 = QShortcut(QKeySequence("F1"), self)
        self._help_shortcut_f1.activated.connect(lambda: ShortcutsDialog(self).exec())
        self._help_shortcut_q = QShortcut(QKeySequence("?"), self)
        self._help_shortcut_q.activated.connect(lambda: ShortcutsDialog(self).exec())

    @property
    def loading_progress_bar(self):
        """Legacy QProgressBar facade backed by the TaskProgressWidget.

        Lets the mask/skeleton generation handlers keep calling the old bar API
        unchanged while the nicer status-bar widget is what actually renders.
        """
        return _ProgressBarShim(self.task_progress)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop Dash server threads before the window (and embedded viz widgets) are destroyed."""
        try:
            if self.root_length_viz is not None:
                self.close_root_length_visualization()
            if self.root_area_viz is not None:
                self.close_root_area_visualization()
        except Exception:
            logger.exception("Error cleaning up visualizations on close")
        QApplication.processEvents()
        super().closeEvent(event)

    def _build_body(self) -> QWidget:
        """Build the left/right splitter body (presentation extract of init_ui).

        Returns a container widget holding the QSplitter. PRESERVES the four
        ``right_panel.addWidget`` calls in their fixed index order
        (0=display, 1=mask_tracing, 2=visualization, 3=skeleton_correction).
        """
        body = QWidget()
        main_layout = QHBoxLayout(body)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Set splitter properties. Handle colour comes from the global SPROUTS
        # QSS (QSplitter::handle -> --border), not an inline neon override.
        splitter.setHandleWidth(5)

        # Left Panel - use extracted module
        left_panel = ui_panels.create_left_panel(self)
        splitter.addWidget(left_panel)

        # Right Panel (Stacked Widget for Display Area and Mask Tracing)
        self.right_panel = QStackedWidget()
        self.right_panel.addWidget(ui_panels.create_right_panel(self))
        self.right_panel.addWidget(self.mask_tracing_interface)
        self.visualization_widget = QWidget()  # Placeholder for visualization
        self.right_panel.addWidget(self.visualization_widget)
        self.right_panel.addWidget(self.skeleton_correction_interface)
        splitter.addWidget(self.right_panel)

        splitter.setSizes([400, 800])
        return body

    def _build_shell(self) -> QWidget:
        """Build the guided shell: titlebar / ribbon / action-bar over the body
        (stretch) over the statusline. Returns the shell container.

        Presentation wrapper only — the body (and its four right_panel indices)
        is unchanged; the chrome bands forward to existing handlers.
        """
        from . import shell_chrome

        shell = QWidget()
        col = QVBoxLayout(shell)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        col.addWidget(shell_chrome.build_titlebar(self))
        col.addWidget(shell_chrome.build_ribbon(self))
        col.addWidget(shell_chrome.build_action_bar(self))
        col.addWidget(self._build_body(), 1)
        col.addWidget(shell_chrome.build_statusline(self))
        return shell

    # Sentinel: presence signals the PR4 guided shell (app_stack) is wired, used
    # by the shell smoke test to gate the MainWindow contract assertions.
    _pr4_shell_ready = True

    def init_ui(self):
        # Guided app shell: a QStackedWidget that flips between the Welcome
        # screen (index 0) and the working shell/body (index 1). The window
        # starts on Welcome; load_images flips to the shell (T4.6).
        from app.gui.welcome_screen import WelcomeWidget
        from app.gui.loading_overlay import LoadingOverlay

        self.app_stack = QStackedWidget()
        self.welcome = WelcomeWidget(on_get_started=self.load_images)
        self.app_stack.addWidget(self.welcome)          # index 0

        shell = self._build_shell()
        self.app_stack.addWidget(shell)                 # index 1

        self.setCentralWidget(self.app_stack)
        self.app_stack.setCurrentIndex(0)

        # Full-window loading overlay (parented to the window), hidden until a
        # directory is chosen.
        self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.hide()

        # Status Bar with first-class task-progress widget.
        # Replaces the old cramped status-bar QProgressBar (text-on-bar). The
        # legacy loading_progress_bar API is preserved via the _ProgressBarShim
        # exposed by the loading_progress_bar property, so generation handlers
        # are unchanged.
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.task_progress = TaskProgressWidget()
        self.status_bar.addPermanentWidget(self.task_progress)

        # Transient success/feedback toasts (bottom-right of the window). Hard
        # errors keep using QMessageBox.critical; this is additive.
        from app.gui.widgets import ToastManager
        self.toasts = ToastManager(self)

    def notify(self, message: str, kind: str = "success", timeout: int = 3200) -> None:
        """Show a transient toast. ``kind`` in success/info/warn/danger."""
        toasts = getattr(self, "toasts", None)
        if toasts is not None:
            toasts.show(message, kind=kind, timeout=timeout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        toasts = getattr(self, "toasts", None)
        if toasts is not None:
            toasts.reposition()
        overlay = getattr(self, "loading_overlay", None)
        if overlay is not None:
            overlay.reposition()

    def set_opengl_viewports_enabled(self, enabled: bool) -> None:
        """Enable/disable QOpenGLWidget-based viewports across the app.

        Qt6 limitation: QtWebEngine (used for Dash visualizations via QWebEngineView)
        cannot render in a top-level window that contains any QOpenGLWidget.
        We disable OpenGL viewports only while visualizations are shown, and
        re-enable them afterwards for performance elsewhere.
        """
        enabled = bool(enabled)

        # Main image display view
        try:
            gv = getattr(self.display_controller, "magnifying_view", None)
            if gv is not None and hasattr(gv, "set_opengl_viewport_enabled"):
                gv.set_opengl_viewport_enabled(enabled)
        except RuntimeError as e:
            logger.warning("Failed to toggle OpenGL on display view: %s", e)

        # Mask tracing view
        try:
            gv = getattr(self.mask_tracing_interface, "graphics_view", None)
            if gv is not None and hasattr(gv, "set_opengl_viewport_enabled"):
                gv.set_opengl_viewport_enabled(enabled)
        except RuntimeError as e:
            logger.warning("Failed to toggle OpenGL on mask tracing view: %s", e)

        # Skeleton correction view
        try:
            gv = getattr(self.skeleton_correction_interface, "graphics_view", None)
            if gv is not None and hasattr(gv, "set_opengl_viewport_enabled"):
                gv.set_opengl_viewport_enabled(enabled)
        except RuntimeError as e:
            logger.warning("Failed to toggle OpenGL on skeleton correction view: %s", e)

    # ==================== File Tree Methods (delegated) ====================
    
    def find_tree_item_by_image_name(self, image_name):
        """Recursively find a tree item by its stored image name"""
        return file_tree_manager.find_tree_item_by_image_name(self.file_list, image_name)

    def highlight_saved_mask(self, image_path):
        file_tree_manager.highlight_saved_mask(self, image_path)

    def unhighlight_cleared_mask(self, image_path):
        file_tree_manager.unhighlight_cleared_mask(self, image_path)

    def populate_file_list(self):
        """Populate the file tree with hierarchical structure"""
        file_tree_manager.populate_file_list(self)

    def update_file_list_mask_status(self):
        """Update the file tree items to show mask status"""
        file_tree_manager.update_file_list_mask_status(self)

    def natural_sort_key(self, text):
        """Create a sort key that handles numbers naturally"""
        return file_tree_manager.natural_sort_key(text)

    # ==================== Visualization Methods (delegated) ====================
    
    def toggle_root_length_visualization(self):
        """Toggle between showing and hiding the root length visualization."""
        visualization_manager.toggle_root_length_visualization(self)

    def show_root_length_visualization(self):
        """Show the root length visualization."""
        visualization_manager.show_root_length_visualization(self)

    def close_root_length_visualization(self):
        """Safely close and cleanup the root length visualization."""
        visualization_manager.close_root_length_visualization(self)

    def toggle_root_area_visualization(self):
        """Toggle between showing and hiding the root area visualization."""
        visualization_manager.toggle_root_area_visualization(self)

    def show_root_area_visualization(self):
        """Show the root area visualization."""
        visualization_manager.show_root_area_visualization(self)

    def close_root_area_visualization(self):
        """Safely close and cleanup the root area visualization."""
        visualization_manager.close_root_area_visualization(self)

    def find_test_latest_dir(self, start_path, max_depth=3):
        """Efficiently search for 'test_latest' directory."""
        return visualization_manager.find_test_latest_dir(start_path, max_depth)

    def on_visualization_server_closed(self):
        """Handle root length visualization server closed event."""
        if self.root_length_viz:
            visualization_manager.close_root_length_visualization(self)

    def on_area_visualization_server_closed(self):
        """Handle root area visualization server closed event."""
        if self.root_area_viz:
            visualization_manager.close_root_area_visualization(self)

    # ==================== Image Loading ====================
    
    def load_images(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if dir_name:
            # Leave the Welcome screen for the working shell and show the
            # full-window loading overlay (PR4 T4.6).
            if getattr(self, "app_stack", None) is not None:
                self.app_stack.setCurrentIndex(1)
            if getattr(self, "loading_overlay", None) is not None:
                self.loading_overlay.start("Loading images")

            # Show progress before loading.
            self.task_progress.start("Loading images")

            # Clear undo/redo stacks if mask tracing interface exists
            if hasattr(self, "mask_tracing_interface"):
                self.mask_tracing_interface.undo_stack.clear()
                self.mask_tracing_interface.redo_stack.clear()
                self.mask_tracing_interface.last_point = None
                self.mask_tracing_interface.drawing = False
                if (
                    hasattr(self.mask_tracing_interface, "mask_pixmap")
                    and self.mask_tracing_interface.mask_pixmap
                ):
                    self.mask_tracing_interface.mask_pixmap.fill(
                        Qt.GlobalColor.transparent
                    )

            # Load images through image manager
            self.image_manager.load_images(dir_name)

    def update_loading_progress(self, value):
        """Update the loading progress bar"""
        self.task_progress.set_progress(value)
        if getattr(self, "loading_overlay", None) is not None:
            self.loading_overlay.set_progress(value)

    def on_loading_finished(self, images, fake_images, masks, has_fake_real_pairs):
        """Handle completion of image loading"""
        self.task_progress.finish(f"Loaded {len(images)} images")
        if getattr(self, "loading_overlay", None) is not None:
            self.loading_overlay.hide()
        self.populate_file_list()
        self.status_bar.showMessage(f"Loaded {len(images)} images", 5000)
        self.notify(f"Loaded {len(images)} images")

        # Swap the index-0 display page from the empty state to the real display
        # area now that images exist.
        if images and getattr(self, "display_page_stack", None) is not None:
            self.display_page_stack.setCurrentIndex(0)

    # ==================== Tree Item Selection ====================
    
    def mask_exists(self, image_name):
        image_path = self.image_manager.get_image_path(image_name)
        mask_dir = os.path.join(os.path.dirname(image_path), "mask")
        mask_path = os.path.join(mask_dir, os.path.basename(image_name))
        exists = os.path.exists(mask_path)
        return exists

    def _image_has_mask(self, stored_name):
        """Return True if the image has a saved mask.

        Checks both the on-disk ``mask`` directory and the image manager's
        in-memory state. Used to drive the green/white tree-item coloring so it
        can be re-evaluated (rather than blindly reset) when leaving the
        mask-tracing view (F-016).
        """
        image_path = self.image_manager.get_image_path(stored_name)
        if not image_path:
            return False
        mask_dir = os.path.join(os.path.dirname(image_path), "mask")
        mask_path = os.path.join(mask_dir, os.path.basename(image_path))
        return os.path.exists(mask_path) or self.image_manager.has_mask(stored_name)

    def on_tree_item_clicked(self, item, column):
        """Handle tree item clicks - expand/collapse folders, load images"""
        # Get the stored image name from the item's user data
        image_name = item.data(0, Qt.ItemDataRole.UserRole)

        # If item is not an image (it's a folder node), toggle expansion
        if image_name is None:
            item.setExpanded(not item.isExpanded())
            return

        # If it's an image, load it
        self.on_image_selected(item)

    def on_image_selected(self, item):
        """Load and display the selected image"""
        # Get the stored image name from the item's user data
        image_name = item.data(0, Qt.ItemDataRole.UserRole)

        # If item is not an image (it's a folder node), do nothing
        if image_name is None:
            return

        image_path = self.image_manager.get_image_path(image_name)
        if image_path is None:
            return

        if self.right_panel.currentWidget() == self.mask_tracing_interface:
            self.mask_tracing_interface.load_image(image_path)
        elif self.right_panel.currentWidget() == self.skeleton_correction_interface:
            self.skeleton_correction_interface.load_image(
                image_path, images_base_folder=self.image_manager.original_folder
            )
        else:
            self.display_controller.display_selected_image_by_name(image_name)

    # ==================== Calculation Methods ====================
    
    def calculate_root_length(self):
        self.skeleton_handler.calculate_root_length()

    def calculate_root_area(self):
        self.skeleton_handler.calculate_root_area()

    def update_display(self):
        self.display_controller.update_display()

    def load_results(self, results_dir):
        """
        Load results after skeleton generation.
        This method updates the skeleton information without reloading all images.
        """
        images_dir = os.path.join(results_dir, "images")
        if not os.path.isdir(images_dir):
            QMessageBox.critical(
                self, "Error", f"Images directory not found at: {images_dir}"
            )
            return

        # Update the skeleton paths
        if self.image_manager:
            self.image_manager._find_processed_base_path()
            # Update the view mode to trigger skeleton loading
            self.view_mode_combo.setCurrentText(self.view_mode_combo.currentText())

        self.status_bar.showMessage("Results loaded successfully.", 5000)
        self.notify("Results loaded successfully")

    # ==================== Mask Tracing ====================
    
    def toggle_mask_tracing(self):
        if self.right_panel.currentIndex() != 1:  # If not in mask tracing view
            self.switch_right_panel("mask_tracing")
            self.toggle_mask_tracing_button.setText("Return to Main View")

            # Connect signals for mask tracing interface.
            # Guarded so repeated enter->exit->enter cycles do not stack up
            # duplicate connections (F-015): each slot would otherwise fire
            # multiple times per signal because the exit branch removes only one
            # connection per disconnect call.
            if not self._mask_tracing_signals_connected:
                self.mask_tracing_interface.mask_saved.connect(self.on_mask_saved)
                self.mask_tracing_interface.mask_cleared.connect(self.on_mask_cleared)
                self.mask_tracing_interface.b_key_status_changed.connect(
                    self.on_b_key_status_changed
                )
                self._mask_tracing_signals_connected = True

            # Update mask status colors when entering mask tracing view
            def update_mask_colors(item):
                stored_name = item.data(0, Qt.ItemDataRole.UserRole)
                if stored_name:  # Only update leaf items (images)
                    if self._image_has_mask(stored_name):
                        item.setForeground(0, QColor("green"))
                    else:
                        item.setForeground(0, QColor("white"))

                # Recursively update children
                for i in range(item.childCount()):
                    update_mask_colors(item.child(i))

            # Update all top-level items
            for i in range(self.file_list.topLevelItemCount()):
                update_mask_colors(self.file_list.topLevelItem(i))

        else:  # If in mask tracing view, switch back to main view
            self.switch_right_panel("display")
            self.toggle_mask_tracing_button.setText("✏️ Toggle Mask Tracing")

            # Disconnect signals (mirrors the guarded connect above, F-015).
            if self._mask_tracing_signals_connected:
                self.mask_tracing_interface.mask_saved.disconnect(self.on_mask_saved)
                self.mask_tracing_interface.mask_cleared.disconnect(
                    self.on_mask_cleared
                )
                self.mask_tracing_interface.b_key_status_changed.disconnect(
                    self.on_b_key_status_changed
                )
                self._mask_tracing_signals_connected = False

            # Re-evaluate mask status colors when leaving mask tracing view.
            # F-016: do NOT blindly reset every item to white — that wiped the
            # green markers on legitimately saved masks. Items with a saved mask
            # stay green; only items without a mask are reset to white.
            def reset_colors(item):
                stored_name = item.data(0, Qt.ItemDataRole.UserRole)
                if stored_name:  # Only update leaf items (images)
                    if self._image_has_mask(stored_name):
                        item.setForeground(0, QColor("green"))
                    else:
                        item.setForeground(0, QColor("white"))

                # Recursively reset children
                for i in range(item.childCount()):
                    reset_colors(item.child(i))

            # Reset all top-level items
            for i in range(self.file_list.topLevelItemCount()):
                reset_colors(self.file_list.topLevelItem(i))

        # Load the current image into the mask tracing interface
        current_item = self.file_list.currentItem()
        if current_item:
            self.on_image_selected(current_item)

    def on_b_key_status_changed(self, is_pressed):
        """Observer method for B key status changes in mask tracing interface."""
        if is_pressed:
            self.status_bar.showMessage(
                "B key pressed - use mouse wheel to adjust brush size", 3000
            )
        else:
            self.status_bar.showMessage("B key released", 3000)

    def on_mask_saved(self, image_path):
        # Always emit the signal
        self.mask_saved.emit(image_path)

        # Only update the visual status if we're in mask tracing view
        if self.right_panel.currentWidget() == self.mask_tracing_interface:
            image_name = os.path.splitext(
                os.path.basename(os.path.normpath(image_path))
            )[0]
            item = self.find_tree_item_by_image_name(image_name)
            if item:
                item.setForeground(0, QColor("green"))
            self.status_bar.showMessage(f"Mask saved for {image_name}", 3000)
            self.notify(f"Mask saved for {image_name}")

    def on_mask_cleared(self, image_path):
        # Always emit the signal
        self.mask_cleared.emit(image_path)

        # Only update the visual status if we're in mask tracing view
        if self.right_panel.currentWidget() == self.mask_tracing_interface:
            image_name = os.path.splitext(
                os.path.basename(os.path.normpath(image_path))
            )[0]
            item = self.find_tree_item_by_image_name(image_name)
            if item:
                item.setForeground(0, QColor("white"))

    # ==================== Skeleton Correction (isolated stream) ====================
    
    def toggle_skeleton_correction(self):
        """
        Toggle the Skeleton Correction editor view.

        Important: This editor is intentionally isolated from the app's existing
        skeleton generation + main display streams.
        """
        # Right panel indexes:
        # 0 = display, 1 = mask tracing, 2 = visualization, 3 = skeleton correction
        if self.right_panel.currentIndex() != 3:
            # If leaving mask tracing, ensure its signals are disconnected cleanly
            if self.right_panel.currentIndex() == 1:
                self.toggle_mask_tracing()

            self.switch_right_panel("skeleton_correction")
            if hasattr(self, "toggle_skeleton_correction_button"):
                self.toggle_skeleton_correction_button.setText("Return to Main View")
        else:
            self.switch_right_panel("display")
            if hasattr(self, "toggle_skeleton_correction_button"):
                self.toggle_skeleton_correction_button.setText("✏️ Skeleton Correction")

        # Load the current image into the skeleton correction interface (if any)
        current_item = self.file_list.currentItem()
        if current_item:
            self.on_image_selected(current_item)

    # ==================== Panel Switching ====================
    
    def switch_right_panel(self, panel):
        if panel == "display":
            self.right_panel.setCurrentIndex(0)
        elif panel == "mask_tracing":
            self.right_panel.setCurrentIndex(1)
        elif panel == "visualization":
            self.right_panel.setCurrentIndex(2)
        elif panel == "skeleton_correction":
            self.right_panel.setCurrentIndex(3)
