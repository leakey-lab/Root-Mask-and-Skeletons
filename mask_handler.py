from PyQt6.QtWidgets import QMessageBox
from mask_tracing_interface import MaskTracingInterface


class MaskHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        self.mask_tracing_interface = MaskTracingInterface()

    def show_mask_tracing_interface(self):
        self.main_window.right_panel.setCurrentWidget(self.mask_tracing_interface)

    def hide_mask_tracing_interface(self):
        self.main_window.right_panel.setCurrentWidget(self.main_window.display_area)

    def toggle_mask_tracing_interface(self):
        if self.main_window.right_panel.currentWidget() == self.mask_tracing_interface:
            self.hide_mask_tracing_interface()
        else:
            self.show_mask_tracing_interface()

    def load_image_for_tracing(self, image_path):
        self.mask_tracing_interface.load_image(image_path)

    def save_traced_mask(self, save_path):
        self.mask_tracing_interface.save_mask(save_path)
