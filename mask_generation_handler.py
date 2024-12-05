from PyQt6.QtWidgets import QMessageBox, QProgressBar
from PyQt6.QtCore import QThread, pyqtSignal
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from mask_model.model import ResNetSkeleton


class MaskGenerationThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, input_dir, output_dir, model):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model = model

    def run(self):
        try:
            # Get list of image files
            image_files = [
                f
                for f in os.listdir(self.input_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            total_images = len(image_files)

            # Set up image transforms
            transform = transforms.Compose(
                [
                    transforms.Resize((480, 640)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5514, 0.4094, 0.3140], std=[0.1299, 0.1085, 0.0914]
                    ),
                ]
            )

            # Process each image
            for i, filename in enumerate(image_files):
                try:
                    # Load and transform image
                    input_path = os.path.join(self.input_dir, filename)
                    image = Image.open(input_path).convert("RGB")
                    image_tensor = (
                        transform(image)
                        .unsqueeze(0)
                        .to(next(self.model.parameters()).device)
                    )

                    # Generate mask
                    with torch.no_grad():
                        mask = self.model(image_tensor)

                    # Convert to numpy and threshold
                    mask_np = mask.squeeze().cpu().numpy()

                    # Properly binarize the mask using a threshold
                    binary_mask = (mask_np > 0.5).astype(np.uint8) * 255

                    # Convert to PIL Image
                    mask_pil = Image.fromarray(binary_mask, mode="L")

                    # Save binary mask
                    output_path = os.path.join(
                        self.output_dir, os.path.splitext(filename)[0] + ".png"
                    )
                    mask_pil.save(output_path, "PNG")

                    # Update progress
                    progress = int((i + 1) / total_images * 100)
                    self.progress.emit(progress)

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

            self.finished.emit(self.output_dir)

        except Exception as e:
            self.error.emit(str(e))


class MaskGenerationHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generation_thread = None

        # Initialize the model
        self._initialize_model()

    def _initialize_model(self):
        try:
            weights_path = os.path.join(
                os.path.dirname(__file__),
                "checkpoints",
                "mask_weights",
                "best_mask_model_V5.pth",
            )

            if os.path.exists(weights_path):
                self.model = ResNetSkeleton(num_classes=1, pretrained=False)
                self.model.load_state_dict(
                    torch.load(weights_path, map_location=self.device)
                )
                self.model = self.model.to(self.device)
                self.model.eval()
                print("Mask generation model initialized successfully")
            else:
                print(f"Model weights not found at {weights_path}")
                self.model = None
        except Exception as e:
            print(f"Failed to initialize mask generation model: {str(e)}")
            self.model = None

    def generate_masks(self):
        """Generate masks using the ResNet model"""
        if self.model is None:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "Mask generation model not initialized. Please ensure model weights are present.",
            )
            return

        if not self.main_window.image_manager.images:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "No images loaded. Please load images first.",
            )
            return

        try:
            # Get input directory and create output directory
            first_image_path = next(
                iter(self.main_window.image_manager.images.values())
            )
            input_dir = os.path.dirname(first_image_path)
            output_dir = os.path.join(input_dir, "mask")
            os.makedirs(output_dir, exist_ok=True)

            # Create and start generation thread
            self.generation_thread = MaskGenerationThread(
                input_dir, output_dir, self.model
            )
            self.generation_thread.progress.connect(self.update_progress)
            self.generation_thread.finished.connect(self.on_generation_finished)
            self.generation_thread.error.connect(self.on_generation_error)
            self.generation_thread.start()

            # Show progress bar
            self.main_window.status_bar.showMessage("Generating masks...")

        except Exception as e:
            QMessageBox.critical(
                self.main_window, "Error", f"Error starting mask generation: {str(e)}"
            )

    def update_progress(self, value):
        self.main_window.status_bar.showMessage(f"Generating masks... {value}%")

    def on_generation_finished(self, output_dir):
        self.main_window.status_bar.showMessage("Mask generation completed", 5000)
        QMessageBox.information(
            self.main_window,
            "Success",
            f"Masks generated successfully and saved to:\n{output_dir}",
        )

    def on_generation_error(self, error_message):
        self.main_window.status_bar.showMessage("Error during mask generation", 5000)
        QMessageBox.critical(
            self.main_window, "Error", f"Error during mask generation: {error_message}"
        )
