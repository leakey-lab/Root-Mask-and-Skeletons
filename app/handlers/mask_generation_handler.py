from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import numpy as np
from app.mask_model.model import ResNetSkeleton
from resources.resource_utils import get_resource_path


class ImageDataset(data.Dataset):
    """Custom Dataset for loading images efficiently"""

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, filename


class MaskGenerationThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, input_dir, output_dir, model, device):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model = model
        self.device = device

    def run(self):
        try:
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

            # Create dataset and dataloader
            dataset = ImageDataset(self.input_dir, transform=transform)
            total_images = len(dataset)

            # Optimal batch size and num_workers for GPU utilization
            # Adjust batch_size based on GPU memory (8-16 is good for most GPUs)
            batch_size = 16
            num_workers = min(8, os.cpu_count() or 4)  # Use up to 8 CPU cores

            dataloader = data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if self.device.type == "cuda" else False,
                prefetch_factor=2 if num_workers > 0 else None,
                persistent_workers=True if num_workers > 0 else False,
            )

            processed_count = 0

            # Process images in batches
            with torch.no_grad():
                for batch_images, batch_filenames in dataloader:
                    try:
                        # Move batch to GPU
                        batch_images = batch_images.to(self.device)

                        # Generate masks for entire batch
                        batch_masks = self.model(batch_images)

                        # Process each mask in the batch
                        for mask, filename in zip(batch_masks, batch_filenames):
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

                            processed_count += 1

                            # Update progress after each image for better granularity
                            progress = int((processed_count / total_images) * 100)
                            self.progress.emit(progress)

                    except Exception:
                        continue

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
            weights_path = get_resource_path(
                os.path.join("checkpoints", "mask_weights", "best_mask_model_V5.pth")
            )

            if os.path.exists(weights_path):
                self.model = ResNetSkeleton(num_classes=1, pretrained=False)
                self.model.load_state_dict(
                    torch.load(weights_path, map_location=self.device)
                )
                self.model = self.model.to(self.device)
                self.model.eval()
            else:
                self.model = None
        except Exception:
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

            # Use the existing loading progress bar instead of creating a new one
            self.main_window.loading_progress_bar.setValue(0)
            self.main_window.loading_progress_bar.show()
            self.main_window.loading_progress_bar.setTextVisible(True)
            self.main_window.loading_progress_bar.setFormat("Generating masks: %p%")

            # Create and start generation thread
            self.generation_thread = MaskGenerationThread(
                input_dir, output_dir, self.model, self.device
            )
            self.generation_thread.progress.connect(self.update_progress)
            self.generation_thread.finished.connect(self.on_generation_finished)
            self.generation_thread.error.connect(self.on_generation_error)
            self.generation_thread.start()

            # Show status message
            self.main_window.status_bar.showMessage("Generating masks...")

        except Exception as e:
            QMessageBox.critical(
                self.main_window, "Error", f"Error starting mask generation: {str(e)}"
            )

    def update_progress(self, value):
        """Update the progress bar value"""
        self.main_window.loading_progress_bar.setValue(value)
        self.main_window.status_bar.showMessage(f"Generating masks... {value}%")

    def on_generation_finished(self, output_dir):
        """Handle completion of mask generation"""
        # Hide the progress bar
        self.main_window.loading_progress_bar.hide()

        self.main_window.status_bar.showMessage("Mask generation completed", 5000)
        QMessageBox.information(
            self.main_window,
            "Success",
            f"Masks generated successfully and saved to:\n{output_dir}",
        )

    def on_generation_error(self, error_message):
        """Handle error during mask generation"""
        # Hide the progress bar
        self.main_window.loading_progress_bar.hide()

        self.main_window.status_bar.showMessage("Error during mask generation", 5000)
        QMessageBox.critical(
            self.main_window, "Error", f"Error during mask generation: {error_message}"
        )
