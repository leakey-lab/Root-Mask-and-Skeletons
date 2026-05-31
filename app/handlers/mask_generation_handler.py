import logging
import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

from app.config import MASK_BATCH_SIZE, MASK_THRESHOLD, MASK_WEIGHTS_PATH
from app.inference import runtime
from app.mask_model.model import ResNetSkeleton

logger = logging.getLogger(__name__)


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
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            logger.warning("Skipping unreadable image %s: %s", image_path, exc)
            # Return a blank RGB image so the DataLoader batch dimension stays consistent.
            image = Image.new("RGB", (640, 480), color=0)

        if self.transform:
            image = self.transform(image)

        return image, filename


class MaskGenerationThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, list)
    error = pyqtSignal(str)

    def __init__(self, input_dir, output_dir, model, device):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model = model
        self.device = device
        self.rt = runtime.get_runtime()

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

            # Create dataset and dataloader.
            # num_workers > 0 deadlocks inside a QThread on Windows (F-022); use 0.
            dataset = ImageDataset(self.input_dir, transform=transform)
            total_images = len(dataset)

            dataloader = data.DataLoader(
                dataset,
                batch_size=MASK_BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=(self.device.type == "cuda"),
            )

            failed_images: list[str] = []
            processed_count = 0
            last_emitted_pct = -1

            # Ensure the model is in eval mode for inference (F-022 / ML correctness).
            self.model.eval()

            # Use inference_mode for maximum performance — no grad tape overhead (F-022).
            with torch.inference_mode(), self.rt.autocast():
                for batch_images, batch_filenames in dataloader:
                    try:
                        # Move batch to device
                        batch_images = batch_images.to(self.device)
                        if self.rt.use_channels_last and self.device.type == "cuda":
                            batch_images = batch_images.contiguous(
                                memory_format=torch.channels_last
                            )

                        # Generate masks for entire batch
                        batch_masks = self.model(batch_images)

                        # Process each mask in the batch
                        for mask, filename in zip(batch_masks, batch_filenames):
                            # Convert to numpy and threshold using config constant (F-005 / config).
                            mask_np = mask.squeeze().cpu().numpy()
                            binary_mask = (mask_np > MASK_THRESHOLD).astype(np.uint8) * 255

                            # Convert to PIL Image and save
                            mask_pil = Image.fromarray(binary_mask, mode="L")
                            output_path = os.path.join(
                                self.output_dir, os.path.splitext(filename)[0] + ".png"
                            )
                            try:
                                mask_pil.save(output_path, "PNG")
                            except OSError as save_exc:
                                logger.error(
                                    "Failed to save mask for %s: %s", filename, save_exc
                                )
                                failed_images.append(filename)
                                continue

                            processed_count += 1
                            progress = int((processed_count / total_images) * 100)
                            # Emit only when the integer percent changes to avoid flooding
                            # the Qt event loop with duplicate cross-thread signals (perf).
                            if progress != last_emitted_pct:
                                last_emitted_pct = progress
                                self.progress.emit(progress)

                    except (RuntimeError, torch.cuda.OutOfMemoryError) as batch_exc:
                        # Narrow per-batch except: log the failure, collect names, continue
                        # with the remaining batches rather than silently swallowing (F-021).
                        names = list(batch_filenames)
                        logger.error(
                            "Batch processing failed for images %s: %s", names, batch_exc
                        )
                        failed_images.extend(names)
                        # Free GPU memory before next batch when on CUDA (F-022).
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue

            if failed_images:
                logger.warning(
                    "Mask generation completed with %d failed image(s): %s",
                    len(failed_images),
                    failed_images,
                )
            # Emit the failure list as a structured payload so filenames containing
            # commas or special tokens cannot corrupt the decode (correctness).
            self.finished.emit(self.output_dir, failed_images)

        except Exception as exc:  # noqa: BLE001 — thread boundary; must not propagate
            logger.exception("Mask generation thread encountered an unexpected error")
            self.error.emit(str(exc))


class MaskGenerationHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generation_thread = None

        # Initialize the model
        self._initialize_model()

    def _initialize_model(self):
        # Use the centralised config path — no CWD dependency (F-005 / config adoption).
        weights_path = MASK_WEIGHTS_PATH

        if not os.path.exists(weights_path):
            logger.warning(
                "Mask weights not found at %s; mask generation will be unavailable.",
                weights_path,
            )
            self.model = None
            return

        try:
            self.model = ResNetSkeleton(num_classes=1, pretrained=False)
            # weights_only=True prevents arbitrary code execution via pickle (F-023 / SS-01).
            state_dict = torch.load(
                weights_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict)
            # Move the model to its device once at load time; never reload per call (F-022).
            self.model = self.model.to(self.device)
            self.model.eval()
            rt = runtime.get_runtime()
            if rt.use_channels_last and self.device.type == "cuda":
                self.model = self.model.to(memory_format=torch.channels_last)
            logger.info(
                "Mask model loaded from %s on device %s", weights_path, self.device
            )
        except (RuntimeError, OSError, ValueError) as exc:
            # Log the specific exception so the init failure is visible (F-102).
            logger.exception(
                "Failed to load mask model weights from %s: %s", weights_path, exc
            )
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

        # Guard against re-entry: a second click would orphan the running QThread and
        # interleave writes to the same mask dir (bug).
        if self.generation_thread is not None and self.generation_thread.isRunning():
            self.main_window.status_bar.showMessage(
                "Mask generation already in progress...", 5000
            )
            return

        try:
            # Prefer image_manager.original_folder over inferring from the first image path
            # (mitigates F-068: mask input dir should not depend on iteration order).
            if getattr(self.main_window.image_manager, "original_folder", None):
                input_dir = self.main_window.image_manager.original_folder
            else:
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

        except (OSError, StopIteration) as exc:
            logger.exception("Error starting mask generation")
            QMessageBox.critical(
                self.main_window, "Error", f"Error starting mask generation: {exc}"
            )

    def update_progress(self, value):
        """Update the progress bar value"""
        self.main_window.loading_progress_bar.setValue(value)
        self.main_window.status_bar.showMessage(f"Generating masks... {value}%")

    def on_generation_finished(self, output_dir, failed):
        """Handle completion of mask generation"""
        self.main_window.loading_progress_bar.hide()

        # The thread now delivers the failure list as a structured argument (F-021).
        if failed:
            self.main_window.status_bar.showMessage(
                f"Mask generation completed with {len(failed)} failure(s)", 5000
            )
            QMessageBox.warning(
                self.main_window,
                "Partial Success",
                f"Masks saved to:\n{output_dir}\n\n"
                f"{len(failed)} image(s) could not be processed:\n"
                + "\n".join(failed[:20])
                + ("\n…" if len(failed) > 20 else ""),
            )
        else:
            self.main_window.status_bar.showMessage("Mask generation completed", 5000)
            QMessageBox.information(
                self.main_window,
                "Success",
                f"Masks generated successfully and saved to:\n{output_dir}",
            )

    def on_generation_error(self, error_message):
        """Handle error during mask generation"""
        self.main_window.loading_progress_bar.hide()

        logger.error("Mask generation error: %s", error_message)
        self.main_window.status_bar.showMessage("Error during mask generation", 5000)
        QMessageBox.critical(
            self.main_window, "Error", f"Error during mask generation: {error_message}"
        )
