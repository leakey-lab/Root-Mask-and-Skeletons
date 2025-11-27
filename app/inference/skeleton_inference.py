"""
Consolidated Pix2Pix Skeleton Inference Script

This script contains all necessary components for running skeleton inference:
- EnhancedResnetGenerator network architecture
- Image loading and preprocessing
- Model weight loading
- Inference execution and image saving

All constants are hardcoded for the skeletonizer model.
"""

import os
import functools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# =============================================================================
# CONSTANTS - Hardcoded configuration for skeletonizer model
# =============================================================================

CHECKPOINTS_DIR = "./checkpoints"
MODEL_NAME = "skeletonizer"
NORM = "batch"
INPUT_NC = 3
OUTPUT_NC = 3
NGF = 64
LOAD_SIZE = 256
CROP_SIZE = 256
ASPECT_RATIO = 3 / 4
USE_DROPOUT = True  # no_dropout is False by default
PREPROCESS = "resize_and_crop"
EPOCH = "latest"

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG",
    ".png", ".PNG", ".ppm", ".PPM",
    ".bmp", ".BMP", ".tif", ".TIF",
    ".tiff", ".TIFF",
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_image_file(filename):
    """Check if a file is an image based on extension."""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(directory):
    """Find all image files in a directory."""
    images = []
    assert os.path.isdir(directory), f"{directory} is not a valid directory"
    for fname in sorted(os.listdir(directory)):
        if is_image_file(fname):
            path = os.path.join(directory, fname)
            images.append(path)
    return images


class ImageDataset(Dataset):
    """
    Dataset class for batch loading images during inference.
    
    Args:
        image_paths: List of image file paths
        transform: Transform to apply to each image
        grayscale: Whether to convert images to grayscale
    """
    
    def __init__(self, image_paths, transform, grayscale=False):
        self.image_paths = image_paths
        self.transform = transform
        self.grayscale = grayscale
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = img
        
        # Return tensor and path for saving later
        return img_tensor, img_path


def tensor2im(input_image, imtype=np.uint8):
    """Convert a Tensor array into a numpy image array."""
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to disk."""
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def get_transform(grayscale=False):
    """Create image transformation pipeline."""
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    
    # Resize
    osize = [LOAD_SIZE, LOAD_SIZE]
    transform_list.append(transforms.Resize(osize, transforms.InterpolationMode.BICUBIC))
    
    # Convert to tensor and normalize
    transform_list.append(transforms.ToTensor())
    if grayscale:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    return transforms.Compose(transform_list)


# =============================================================================
# NETWORK ARCHITECTURE
# =============================================================================

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer."""
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError(f"normalization layer [{norm_type}] is not found")
    return norm_layer


def move_net_to_device(net, gpu_ids=[]):
    """Move network to GPU without initializing weights (for loading from checkpoint)."""
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    return net


class ResnetBlock(nn.Module):
    """ResNet block with skip connections."""
    
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer."""
    
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DilatedConvBlock(nn.Module):
    """Dilated convolution block."""
    
    def __init__(self, in_channels, out_channels, norm_layer, stride=1):
        super(DilatedConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, dilation=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=2, dilation=2, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=4, dilation=4, bias=False)
        self.bn3 = norm_layer(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class EnhancedResnetGenerator(nn.Module):
    """Enhanced ResNet-based generator with dilated convolutions and SE attention."""
    
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, n_blocks=9, padding_type="reflect", use_attention=True):
        super(EnhancedResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        # Downsampling with dilated convolutions
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [DilatedConvBlock(ngf * mult, ngf * mult * 2, norm_layer, stride=2)]

        # ResNet blocks with SE attention
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                           use_dropout=use_dropout, use_bias=use_bias)
            ]
            if use_attention and i == n_blocks // 2:
                model += [SELayer(ngf * mult)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        # Final convolution
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# =============================================================================
# INFERENCE MODEL
# =============================================================================

class SkeletonModel:
    """Skeleton inference model wrapper."""
    
    def __init__(self, gpu_ids=[]):
        self.gpu_ids = gpu_ids
        self.device = torch.device(f'cuda:{gpu_ids[0]}') if gpu_ids else torch.device('cpu')
        
        # Create generator network
        norm_layer = get_norm_layer(norm_type=NORM)
        self.netG = EnhancedResnetGenerator(
            INPUT_NC, OUTPUT_NC, NGF, norm_layer=norm_layer,
            use_dropout=USE_DROPOUT, n_blocks=9
        )
        # Move to GPU without initializing weights (weights will be loaded from checkpoint)
        self.netG = move_net_to_device(self.netG, gpu_ids)
        
        # Load pretrained weights from checkpoint
        self._load_network()
        
        # Set to eval mode
        self.netG.eval()
    
    def _load_network(self):
        """Load pretrained weights."""
        load_filename = f"{EPOCH}_net_G.pth"
        load_path = os.path.join(CHECKPOINTS_DIR, MODEL_NAME, load_filename)
        
        net = self.netG
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        
        print(f"loading the model from {load_path}")
        state_dict = torch.load(load_path, map_location=str(self.device))
        
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        
        # Patch InstanceNorm checkpoints
        for key in list(state_dict.keys()):
            self._patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        
        net.load_state_dict(state_dict)
    
    def _patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility."""
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self._patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
    
    def run(self, input_tensor):
        """Run inference on input tensor."""
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.netG(input_tensor)
        return output
    
    def run_batch(self, input_batch):
        """
        Run inference on a batch of input tensors.
        
        Args:
            input_batch: Batch of input tensors [batch_size, channels, height, width]
            
        Returns:
            Batch of output tensors
        """
        input_batch = input_batch.to(self.device)
        with torch.no_grad():
            output_batch = self.netG(input_batch)
        return output_batch


# =============================================================================
# MAIN INFERENCE FUNCTION
# =============================================================================

def run_inference(input_dir: str, output_dir: str, progress_callback=None, batch_size=8):
    """
    Run skeleton inference on all images in input_dir using batch processing.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        progress_callback: Optional callback function(current, total) for progress updates
        batch_size: Number of images to process in each batch (default: 4)
    
    Returns:
        Path to the results directory
    """
    
    # Setup GPU
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids = [0]
        print("Debug: Using CUDA")
    else:
        print("Debug: Using CPU")
    
    # Create output directory structure - save directly to skeletons folder
    images_dir = output_dir
    os.makedirs(images_dir, exist_ok=True)
    
    # Load model
    model = SkeletonModel(gpu_ids=gpu_ids)
    
    # Get image paths
    image_paths = make_dataset(input_dir)
    total_images = len(image_paths)
    print(f"Debug: Found {total_images} images")
    
    if total_images == 0:
        print("Debug: No images found in input directory")
        return images_dir
    
    # Create transform
    transform = get_transform(grayscale=(INPUT_NC == 1))
    
    # Create dataset and dataloader
    dataset = ImageDataset(image_paths, transform, grayscale=(INPUT_NC == 1))
    
    # Use num_workers=2 for parallel data loading, pin_memory for faster GPU transfer
    num_workers = 8 if len(image_paths) > batch_size else 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Process batches
    processed_count = 0
    for batch_idx, (input_batch, img_paths) in enumerate(dataloader):
        current_batch_size = input_batch.size(0)
        # Run inference on batch
        output_batch = model.run_batch(input_batch)
        
        # Process each image in the batch
        for i in range(current_batch_size):
            output_tensor = output_batch[i:i+1]
            img_path = img_paths[i]
            
            # Convert to numpy and save
            output_numpy = tensor2im(output_tensor)
            
            # Get output filename
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(images_dir, f"{img_name}_fake.png")
            
            save_image(output_numpy, output_path, aspect_ratio=ASPECT_RATIO)
            
            processed_count += 1
            
            # Progress update after each image for smoother UI updates
            if progress_callback:
                progress_callback(processed_count, total_images)
    
    return images_dir