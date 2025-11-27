# Building Executable for Root Mask and Skeletons Application

This guide will help you create a standalone executable (.exe) for your Root Mask and Skeletons application.

## Prerequisites

1. Make sure you're in your virtual environment:
   ```bash
   .\venv\Scripts\activate
   ```

2. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```

## Building the Executable

### Option 1: Using the Spec File (Recommended)

The `build_executable.spec` file is already configured with all necessary settings.

```bash
pyinstaller build_executable.spec
```

### Option 2: Using Command Line (Alternative)

If you want to customize further or rebuild from scratch:

```bash
pyinstaller --name="RootMaskAndSkeletons" ^
    --windowed ^
    --add-data "themes;themes" ^
    --add-data "checkpoints;checkpoints" ^
    --add-data "mask_model;mask_model" ^
    --add-data "normalization_defaults.json;." ^
    --hidden-import=torch ^
    --hidden-import=torchvision ^
    --hidden-import=cv2 ^
    --hidden-import=skimage ^
    --hidden-import=PyQt6 ^
    --hidden-import=pyqtgraph ^
    --hidden-import=plotly ^
    --hidden-import=dash ^
    main.py
```

## Output

After building, you'll find:
- **dist/RootMaskAndSkeletons/** - Folder containing your executable and all dependencies
- **build/** - Temporary build files (can be deleted)
- **RootMaskAndSkeletons.exe** - Your application executable (inside the dist folder)

## Running the Executable

Navigate to the `dist/RootMaskAndSkeletons/` folder and run:
```bash
RootMaskAndSkeletons.exe
```

## Distribution

To distribute your application:
1. Compress the entire `dist/RootMaskAndSkeletons/` folder into a ZIP file
2. Share the ZIP with users
3. Users can extract and run the .exe file directly without installing Python

## CUDA Build Information

**Your build includes CUDA 12.8 support for GPU acceleration!**

- Expected size: **2-3 GB** (due to CUDA libraries)
- End users **MUST** have NVIDIA GPU with updated drivers (525.60.13+)
- See `CUDA_DISTRIBUTION_NOTES.md` for complete requirements

### What This Means:
- ✅ **Fast**: 10-50x faster mask generation on GPU
- ⚠️ **Large**: ~2-3 GB executable size
- ⚠️ **Requirements**: NVIDIA GPU + drivers required

## Troubleshooting

### Issue: CUDA-related build errors
**Solution**: 
- Ensure PyTorch with CUDA is installed: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Issue: Missing DLL errors
**Solution**: Make sure all required model files are in the `checkpoints` folder

### Issue: "Failed to execute script" error
**Solution**: Run with console enabled to see errors:
- Edit `build_executable.spec`
- Change `console=False` to `console=True`
- Rebuild with `pyinstaller build_executable.spec`

### Issue: Application crashes on startup
**Solution**: Check if all data files are included:
- themes/dark_theme.qss
- checkpoints/mask_weights/best_mask_model_V5.pth
- checkpoints/skeletonizer/latest_net_G.pth
- normalization_defaults.json

### Issue: Large executable size (2-3 GB)
**Solution**: This is normal for CUDA-enabled PyTorch applications. Size breakdown:
- PyTorch CUDA runtime: ~1.5 GB
- CUDA libraries: ~500 MB
- Application + models: ~500 MB

To reduce size (if users don't need GPU):
- Build with CPU-only PyTorch instead
- Install: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- Rebuild (will be ~800 MB instead)

## Advanced Options

### Creating a Single File Executable

If you prefer a single .exe file instead of a folder:

```bash
pyinstaller --onefile build_executable.spec
```

**Note**: This creates a single file but has slower startup time as it extracts files to temp folder on each run.

### Adding an Application Icon

1. Get an .ico file for your application
2. Edit `build_executable.spec`
3. Change `icon=None` to `icon='path/to/your/icon.ico'`
4. Rebuild

## Build Times

- First build: 5-15 minutes (depending on your system)
- Subsequent builds: 2-5 minutes

## Testing

Before distributing:
1. Test the executable on your machine
2. Test on a clean Windows machine without Python installed
3. Verify all features work correctly
4. Check that model inference works properly

## Notes

- The executable is platform-specific (Windows .exe works only on Windows)
- For macOS/Linux, build on respective platforms
- PyTorch adds significant size (~500MB+) to the executable
- Keep the original Python source code for updates and maintenance
