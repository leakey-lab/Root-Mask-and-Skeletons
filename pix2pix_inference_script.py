import os
from options.test_options import TestOptions
from models import create_model
from data import create_dataset
from util.visualizer import save_images
from util import html


def main():
    print("Debug: Starting pix2pix inference script")

    # Set up options
    opt = TestOptions().parse()
    print(f"Debug: Options parsed: {opt}")

    # Hard-code some options
    opt.name = "skeletonizer"
    opt.model = "test"
    opt.netG = "enhanced_resnet"
    opt.direction = "AtoB"
    opt.dataset_mode = "single"
    opt.norm = "batch"
    opt.num_threads = 20
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.results_dir = f"{opt.dataroot}/output"
    opt.aspect_ratio = 3 / 4
    # opt.aspect_ratio = 0.94
    opt.load_size = 256
    # opt.load_size = 255
    # opt.crop_size = 255

    print(f"Debug: Dataroot: {opt.dataroot}")
    print(f"Debug: Results directory: {opt.results_dir}")

    # Create dataset
    try:
        dataset = create_dataset(opt)
        total_images = len(dataset)
        print(f"Debug: Dataset created with {total_images} images")
    except Exception as e:
        print(f"Debug: Error creating dataset: {str(e)}")
        raise

    # Create model
    try:
        model = create_model(opt)
        model.setup(opt)
        print("Debug: Model created and set up")
    except Exception as e:
        print(f"Debug: Error creating model: {str(e)}")
        raise

    # Create website for results
    web_dir = os.path.join(opt.results_dir, opt.name, f"{opt.phase}_{opt.epoch}")
    webpage = html.HTML(
        web_dir, f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}"
    )
    print(f"Debug: Web directory created at {web_dir}")

    # Test
    for i, data in enumerate(dataset):
        print(f"Processing image {i+1}/{total_images}")
        try:
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            save_images(
                webpage,
                visuals,
                img_path,
                aspect_ratio=opt.aspect_ratio,
                width=opt.display_winsize,
            )

            # Calculate and print progress
            progress = int((i + 1) / total_images * 100)
            print(f"Progress: {progress}%")
        except Exception as e:
            print(f"Debug: Error processing image {i+1}: {str(e)}")
            raise

    webpage.save()
    print("Debug: HTML saved, script completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Debug: Unhandled exception in main: {str(e)}")
        raise
