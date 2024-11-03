import torch
from PIL import Image
import os
from tools import get_file_paths, mask_image, compose_side_by_side, load_file_paths
from diffusers import StableDiffusionInpaintPipeline

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Stable Diffusion Inpainting model and move it to GPU
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float32
)
pipe.to(device)

# Set your dataset directory and output directory
directory = "/common/home/users/b/bryanchua.2022/scratchDirectory/sendgpu/inpainting"  # Input images
output_dir = "/common/home/users/b/bryanchua.2022/scratchDirectory/sendgpu/stable_diffusionV2/generator/output"  # Output images
# loads file paths into a list and writes to a txt file
if os.path.exists("/common/home/users/b/bryanchua.2022/scratchDirectory/sendgpu/stable_diffusionV2/generator/files.txt"):
    # files txt file exists, initalize list using files txt
    list_files = load_file_paths("/common/home/users/b/bryanchua.2022/scratchDirectory/sendgpu/stable_diffusionV2/generator/files.txt")
else:
    list_files = get_file_paths(directory)
print(list_files)
count = 1

# Process each image in the dataset
for image_address in list_files:
    print(image_address)
    # Open the image
    image = Image.open(image_address).convert("RGB")
    width, height = image.size
    name = os.path.join(output_dir, str(count))  # Create output folder for each image
    os.makedirs(name, exist_ok=True)

    # Save the original image only once per count
    image.resize((width, height)).save(os.path.join(name, "original.png"))

    # Apply different masking techniques
    mask_types = ['random_patch', 'upper_white', 'upper_black', 'left_white', 'left_black']
    for mask_type in mask_types:
        output_path = os.path.join(name, f"mask_{mask_type}.png")

        # Apply the corresponding mask type using `mask_image`
        masked_image = mask_image(image_address, mask_type, output_path).convert("RGB")

        # Inpainting prompt
        prompt = "generate a painting compatible with the rest of the image"

        # Perform inpainting using the masked image and original image
        image_inpainting = pipe(prompt=prompt, image=image.resize((512, 512)), mask_image=masked_image).images[0]

        # Save the inpainting result and the group image
        image_inpainting.resize((width, height)).save(os.path.join(name, f"inpainting_{mask_type}.png"))
        compose_side_by_side(image_inpainting.resize((width, height)), image.resize((width, height)),
                             os.path.join(name, f"group_{mask_type}.png"))

    count += 1
