import torch
from PIL import Image
import os
from tools import get_shuffled_file_paths, mask_image, compose_side_by_side
from diffusers import StableDiffusionInpaintPipeline

# Load the Stable Diffusion Inpainting model
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float32
)
pipe.to("cpu")

# Set your dataset directory and output directory
directory = "C:/Users/kaege/OneDrive/Desktop/SMU/Year 3/Project/similar/inpainting"  # Input images
output_dir = "C:/Users/kaege/DeepfakeDetectionInArt/stable_diffusionV2/generator/output"  # Output images
list_files = get_shuffled_file_paths(directory)

count = 20000
print(list_files[0])

# Process each image in the dataset
for image_address in list_files:
    print(image_address)
    if not image_address.lower().endswith(".png"):
        continue

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
    if count > 20000 + 1500:  # Limit the total processed images
        break

