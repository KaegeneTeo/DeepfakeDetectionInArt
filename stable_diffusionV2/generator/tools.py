from PIL import Image, ImageDraw
from PIL import ImageFont
import random
import os

def get_file_paths(directory):
    file_paths = []
    with open("/common/home/users/b/bryanchua.2022/scratchDirectory/sendgpu/stable_diffusionV2/generator/files.txt", 'a') as f:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower() == "original.png":
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path):
                        file_paths.append(file_path)
                        f.write(file_path + '\n')  # Write the path to the file

    return file_paths


def load_file_paths(input_file):
    if os.path.exists(input_file):
        with open(input_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return []


def mask_image(image_path, mask_type, output_path="/common/home/users/b/bryanchua.2022/scratchDirectory/sendgpu/stable_diffusionV2/generator/output"):
    # Open the image
    image = Image.open(image_path)
    width, height = image.size

    # Create a new image for the mask
    new_image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(new_image)

    # Apply the specified masking type
    if mask_type == "random_patch":
        percentage = random.randint(40, 60)
        new_image = Image.new("RGB", (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(new_image)
        total_white_area = int((width * height) * (percentage / 100))
        white_area = 0
        while white_area < total_white_area:
            # Generate random position and size for the white patch
            x, y = random.randint(0, width - 1), random.randint(0, height - 1)
            w, h = random.randint(width // 16, width // 4 + 1), random.randint(height // 16, height // 4 + 1)
            # Draw the white patch
            draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))
            # Update the current area of white patches
            white_area = sum([1 for pixel in new_image.getdata() if pixel == (255, 255, 255)])
    elif mask_type == "top_right_white":
        draw.polygon([(0, 0), (width, 0), (width, height)], fill=(255, 255, 255))  # Upper half white
        draw.polygon([(0, 0), (0, height), (width, height)], fill=(0, 0, 0))  # Lower half black
    elif mask_type == "top_right_black":
        draw.polygon([(0, 0), (width, 0), (width, height)], fill=(0, 0, 0))  # Upper half black
        draw.polygon([(0, 0), (0, height), (width, height)], fill=(255, 255, 255))  # Lower half white
    elif mask_type == "top_left_white":
        draw.polygon([(0, 0), (width, 0), (0, height)], fill=(0, 0, 0))  # Left half white
        draw.polygon([(0, height), (width, height), (width, 0)], fill=(255, 255, 255))  # Right half black
    elif mask_type == "top_left_black":
        draw.polygon([(0, 0), (width, 0), (0, height)], fill=(255, 255, 255))  # Left half black
        draw.polygon([(0, height), (width, height), (width, 0)], fill=(0, 0, 0))  # Right half white

    # Save the masked image
    new_image.save(output_path)
    return new_image

def compose_side_by_side(left_image, right_image, output_path, margin=10):
    left_label = "inpainting"
    right_label = "original"
    width, height = left_image.size
    new_width = width * 2 + margin
    hiegt_increase = 40
    new_height = height + hiegt_increase
    font = ImageFont.truetype("/common/home/users/b/bryanchua.2022/scratchDirectory/sendgpu/stable_diffusionV2/generator/ARIAL.TTF", size=30)

    # Create a new image with the combined dimensions
    combined_image = Image.new("RGB", (new_width, new_height), color=(255, 255, 255))

    # Paste the two input images side by side with a margin
    combined_image.paste(left_image, (0, 0))
    combined_image.paste(right_image, (width + margin, 0))

    # Add text labels below the images
    draw = ImageDraw.Draw(combined_image)
    left_text_width, _ = draw.textbbox((0, 0), left_label, font=font)[2:]
    right_text_width, _ = draw.textbbox((0, 0), right_label, font=font)[2:]
    draw.text(((width - left_text_width) // 2, height), left_label, font=font, fill=(0, 0, 0))
    draw.text((width // 2 + (width * 2 + margin - right_text_width) // 2, height), right_label, font=font, fill=(0, 0, 0))

    # Save the composite image
    combined_image.save(output_path)

    # Return the PIL image object
    return combined_image

