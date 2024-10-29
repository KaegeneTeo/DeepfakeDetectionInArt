from PIL import Image, ImageDraw
from PIL import ImageFont
import random
import os

def get_shuffled_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                file_paths.append(file_path)
    # Shuffle the list of file paths
    random.shuffle(file_paths)
    return file_paths

def mask_image(image_path, mask_type, output_path="C:/Users/kaege/DeepfakeDetectionInArt/gan/generator/tmp2"):
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
    elif mask_type == "upper_white":
        draw.rectangle([(0, 0), (width, height // 2)], fill=(255, 255, 255))  # Upper half white
        draw.rectangle([(0, height // 2), (width, height)], fill=(0, 0, 0))  # Lower half black
    elif mask_type == "upper_black":
        draw.rectangle([(0, 0), (width, height // 2)], fill=(0, 0, 0))  # Upper half black
        draw.rectangle([(0, height // 2), (width, height)], fill=(255, 255, 255))  # Lower half white
    elif mask_type == "left_white":
        draw.rectangle([(0, 0), (width // 2, height)], fill=(255, 255, 255))  # Left half white
        draw.rectangle([(width // 2, 0), (width, height)], fill=(0, 0, 0))  # Right half black
    elif mask_type == "left_black":
        draw.rectangle([(0, 0), (width // 2, height)], fill=(0, 0, 0))  # Left half black
        draw.rectangle([(width // 2, 0), (width, height)], fill=(255, 255, 255))  # Right half white

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
    font = ImageFont.truetype("arial.ttf", size=30)

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

