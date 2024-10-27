# IS424G2T32024

# Deepfake Detection Using CNN, SVM, and GAN

## Project Overview
This project aims to develop a robust system for detecting deepfake images of the inpainted class by combining three powerful machine learning models: Convolutional Neural Network (CNN), Support Vector Machine (SVM), and Generative Adversarial Network (GAN). The goal is to create a multi-model approach that leverages the strengths of each technique to improve the accuracy and reliability of deepfake detection.

**Team Members**: Ying Xuan, Bryan, Zhi You, Zhi Xuan, Cedric, Kaegene, Hao Tian, Hong Teng

### Why Deepfake Detection?
With the rapid advancement of AI, deepfake technology has made it possible to generate realistic fake images and videos, which can be used for malicious purposes such as misinformation, identity theft, and more. This project seeks to address these challenges by developing a solution capable of identifying altered images and preventing the spread of misinformation for inpainted artworks.

## Project Architecture

The solution consists of three main components:

1. **Convolutional Neural Network (CNN)**:
   - A CNN is used for feature extraction from input images.
   - It captures spatial hierarchies in the image data, which aids in distinguishing real images from altered ones.
   - The extracted features are then used as input for further classification.

2. **Support Vector Machine (SVM)**:
   - The features extracted from the CNN are fed into an SVM for binary classification (real vs. fake).
   - SVM is effective in handling high-dimensional data, making it suitable for this use case.

3. **Generative Adversarial Network (GAN)**:
   - A GAN is employed to generate synthetic data for training and simulate realistic altered images.
   - **GAN Components**:
     - **Generator**:
       - Utilizes [Stable Diffusion 2](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) to create synthetic images.
     - **Discriminator**:
       - Uses a CNN for feature extraction from both real and generated images.
       - The extracted features are then classified using SVM to determine if the images are real or fake.

## Workflow

### 1. Data Collection
The dataset for this project was sourced from [Kaggle Deepfake Challenge](https://www.kaggle.com/datasets/danielmao2019/deepfakeart?resource=download). The original images from this dataset were used as the foundation for generating altered images. To create the inpainting category, we employed the [Stable Diffusion 2](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) model, our GAN generator, to produce synthetic images.

Each record in the inpainting category consists of three images:

- **Source Image**: The original image taken from the WikiArt dataset.
- **Inpainting Image**: The image generated by the Stable Diffusion 2 model, which fills in masked areas of the source image.
- **Masking Image**: A black-and-white image indicating the areas of the original image that were inpainted (white areas represent masked regions).

#### Inpainting Process
The prompt used for generating the inpainting image is:  
*"Generate a painting compatible with the rest of the image."*

The dataset comprises over **5,063 records**, with original images masked between **40%-60%**. We applied one of the following masking schemas randomly:

- **Side Masking**: Masking the top, bottom, left, or right sides of the source image.
- **Diagonal Masking**: Masking the upper right, upper left, lower right, or lower left diagonal sections of the source image.
- **Random Masking**: Randomly selecting parts of the source image to mask.

The generated inpainting images were then used as input for the **CNN** and **SVM** models to enhance the detection of deepfakes.

### 2. Preprocessing
- Resize and normalize the images.
- Augment the dataset to enhance model generalization.

### 3. Data Augmentation and Adversarial Training (GAN)
- Train a GAN to generate synthetic deepfake images.
- Use these synthetic images to improve the training of the CNN and SVM.

### 4. Classification (SVM)
- Train an SVM model on the features extracted by the CNN.
- Perform hyperparameter tuning to optimize the SVM for accuracy.

### 5. Feature Extraction (CNN)
- Train a CNN model to extract features from the input images.
- Use pre-trained models (e.g., VGG16, ResNet) as a starting point for transfer learning.


### 6. Model Evaluation
- Evaluate the combined model's performance using metrics such as accuracy, precision, recall, and F1-score.
- Perform cross-validation to assess the robustness of the system.

### References
- [Kaggle Deepfake Challenge Dataset](https://www.kaggle.com/datasets/danielmao2019/deepfakeart?resource=download) or [This](https://github.com/h-aboutalebi/DeepfakeArt)
- [Stable Diffusion 2 Model](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)

## Installation
To run this project, you'll need the following dependencies:

- Python 3.10 or higher
- PyTorch (for CNN and GAN implementation)
- Scikit-learn (for SVM)
- OpenCV (for image processing)
- NumPy and Pandas (for data manipulation)
- Matplotlib (for plotting and visualization)
- Pillow (for image handling)
- diffusers (for Stable Diffusion inpainting pipeline)

You can install the required libraries using:
```bash
pip install torch scikit-learn opencv-python-headless numpy pandas matplotlib pillow diffusers

