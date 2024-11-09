# IS424G2T32024

# Deepfake Detection Using CNN, SVM, and Stable Diffusion

## Project Overview
This project aims to develop a robust system for detecting deepfake images of the inpainted class by combining three powerful machine learning models: Convolutional Neural Network (CNN), Support Vector Machine (SVM), and Stable Diffusion. The goal is to create a multi-model approach that leverages the strengths of each technique to improve the accuracy and reliability of deepfake detection.

**Team Members**: Ying Xuan, Bryan, Zhi You, Zhi Xuan, Cedric, Kaegene, Hao Tian, Hong Teng

## Dataset
This is the link to the [original dataset](https://www.kaggle.com/datasets/danielmao2019/deepfakeart?resource=download)
The diffusion model we used was modified based on [this model](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)

## Installation
To run this project, you'll need the following dependencies:

- Python 3.10 or higher
- Scikit-learn (for SVM)
- OpenCV (for image processing)
- NumPy and Pandas (for data manipulation)
- Matplotlib (for plotting and visualization)
- Pillow (for image handling)
- diffusers (for Stable Diffusion inpainting pipeline)

You can install the required libraries using:
```bash
pip install scikit-learn opencv-python-headless numpy pandas matplotlib pillow diffusers

