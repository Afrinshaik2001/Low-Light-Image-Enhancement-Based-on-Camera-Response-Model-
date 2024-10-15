# Low-Light Image Enhancement Using Camera Response Model

This repository implements a low-light image enhancement algorithm using a camera response model (CRM) to improve image quality in underexposed conditions. The model leverages the camera's response function to adjust exposure, reduce noise, and restore details lost in low-light images.

## Features

- **Noise Reduction**: Suppresses noise while preserving details using CRM-based noise modeling.
- **Exposure Adjustment**: Enhances underexposed images using an inverse camera response function.
- **Dynamic Range Improvement**: Recovers details from both shadow and highlight regions.
- **Deep Learning Support**: Integrates with deep learning models for end-to-end image enhancement.
- **Multi-Exposure Fusion**: Supports multi-exposure image fusion for better dynamic range.

## Model Architecture

We use a combination of traditional CRM-based enhancement and deep learning techniques. The pipeline includes the following:

- **Inverse Camera Response Model**: To linearize the image and adjust exposure.
- **Noise Suppression**: Based on noise characteristics estimated from the CRM.
- **Deep Learning-based Enhancement (Optional)**: Uses convolutional neural networks to improve image quality further.

## Dataset

The model is trained and tested on public datasets of low-light images. You can download the dataset we used from [link to dataset]. The datasets include pairs of low-light and well-exposed images for supervised training.

Example datasets include:
- **LOL (Low-Light) Dataset**: Contains paired low-light and bright images.
- **SID (See-in-the-Dark) Dataset**: Raw images in very low-light conditions.

## Results

We evaluate the model on standard benchmarks, and the results show significant improvements in image quality compared to existing methods. 

| Method               | VSI   | SSIM   |
|----------------------|--------|--------|
| Without CRM          | 0.39  | 0.73   |
| With CRM (Ours)      | 0.84  | 0.89   |
![Figure 1 10_16_2024 1_22_03 AM](https://github.com/user-attachments/assets/16d6edb5-ecac-4fbb-ad36-ddeb5a80371d)



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

