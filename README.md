# CUDA Specialization Capstone Project

## Project Title: Optimizing Image Classification with Convolutional Neural Networks using CUDA

### Overview
This project demonstrates how to optimize image classification using Convolutional Neural Networks (CNNs) with CUDA for GPU acceleration. The CIFAR-10 dataset is used for training and evaluation. By leveraging CUDA, we aim to accelerate the training and inference processes of the neural network, showcasing the power of GPU computing in deep learning tasks.

### Requirements
- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy

### Setup
1. **Clone the Repository**:
    ```sh
    git clone https://github.com/yourusername/cuda_cnn_project.git
    cd cuda_cnn_project
    ```

2. **Install the Required Packages**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Project**:
    ```sh
    python main.py
    ```

### Project Structure
- `main.py`: Main script to run the project. It handles data loading, model training, and evaluation.
- `cnn_cuda.py`: Defines the CNN architecture with CUDA. This module contains the model class with CUDA operations.
- `train.py`: Contains functions for training and validation. This module manages the training loop, validation, and performance tracking.
- `utils.py`: Utility functions for plotting training curves. This module includes functions for visualizing training and validation loss and accuracy.
- `README.md`: Project documentation, including setup instructions and project details.
- `requirements.txt`: List of required packages for the project.

### Implementation Details
1. **Data Preprocessing and Augmentation**:
   - Uses `torchvision.transforms` for data augmentation, including random horizontal flip and random cropping.
   - Normalizes the images using the mean and standard deviation of the CIFAR-10 dataset.

2. **Model Architecture**:
   - A simple CNN architecture with two convolutional layers, followed by max-pooling layers.
   - Fully connected layers with a dropout layer for regularization.

3. **Training and Validation**:
   - The training loop includes forward pass, loss computation, backward pass, and optimizer step.
   - Validation loop computes the validation loss and accuracy after each epoch.

4. **CUDA Acceleration**:
   - The model and data are moved to GPU using `.cuda()` to leverage CUDA for faster computation.

### Results
After training completes, the script will plot the following:
- Training and validation loss curves.
- Validation accuracy curve.

These plots help visualize the model's performance over epochs and provide insights into how well the model is learning and generalizing.

### Example Output
![Training and Validation Loss](plots/training_validation_loss.png)
![Validation Accuracy](plots/validation_accuracy.png)

### Acknowledgements
This project is part of the CUDA Specialization Capstone Project. It aims to demonstrate the use of CUDA for accelerating deep learning models and highlights the benefits of GPU computing.

### Future Work
- Experiment with deeper and more complex CNN architectures.
- Implement advanced data augmentation techniques.
- Explore hyperparameter tuning for better performance.
- Compare the performance with other GPU acceleration libraries like cuDNN.
