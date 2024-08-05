# Leukemia Cancer Detection using CNN

This project is an educational tool designed to demonstrate the use of Convolutional Neural Networks (CNNs) for image classification. The app allows users to upload an image of a cell and predicts whether the cell is cancerous or normal.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Streamlit App](#streamlit-app)
- [Disclaimer](#disclaimer)
- [Contact](#contact)

## Overview
Leukemia Cancer Detection is a project that uses a CNN model trained on images of cancerous and normal cells. The model can classify uploaded images to determine if they show signs of leukemia. This project is for educational purposes only and should not be used for medical diagnosis.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-github-username/leukemia-cancer-detection.git
    cd leukemia-cancer-detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download or place your trained model `leukemia_cnn_model.h5` in the project directory.

## Usage
To run the Streamlit app, use the following command:
```bash
streamlit run app.py

This will start the Streamlit server, and you can open the web app in your browser.

Model Training
The model was trained using TensorFlow and Keras. The training data includes images of cancerous and normal cells, organized in separate directories for training, validation, and testing.

Training Code
The training code is included in the train_model.ipynb Jupyter Notebook. Here is a brief overview of the steps involved:

Data Preprocessing: Load and preprocess the images from the training, validation, and test directories.

Model Building: Create a CNN model with layers for convolution, pooling, flattening, and dense layers.

Model Training: Compile the model and train it using the preprocessed data.

Model Saving: Save the trained model to a file (leukemia_cnn_model.h5).

