# Face Recognition System Using FaceNet and SVM

## Project Overview

This project implements a Face Recognition System using a pretrained FaceNet model for feature extraction and a Support Vector Machine (SVM) classifier for face identification. The system detects faces from images, generates 512-dimensional facial embeddings, and predicts the identity of individuals with high accuracy.

## Features

* Face detection using MTCNN
* Face embedding generation using pretrained FaceNet
* Face recognition using Support Vector Machine (SVM)
* Balanced dataset preprocessing
* Random test image evaluation
* Accuracy and confidence score prediction
* Google Colab compatible implementation

## Dataset

This project uses the **Labeled Faces in the Wild (LFW)** dataset.

### Dataset Preprocessing

To improve the quality and fairness of the training data, the original LFW dataset was preprocessed to create a balanced dataset.

The preprocessing process includes:

* Scanning all individuals in the LFW dataset.
* Selecting only those individuals who have **at least 20 face images**.
* Randomly selecting **exactly 20 images** for each eligible individual.
* Copying the selected images into a new directory named **LFW_72_People**.
* Creating a balanced dataset where every person contributes the same number of images.

This preprocessing resulted in a dataset containing:

* **72 individuals**
* **20 images per individual**
* **1,440 face images** in total

This balanced dataset helps reduce class imbalance and improves the performance and reliability of the face recognition model.

### Dataset Structure

```text
LFW_72_People/
│── Person_1/
│     ├── image1.jpg
│     ├── image2.jpg
│     └── ...
│
│── Person_2/
│     ├── image1.jpg
│     └── ...
│
└── Person_72/
```

## Technologies Used

* Python
* TensorFlow
* Keras FaceNet
* MTCNN
* OpenCV
* NumPy
* Scikit-learn
* Matplotlib

## Workflow

1. Download and preprocess the LFW dataset.
2. Create a balanced dataset with 72 individuals and 20 images per person.
3. Detect faces using MTCNN.
4. Crop and resize faces to **160 × 160** pixels.
5. Generate **512-dimensional embeddings** using the pretrained FaceNet model.
6. Encode class labels.
7. Train an SVM classifier using the extracted embeddings.
8. Predict identities for unseen face images.
9. Evaluate the model using test images.

## Model Performance

* **Feature Extractor:** FaceNet (Pretrained)
* **Classifier:** Support Vector Machine (SVM)
* **Embedding Size:** 512 Dimensions
* **Random Test Accuracy:** 100% (5/5 Random Test Images)

> Note: The 100% accuracy shown above is based on a small random sample. The complete test dataset should be used for a comprehensive evaluation.

## Project Structure

```text
Face-Recognition/
│── Facial_Regonation.ipynb
│── LFW_72_People/
│── train.csv
│── test.csv
│── README.md
```

## Installation

Install the required libraries:

```bash
pip install keras-facenet mtcnn tensorflow opencv-python scikit-learn matplotlib
```

## Future Improvements

* Real-time face recognition using a webcam.
* Deploy the model using Flask or Streamlit.
* Support unknown face detection.
* Improve performance using larger datasets.
* Experiment with different classifiers and hyperparameter tuning.

## Results

The project successfully demonstrates a complete face recognition pipeline by combining **MTCNN** for face detection, **FaceNet** for feature extraction, and **SVM** for classification. Creating a balanced dataset during preprocessing improves the fairness of the training process and contributes to reliable recognition performance.
