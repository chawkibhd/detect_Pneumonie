# Project README: Pneumonia Detection from Chest X-Ray Images

This project aims to detect pneumonia from lung X-ray images using various classification approaches: two based on Logistic Regression (LR) and one based on a Convolutional Neural Network (CNN).

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Project Objective](#project-objective)  
3. [Medical Context: Pneumonia](#medical-context-pneumonia)  
4. [Methodology](#methodology)  
   - [Data Collection](#data-collection)  
   - [Image Preprocessing](#image-preprocessing)  
   - [Feature Extraction (for LR)](#feature-extraction-for-lr)  
   - [Classification (LR and CNN)](#classification-lr-and-cnn)  
5. [Project Structure](#project-structure)  
6. [Prerequisites and Installation](#prerequisites-and-installation)  
7. [Usage](#usage)  
8. [Performance Evaluation](#performance-evaluation)  
9. [Conclusion and Future Work](#conclusion-and-future-work)  
10. [Authors and Credits](#authors-and-credits)  

---

## 1. Introduction
The rise of artificial intelligence and machine learning in healthcare has enabled increasingly efficient medical image analysis tools. Chest X-ray is the most common method to detect pneumonia. This project implements several automated detection methods to help healthcare professionals refine their diagnoses.

---

## 2. Project Objective
This project seeks to **classify lung X-ray images** into two categories:
- **“Normal” images** (no pneumonia)
- **“Pneumonia” images** (presence of pneumonia)

To achieve this, we propose **three distinct approaches**:
1. **Logistic Regression** with **a single column** containing combined extracted features (Gabor, DCT, Fourier, PHOG).  
2. **Logistic Regression** with **multiple columns**, where each column represents one feature type (Gabor, DCT, Fourier, PHOG).  
3. **Convolutional Neural Network (CNN)**, which learns directly from the raw images.

---

## 3. Medical Context: Pneumonia
Pneumonia is a respiratory infection that affects the lungs, typically caused by bacteria, viruses, or fungi. Chest X-rays are essential for diagnosis, but reading them can be complex and time-consuming. Automating detection helps reduce diagnosis time and improve accuracy.

---

## 4. Methodology

### Data Collection
- **5200 total** X-ray images, including **1324 normal images** and **3876 images** showing pneumonia.
- Images were provided by a professor, covering various pneumonia cases (bacterial, viral, etc.).

### Image Preprocessing
- **Histogram equalization** to enhance image quality.  
- **Noise reduction** (filtering) to remove unwanted artifacts.  
- **Dimension normalization** for consistent image sizes across samples.

### Feature Extraction (for LR)
We use four types of descriptors to represent the images:
1. **Gabor Filter**: highlights textures and orientations.  
2. **Fourier Transform**: analyzes the image in the frequency domain to detect periodic patterns.  
3. **Discrete Cosine Transform (DCT)**: reduces spatial redundancies, useful for compression and variation detection.  
4. **Pyramid Histogram of Oriented Gradients (PHOG)**: highlights contours by counting oriented gradients.

These features are either:
- **Combined** into a single column (*single_feature.csv*), or  
- **Separated** into four columns (*multi_features.csv*).

### Classification (LR and CNN)
1. **Logistic Regression**:  
   - *lr_single_feature.ipynb*: training and evaluation using a single column of features.  
   - *lr_multi_features.ipynb*: training and evaluation using multiple feature columns (Gabor, DCT, Fourier, PHOG in separate columns).

2. **Convolutional Neural Network**:  
   - *tensorflow-cnn.ipynb*: building and training a CNN model from raw images.  
   - *cnn_model.h5*: a saved version of the best CNN model, used for inference or evaluation.

---

## 5. Project Structure

```bash
DETECT_PNEUMONIE/
├── core/
│   ├── ImageFeatureExtractor.py         # CSV filling using feature extraction
│   └── PneumoniaDetectorApp.py          # Tkinter main app
├── data/
│   ├── multi_features.csv               # Gabor, DCT, Fourier, PHOG features in separate columns
│   └── single_feature.csv               # Combined features in a single column
├── model_training/
│   ├── cnn/
│   │   └── tensorflow-cnn.ipynb         # Notebook for CNN training
│   └── logistic_regression/
│       ├── lr_multi_features.ipynb      # Notebook for multi-column LR
│       └── lr_single_feature.ipynb      # Notebook for single-column LR
├── models/
│   ├── logistic_regression_model.pkl     # single feature LR model
│   ├── logreg-v3.pkl                     # Multiple features LR model
│   └── my_model-v4-bestOne.h5            # Best trained CNN model
├── utils/
│   ├── RemplissageCSV.ipynb              # CSV filling using feature extraction
│   ├── Renamepic.ipynb                   # Bulk renaming script for images
│   └── .gitignore
├── environment.yml                       # Conda environment file
├── main.py                               # project entry point
├── README.md                             # This README document
└── test.ipynb                            # Miscellaneous test notebook
```

---

## 6. Prerequisites and Installation

1. **Install Anaconda/Miniconda** (highly recommended) or ensure you have Python 3.x.  
2. Clone this repository:  
   ```bash
   git clone https://github.com/your-repo/detect_pneumonie.git
   cd detect_pneumonie
   ```
3. **Create an environment** from the `environment.yml` file:  
   ```bash
   conda env create -f environment.yml
   conda activate detect_pneumonie_env
   ```
   *Or* manually install the dependencies listed in `environment.yml`.

4. **Verify that you have**:  
   - Python 3.7+  
   - TensorFlow (or PyTorch, if mentioned in the environment)  
   - NumPy, SciPy, scikit-learn, OpenCV, etc.

---

## 7. Usage

### Detection Methods
1. **Logistic Regression (multi or single feature)**  
   - Open either `lr_single_feature.ipynb` or `lr_multi_features.ipynb` in a Jupyter environment.  
   - Run the cells to:  
     1. Load the CSV data.  
     2. Train the LR model.  
     3. Evaluate performance (accuracy, recall, F1, confusion matrix).  

2. **Convolutional Neural Network (CNN)**  
   - Open `tensorflow-cnn.ipynb`.  
   - Run the notebook to:  
     1. Load the images.  
     2. Build, compile, and train the CNN.  
     3. Evaluate performance on a test set.  
   - The final model is saved in `my_model-v4-bestOne.h5`.

### Running via `main.py`
- `main.py` serve as an **entry point** to run a complete pipeline or launch a quick test interface.  
  - For example:  
    ```bash
    python main.py
    ```
  - Options may vary depending on your implementation in `PneumoniaDetectorApp.py`.

### Utility Scripts
- `RemplissageCSV.ipynb`: Demonstration of how to fill in CSVs by extracting features from images.  
- `Renamepic.ipynb`: Allows bulk renaming of images (useful for dataset organization).  

---

## 8. Performance Evaluation
Model performance is evaluated using:
- **Accuracy**, **Recall**, and **F1-score**.  
- **Confusion matrix** to analyze true positives, false positives, true negatives, and false negatives.

The goal is to minimize false negatives (misdiagnosed patients) and achieve high overall accuracy for a reliable medical tool.

---

## 9. Conclusion and Future Work
This project demonstrates how **different machine learning approaches** (Logistic Regression and CNN) can be applied to pneumonia detection in chest X-rays. Results show that:

- **Logistic Regression** is a simple, fast model that performs well given properly engineered features.  
- **Convolutional Neural Networks (CNN)** often yield better performance, provided there is sufficient data and computational power for training.

**Future Work**:
- Use **more diverse data** from various sources (hospitals, research centers).  
- Enhance the CNN model (additional layers, fine-tuning on pretrained models, etc.).  
- Develop a **user interface** (web API, desktop app) to make integration into medical workflows more seamless.

---

## 10. Authors and Credits
- **Project Team**: Chawki Belhadid, Samir Akram OUNIS.  
- **Data**: 5200 X-ray images.  
- **Libraries**: TensorFlow, scikit-learn, NumPy, OpenCV, etc.