# 🩻 Chest X-Ray Classification Using Deep Learning

## 📌 Project Overview

This project uses deep learning and transfer learning with **ResNet50** to analyze chest X-ray images and predict scores for 18 output classes. It combines two datasets: the COVID-19 Radiography Database and NIH ChestX-ray14.

The model is designed to identify four COVID-19-related image categories and 14 disease labels from the NIH dataset using a shared model architecture.

This is an educational machine learning project and is **not intended for clinical diagnosis**.

## 🎯 Objectives

- Build a combined 18-output chest X-ray classification model.
- Apply transfer learning using a pretrained ResNet50 network.
- Fine-tune the model to improve feature extraction.
- Handle single-label and multi-label datasets in one training pipeline.
- Evaluate model performance using classification metrics.
- Tune class-specific decision thresholds for multi-label predictions.

## 🧠 Model Architecture

**Base model:** ResNet50 pretrained on ImageNet

**Framework:** TensorFlow / Keras

**Input size:** 224 × 224 pixels

**Output layer:** 18 sigmoid outputs

The output classes are:

### COVID-19 Radiography Dataset (4 classes)
1. Normal
2. COVID
3. Lung_Opacity
4. Viral Pneumonia

### NIH ChestX-ray14 (14 disease labels)
1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural_Thickening
14. Hernia

## 📂 Datasets

### 1. COVID-19 Radiography Database
- Four image categories.
- Approximately 21,165 images in the dataset used.
- Used for four-class classification.

Dataset source: [COVID-19 Radiography Database — Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

### 2. NIH ChestX-ray14
- Approximately 112,120 images in the dataset used.
- Contains 14 disease labels.
- Images may have multiple associated findings.

Dataset source: [NIH Chest X-rays — Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)

The original datasets are not included in this repository. Users must obtain them from their respective sources and follow their terms of use.

## ⚙️ Technologies Used

- Python
- TensorFlow and Keras
- ResNet50
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Pillow
- Google Colab

## 🔬 Methodology

1. Load and preprocess images from both datasets.
2. Resize images to 224 × 224 pixels.
3. Apply ResNet50-compatible image preprocessing.
4. Prepare four-class labels and 14-label multi-label targets.
5. Combine the tasks using masked binary cross-entropy loss.
6. Train the model with the pretrained backbone initially frozen.
7. Fine-tune the final layers of ResNet50.
8. Evaluate the model on held-out validation samples.
9. Tune per-class thresholds for NIH multi-label predictions.
10. Generate prediction scores for new chest X-ray images.

## 📊 Results

### COVID-19 Dataset

| Metric | Result |
|---|---:|
| Test accuracy | Approximately 89% |
| Number of test images | 4,233 |

### NIH ChestX-ray14 Dataset

After fine-tuning and threshold optimization on the validation sample:

| Metric | Result |
|---|---:|
| Macro F1-score | Approximately 0.216 |
| Micro F1-score | Approximately 0.291 |

**Note:** Thresholds were optimized using the validation sample, so these scores should not be treated as an independent final test result. The NIH multi-label task remains challenging, and performance varies by disease class.

## 🚀 How to Run

The model was developed in Google Colab.

1. Open the project notebook in Google Colab.
2. Download the datasets from their official dataset pages.
3. Update the dataset paths in the notebook.
4. Install any missing dependencies.
5. Run the preprocessing and training cells in order.
6. Load the saved model to predict on a new X-ray image.

### Load a Saved Model

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "combined_resnet50_18class_finetuned.keras",
    compile=False
)

print("Model loaded successfully!")
print("Output classes:", model.output_shape[-1])
```

The saved model produces 18 sigmoid scores. The model's output order must match the class order listed above.

## 📁 Suggested Repository Structure

```text
chest-xray-multilabel-classification/
│
├── README.md
├── chest_xray_18class.ipynb
├── requirements.txt
├── results/
│   ├── sample_prediction.png
│   └── evaluation_results.png
└── .gitignore
```

The trained model and threshold file can be hosted separately if they are too large for ordinary GitHub uploads.

## ⚠️ Limitations and Disclaimer

- This project is for educational and research purposes only.
- It is not clinically validated and must not be used to diagnose, treat, or rule out diseases.
- Model predictions are scores, not necessarily calibrated probabilities.
- Dataset imbalance, label noise, differences between datasets, and threshold selection may affect performance.
- Results on validation data do not guarantee performance on new hospitals, devices, or patient populations.

## 🔮 Future Improvements

- Evaluate on an independent, patient-level test set.
- Improve performance on underrepresented disease classes.
- Add a simple web interface for image upload and prediction.
- Investigate model interpretability using Grad-CAM.
- Assess calibration and generalization across different datasets.

Choose a license for your own code after reviewing the licenses and terms of the datasets and any pretrained model components. Do not assume that the datasets themselves can be redistributed under your project's license.
