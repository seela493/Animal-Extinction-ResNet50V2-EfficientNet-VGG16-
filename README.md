# Endangered Species Classification using Deep Learning & IUCN Red List Mapping

This project uses deep learning to classify images of endangered animal species and map them to their corresponding IUCN Red List extinction risk categories. It supports conservation efforts by providing an AI-based framework for recognizing and monitoring wildlife from images.

---

##  Project Objectives

- Automatically classify animal species from images using deep learning.
- Evaluate extinction risk categories using the IUCN Red List.
- Compare and stack multiple pre-trained models to improve accuracy.
- Visualize model focus using Grad-CAM to promote transparency and explainability.

---

##  Dataset

- **Source**: [Kaggle – Danger of Extinction](https://www.kaggle.com/datasets)
- 6,484 labeled images across 11 endangered species
- Classes include: Panda, Lion, Rhino, Amur Leopard, Jaguars, Panthers, Chimpanzee, African Elephant, Arctic Fox, Orangutan, Cheetahs

---

## ⚙️ Workflow Summary

1. **Image Preprocessing**
   - Resize to `224x224`, convert to JPG/RGB
   - Remove duplicates and normalize resolution

2. **Exploratory Data Analysis**
   - Class imbalance visualization
   - Image dimension and mode analysis
   - Metadata cleanup

3. **Model Training**
   - Transfer learning: VGG16, EfficientNetB0, ResNet50V2
   - Stratified K-Fold Cross-Validation
   - Hyperparameter tuning using Optuna
   - Stacking ensemble via Logistic Regression

4. **Testing and Evaluation**
   - Confusion matrix, classification report, accuracy
   - Visual explanations using Grad-CAM

5. **Mapping to IUCN Categories**
   - Maps classified species to categories like Critically Endangered, Vulnerable, etc.

---

## Models and Accuracies

| Model          | Pretrained Accuracy | Retrained Accuracy | **Testing Accuracy** |
|----------------|---------------------|---------------------|------------------------|
| VGG16          | ~87%                | ~89%                | **~90%**               |
| EfficientNetB0 | ~93%                | ~95%                | **~93.5%**             |
| ResNet50V2     | ~92%                | ~94%                | **~91.7%**             |
| **Stacked**    | —                   | **~97%**            | **~97.04%**            |

---

##  Metrics Used

- Accuracy and Validation Loss
- Classification Report: Precision, Recall, F1-score
- Confusion Matrix for class-wise performance
- Grad-CAM for image explainability

---

## Results

- EfficientNetB0 performed best as a standalone model.
- The Stacked model (ensemble of VGG16, ResNet50V2, EfficientNetB0) achieved the highest testing accuracy (~97.04%).
- Grad-CAM visualizations provided clear insights into model decision areas.
- Mapped species to appropriate IUCN categories like Endangered, Vulnerable, etc.

---

## Requirements

- Python 3.8+
- TensorFlow / Keras
- Optuna
- Scikit-learn
- Pandas, Matplotlib, PIL, Seaborn

Install dependencies:

```bash
pip install -r requirements.txt
