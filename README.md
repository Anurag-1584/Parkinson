**Parkinson's Disease Detection using Machine Learning**
A machine learning prototype for detecting Parkinson's-like patterns
from voice biomarkers and spiral drawing images.
The project uses the UCI Parkinson's Dataset for voice-based
classification and includes a separate experimental spiral-image
pipeline. A Gradio interface brings the two pipelines together into a
simple multimodal screening prototype.
> \*\*Disclaimer:\*\* This project is for educational and research purposes
> only. It is not a medical diagnostic system and should not be used as
> a substitute for professional clinical evaluation.
**Project Overview**
The notebook covers:
Exploratory Data Analysis (EDA)
Voice-feature preprocessing and scaling
Class balancing using SMOTE and SMOTETomek
Stratified 5-fold cross-validation
Comparison of SVM, Random Forest, and XGBoost
Validation-based classification threshold selection
Final XGBoost evaluation
Feature-importance analysis
Model, scaler, and threshold serialization with Joblib
Experimental `.wav` voice-feature extraction using Parselmouth
Experimental real-time microphone prediction
Separate spiral drawing classification using Random Forest
Gradio-based prototype interface
Combined voice + spiral prototype score
**Dataset**
Voice Dataset
The notebook downloads the UCI Parkinson's Dataset directly from the
UCI repository.
The dataset contains biomedical voice measurements such as:
Fundamental frequency: `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`
Jitter measurements
Shimmer measurements
Noise-to-Harmonics Ratio (NHR)
Harmonics-to-Noise Ratio (HNR)
Nonlinear measures such as RPDE, DFA, spread1, spread2, D2, and PPE
The target column is:
`status = 0` → Healthy
`status = 1` → Parkinson's
The `name` column is removed before model training because it is an
identifier rather than a predictive feature.
Spiral Dataset
The spiral model expects a local image dataset. The recommended
structure is:
``` text
spiral\_dataset/
├── healthy/
│   ├── image1.png
│   └── image2.png
└── parkinson/
    ├── image1.png
    └── image2.png
```
Nested folder structures are also supported as long as the folder names
contain recognizable labels such as `healthy`, `control`, `parkinson`,
`parkinsons`, `pd`, `patient`, or `patients`.
The spiral pipeline is separate from the voice pipeline; the image
features are not mixed directly with the UCI voice features.
**Machine Learning Pipeline**
1. Exploratory Data Analysis
The notebook performs:
Dataset inspection
Missing-value checking
Class-distribution analysis
Feature correlation visualization
Comparison of important voice-feature distributions
2. Train-Test Split
The voice dataset is divided using a stratified 80/20 train-test split.
Scaling is performed with:
``` python
StandardScaler()
```
The scaler is fitted only on the training data and then applied to the
test data.
3. Class Balancing
Because the dataset is imbalanced, two approaches are evaluated:
SMOTE
SMOTETomek
Oversampling is applied only to training data.
For cross-validation, scaling and SMOTE are placed inside an `imblearn`
pipeline so that preprocessing is performed separately within each fold.
4. Model Comparison
The notebook compares three baseline classifiers:
Model           Configuration
---
SVM             RBF kernel
Random Forest   200 estimators
XGBoost         200 estimators
The evaluation includes:
Accuracy
Recall
Precision
F1-score
ROC-AUC
The notebook explicitly treats these as baseline models rather than
fully hyperparameter-tuned models.
Cross-Validation
Because the UCI dataset is relatively small, the notebook uses:
``` text
Stratified 5-Fold Cross-Validation
```
The cross-validation pipeline performs:
``` text
StandardScaler
      ↓
SMOTE
      ↓
Classifier
```
Mean scores and standard deviations are reported to compare both
performance and stability across folds.
Final Voice Model
The notebook uses XGBoost as the final voice model.
A validation set is used to examine different probability thresholds
between `0.10` and `0.85`.
The selected threshold is based primarily on:
F1-score
Recall
Precision
The model is then retrained on the complete training set using SMOTE and
evaluated on the untouched test set.
The notebook reports:
Classification report
ROC-AUC
Confusion matrix
Top 15 feature importances
ROC curve
The actual numerical results depend on the notebook execution and are
intentionally not hard-coded into this README.
**Model Artifacts**
The notebook saves the following files:
``` text
parkinsons\_xgboost\_model.pkl
parkinsons\_scaler.pkl
parkinsons\_threshold.pkl
```
The spiral pipeline can additionally produce:
``` text
spiral\_random\_forest\_model.pkl
```
Experimental Voice Prediction
The notebook contains an experimental audio pipeline using
Parselmouth/Praat.
It extracts a subset of voice features directly from `.wav` audio,
including:
Fundamental frequency
Frequency range
Jitter
RAP
PPQ
DDP
Shimmer
APQ
NHR
HNR
Some nonlinear UCI features cannot be reliably extracted by this
pipeline. Therefore, the notebook trains a separate real-time
demonstration model using only the features that can actually be
extracted from raw audio.
This avoids using placeholder or randomly generated features.
Example:
``` python
predict\_from\_wav("your\_audio.wav")
```
The notebook also supports microphone recording:
``` python
record\_and\_predict()
```
The expected demonstration input is a sustained `"Ahhh"` vocalization.
Spiral Drawing Model
The spiral pipeline converts an image into numerical features using:
Grayscale conversion
Autocontrast
Image resizing
Ink-density statistics
Spatial statistics
Bounding-box characteristics
Pixel-level features
A Random Forest classifier is then trained on the extracted
features.
Example:
``` python
predict\_spiral\_from\_image("spiral.png")
```
The spiral model is intentionally kept separate from the voice model.
Gradio Interface
The notebook includes a Gradio interface that allows the user to:
Upload or record a voice sample
Upload a spiral drawing
Analyze either modality independently
View voice and spiral scores separately
View a combined prototype score
When both inputs are available, the combined score is calculated as the
mean of the available voice and spiral model probabilities.
This is a simple prototype aggregation method, not a clinically
validated multimodal fusion method.
Project Structure
``` text
.
├── parkinson2(7).ipynb
├── parkinsons\_xgboost\_model.pkl
├── parkinsons\_scaler.pkl
├── parkinsons\_threshold.pkl
├── spiral\_random\_forest\_model.pkl
└── spiral\_dataset/
    ├── healthy/
    └── parkinson/
```
The `.pkl` files are generated when the relevant notebook sections are
executed.
**Installation**
Python 3.10 is used by the notebook.
Install the main dependencies:
``` bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install imbalanced-learn xgboost joblib
pip install praat-parselmouth sounddevice scipy
pip install pillow gradio
```
Depending on the environment, microphone/audio dependencies may require
additional system configuration.
Running the Notebook
Clone or download the project.
Create a Python 3.10 environment.
Install the required dependencies.
Open `parkinson2(7).ipynb` in Jupyter Notebook, JupyterLab, or VS
Code.
Run the cells in order.
For the spiral pipeline, create the `spiral\_dataset` folder using
the structure described above.
For the real-time voice demo, allow microphone access.
The Gradio section launches the browser-based prototype.
Limitations
This project has several important limitations:
The UCI voice dataset is small.
The baseline models are not fully hyperparameter-tuned.
The voice and spiral datasets are handled as separate pipelines.
The real-time voice model uses only a subset of the original UCI
features.
The spiral pipeline depends on a separately supplied local image
dataset.
The combined score is a simple average of model probabilities.
The pipelines have not been clinically validated.
Model predictions should not be interpreted as a medical diagnosis.
Future Improvements
Possible improvements include:
Larger and more diverse datasets
More rigorous external validation
Hyperparameter optimization
Better calibration of predicted probabilities
More robust audio preprocessing
Deep-learning-based spiral image models
Learned multimodal fusion instead of probability averaging
Explainability using SHAP or similar methods
Evaluation on an independent clinical dataset
Deployment as a production application after proper validation
Technologies
Languages
Python
Data Science & Machine Learning
NumPy
Pandas
Scikit-learn
XGBoost
Imbalanced-learn
Audio Processing
Parselmouth / Praat
SciPy
SoundDevice
Computer Vision
Pillow
Visualization
Matplotlib
Seaborn
Deployment / Interface
Gradio
Model Serialization
Joblib
Author
Anurag Bisht
---
Note
This repository demonstrates a machine-learning research prototype for
Parkinson's-like voice and spiral-drawing pattern analysis. It is
intended to demonstrate data preprocessing, class balancing, model
comparison, cross-validation, threshold selection, audio feature
extraction, image classification, and basic multimodal application
development.
