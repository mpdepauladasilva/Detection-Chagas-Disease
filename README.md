# Detection-Chagas-Disease -- PhD Thesis Project

This repository contains the source code and experiments developed
during my PhD research focused on the automatic detection and
classification of Chagas disease using electrocardiogram (ECG) signals,
following the guidelines and structure proposed by the George B. Moody
PhysioNet Challenge 2025.

This project is academically oriented and extends the Challenge proposal
with methodological, experimental, and analytical contributions
developed as part of my doctoral thesis.

------------------------------------------------------------------------

## 🧠 About This Research

This work investigates artificial intelligence methods for detecting
cardiac abnormalities associated with Chagas disease using ECG signals.

The main goals of this research are:

-   Develop robust AI models for Chagas disease detection\
-   Evaluate different signal preprocessing strategies\
-   Compare classical machine learning and deep learning approaches\
-   Ensure reproducibility following Challenge submission standards\
-   Analyze model interpretability and generalization across datasets

⚠️ This repository is part of an ongoing PhD thesis and may contain
experimental components under continuous development.

------------------------------------------------------------------------

## 📂 Repository Structure

    ├── train_model.py
    ├── run_model.py
    ├── team_code.py
    ├── helper_code.py
    ├── requirements.txt
    ├── notebooks/
    └── README.md

### Root directory (Challenge-required scripts)

The following scripts follow the official submission format and must
remain in the root directory:

-   `train_model.py` -- Script used to train the model\
-   `run_model.py` -- Script used to run inference\
-   `team_code.py` -- Main implementation (model training, loading,
    inference)\
-   `helper_code.py` -- Utility functions

These scripts are structured to be compatible with the Challenge
evaluation pipeline.

------------------------------------------------------------------------

## 📓 Notebooks

All experimental analyses, exploratory studies, preprocessing
validation, and intermediate experiments are located in the:

    notebooks/

These notebooks include:

-   Signal preprocessing experiments\
-   Feature engineering studies\
-   Model comparison experiments\
-   Validation strategies\
-   Performance analysis and visualization

The notebooks were used during research development and may include
exploratory code not intended for final submission.

------------------------------------------------------------------------

## 🗂 Dataset Access

The datasets used in this research follow the Challenge guidelines and
include:

-   CODE-15%\
-   SaMi-Trop\
-   PTB-XL

These datasets were processed according to the official Challenge
requirements (WFDB formatting, header adjustments, label integration,
etc.).

⚠️ Important:

-   The processed dataset is **not included in this repository**.\
-   The data is stored in a private Google Drive.\
-   Access must be requested directly from the author.

All shared data has already been preprocessed according to the Challenge
specifications and is ready to be used with the provided scripts.

------------------------------------------------------------------------

## ⚙️ How to Run the Project

### 1️⃣ Install dependencies

It is recommended to create a virtual environment and run:

    pip install -r requirements.txt

------------------------------------------------------------------------

### 2️⃣ Train the model

    python train_model.py -d training_data -m model

Where:

-   `training_data` → folder containing processed WFDB files with
    labels\
-   `model` → folder where the trained model will be saved

------------------------------------------------------------------------

### 3️⃣ Run inference

    python run_model.py -d holdout_data -m model -o holdout_outputs

Where:

-   `holdout_data` → folder containing WFDB files (labels optional)\
-   `model` → trained model directory\
-   `holdout_outputs` → folder for predictions

------------------------------------------------------------------------

### 4️⃣ Evaluate the model

Example:

    python evaluate_model.py -d holdout_data -o holdout_outputs -s scores.csv

------------------------------------------------------------------------

## 🐳 Running with Docker

Build the image:

    docker build -t detection-chagas .

Run:

    docker run -it \
      -v /path/to/model:/challenge/model \
      -v /path/to/training_data:/challenge/training_data \
      -v /path/to/holdout_data:/challenge/holdout_data \
      -v /path/to/outputs:/challenge/holdout_outputs \
      detection-chagas bash

------------------------------------------------------------------------

## 🔬 Research Contributions

This PhD work extends baseline approaches by:

-   Investigating advanced preprocessing techniques for ECG signals\
-   Evaluating deep neural network architectures\
-   Studying class imbalance strategies\
-   Performing cross-dataset validation\
-   Analyzing model robustness\
-   Exploring explainability techniques for medical AI

This repository may evolve as the thesis progresses.

------------------------------------------------------------------------

## 📚 Academic Context

This project is part of a doctoral research in Electrical Engineering
focused on:

-   Biomedical signal processing\
-   Artificial intelligence applied to cardiology\
-   Automated detection of Chagas cardiomyopathy

The work contributes to computational tools for improving early
diagnosis in endemic regions.

------------------------------------------------------------------------

## 📩 Dataset Access Request

If you need access to the processed dataset used in this research,
please contact the repository author to request permission.

The dataset:

-   Is stored in Google Drive\
-   Has already been processed according to Challenge specifications\
-   Is organized in WFDB format\
-   Includes integrated demographic and label metadata

------------------------------------------------------------------------

## ⚠️ Disclaimer

This repository is part of an academic PhD thesis and is intended for
research purposes only. It is not a clinical diagnostic tool.