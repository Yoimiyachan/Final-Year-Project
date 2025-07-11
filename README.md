Project Title:
Gaussian Mixture Models (GMM) for ECG Classification with Explainable AI (XAI)

Author:
PHANG YU ZHEN
Multimedia University

-----------------------------------------
REQUIREMENTS
-----------------------------------------

Tools:
- Python 3.10 or above
- Jupyter Notebook (recommended via Anaconda)
- Git (for version control, optional)

Libraries:
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn
- shap
- pywt (for Discrete Wavelet Transform)
- wfdb (for working with MIT-BIH database, if downloading from PhysioNet)

You can install the required libraries using pip:
pip install numpy pandas matplotlib seaborn scipy scikit-learn shap pywt wfdb


-----------------------------------------
DATASET
-----------------------------------------

Dataset Used:
MIT-BIH Arrhythmia Database

Source:
https://physionet.org/content/mitdb/1.0.0/

Notes:
- You may need to convert ECG records into CSV files for preprocessing.
- Ensure annotations are correctly parsed from `.atr` or corresponding annotation files.

-----------------------------------------
HOW TO RUN
-----------------------------------------

1. Launch Jupyter Notebook:
jupyter notebook


2. Open `main.ipynb` file.

3. Execute the cells sequentially:
   - Preprocessing: denoising (Wavelet + Butterworth), normalization
   - Segmentation: beat-wise extraction using annotation
   - Feature Extraction: time-domain and frequency-domain
   - GMM Model Training
   - XAI (SHAP) Explanation and Visualization
   - Evaluation (Confusion Matrix, Accuracy, Precision, Recall, F1-score)

4. Output files:
   - `denoised_labeled_beats.csv`: Cleaned ECG segments
   - `ecg_features.csv`: Features for each beat
   - Confusion matrix and SHAP plots in the notebook

-----------------------------------------
NOTES
-----------------------------------------

- Make sure that the ECG data is stored in the correct path referenced in the notebook.
- If you encounter memory issues, consider downsampling or using a subset of the MIT-BIH records.
- This code is intended for educational and academic use only.
