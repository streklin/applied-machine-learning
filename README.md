# Applied Machine Learning Project: Exoplanet Detection

## Project Overview
This project focuses on building a Machine Learning model to predict whether a star has an exoplanet orbiting around it based on temporal observations of the star's light intensity (flux). If a planet passes between the star and Earth, the light intensity dims compared to other observations, which indicates a possible planetary transit. The dataset used for this project contains flux measurements sourced from the Kepler Telescope via Kaggle.

## Installation and Run Instructions

1. **Get the Project:**
   Navigate into the project directory:
   ```bash
   cd applied-machine-learning
   ```

2. **Install dependencies:**
   Make sure you have Python and pip installed. Install the required libraries using the `requirements.txt` file (you may want to do this within a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook:**
   The project logic and analysis are contained within a Jupyter Notebook. Launch Jupyter Notebook or JupyterLab to view and run the notebook:
   ```bash
   jupyter notebook modeling.ipynb
   ```
   *Note: The dataset is fetched dynamically in the first stages of the notebook via `kagglehub`.*

## Summary of Results

The dataset is extremely imbalanced, with only 37 out of 5087 rows of training data representing a positive match (presence of an exoplanet). The test set features only 5 positive matches out of approximately 500 rows.

To properly analyze and model the data, the following preprocessing steps were applied:
* **Fast Fourier Transform (FFT)** was used to remove high-frequency noise from the signal.
* Data was **binned using z-scores** to reduce dimensionality and extract significant signals of possible planetary transits across the star.
* The observations were **standardized**.

An initial Support Vector Machine (SVC) model was trained using oversampled data. We subsequently fine-tuned the model's hyperparameters using `GridSearchCV`.

**Final Model Performance:**
* **Accuracy Score:** 0.994
* **F1 Score:** 0.72
* **Recall Score:** 0.80
* **Precision Score:** 0.67

Ultimately, the optimized SVC system correctly identified **4 out of 5** stars with an Exoplanet in the test set. It correctly identified all but 2 of the negative cases (stars without exoplanets). It is interesting to note that while the optimized model scored higher on Accuracy, Precision, and F1, the `GridSearchCV` approach involved trading a slight dip in Recall (missing one exoplanet) for a significant reduction in false positives (down from 9 to 2).
