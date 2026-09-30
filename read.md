# Student Performance Prediction

A machine learning project that predicts students' final mathematics grades using demographic, family, lifestyle, and academic-related factors.

## Project Overview

This project uses regression models to predict a student's final grade (`G3`) based on available student information. The goal is to explore how student-related factors are associated with academic performance.

## Dataset

* **Source:** [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
* **File:** `student-mat.csv`
* **Target variable:** `G3` — final mathematics grade
* **Features:** Student demographics, family background, study habits, and other factors.

The `G1` and `G2` columns are excluded to predict the final grade without using earlier exam grades.

## Machine Learning Models

* Random Forest Regressor
* Extra Trees Regressor
* Gradient Boosting Regressor

Models are compared using five-fold cross-validation.

## Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score
* Predictions within selected grade tolerances

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

## How to Run

1. Clone this repository.

2. Install the required packages:

   ```bash
   pip install pandas numpy scikit-learn joblib
   ```

3. Download `student-mat.csv` from the UCI dataset page and place it in the project folder.

4. Run:

   ```bash
   python trainmodel.py
   ```

## Output

The program saves the selected trained model, model comparison results, test predictions, and model metadata in the `student_score_model` directory.


## Disclaimer

This project is for educational purposes. Predictions are estimates and should not be treated as definitive assessments of a student's ability.
