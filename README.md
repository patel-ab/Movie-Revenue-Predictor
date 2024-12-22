# Movie-Revenue-Predictor


A machine learning-based system designed to predict movie revenues by integrating diverse data types, including numerical, categorical, and textual features. The project leverages state-of-the-art techniques such as text embeddings (BERT), Principal Component Analysis (PCA), and feedforward neural networks to provide accurate revenue forecasts, aiding stakeholders in decision-making and resource optimization.

---

## **Table of Contents**
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Dataset Overview](#dataset-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Results](#experimental-results)
- [Future Enhancements](#future-enhancements)

---

## **Introduction**

The movie industry faces significant financial uncertainties. This project aims to mitigate these risks by forecasting box office revenue using historical data and machine learning techniques. The system incorporates heterogeneous data sources, such as:
- **Numerical**: Budget, runtime, etc.
- **Categorical**: Genres, languages.
- **Textual**: Plot summaries, embedded using BERT.

Challenges like missing data, high dimensionality, and feature integration are addressed through advanced preprocessing techniques and predictive modeling, achieving an accuracy of approximately **94%**.

---

## **Technologies Used**

### **Programming Languages and Libraries**
- **Python**: Core programming language.
- **Machine Learning**: Scikit-learn, TensorFlow.
- **Deep Learning**: Transformers (BERT embeddings), TensorFlow.
- **Data Manipulation**: Pandas, NumPy.
- **Visualization**: Matplotlib, Seaborn.

### **Modeling Techniques**
- **Text Embeddings**: BERT for semantic representation of textual data.
- **Dimensionality Reduction**: PCA for reducing high-dimensional data.
- **Predictive Modeling**: Feedforward neural networks for regression.

---

## **Dataset Overview**

### **Data Sources**
- **Numerical Features**: Budget, runtime, etc.
- **Categorical Features**: Genres, production companies, etc.
- **Textual Features**: Plot summaries.

### **Preprocessing Steps**
1. **Handling Missing Values**: Imputation techniques.
2. **Feature Scaling**: Normalization and standardization for numerical features.
3. **Categorical Encoding**: One-hot encoding for categorical variables.
4. **Dimensionality Reduction**: PCA for computational efficiency.

---

## **Features**
- **Revenue Prediction**: Regression analysis to forecast movie revenue.
- **Text Embeddings**: BERT-based embeddings to capture plot semantics.
- **Integrated Pipeline**: Combines numerical, categorical, and textual features into a unified model.
- **Advanced Feature Engineering**:
  - PCA for dimensionality reduction.
  - Encoding for categorical data.

---

## **Installation**

### **Prerequisites**
Ensure you have Python and the necessary libraries installed:
```bash
pip install pandas numpy scikit-learn tensorflow transformers matplotlib seaborn
```

---

## **Usage**

1. **Load Data**:
   - Use datasets with columns like `budget`, `runtime`, `genres`, and `plot`.
   
2. **Preprocess Data**:
   - Clean and preprocess the data using the provided pipeline.

3. **Model Training**:
   - Train the model using feedforward neural networks with integrated features.

4. **Evaluate**:
   - Evaluate the model on test data using metrics like RMSE, MAE, and R².

5. **Run Predictions**:
   - Use the trained model to predict revenue for new movies.

---

## **Experimental Results**

- **Accuracy**: ~94% for test data.
- **Sample Prediction**:
  - **Movie**: *Pirates of the Caribbean: At World's End*.
  - **Actual Revenue**: $961M.
  - **Predicted Revenue**: $904M.

---

## **Future Enhancements**
- **Additional Features**: Include actor popularity and production company success rates.
- **Hyperparameter Tuning**: Use advanced techniques like Optuna or GridSearchCV.
- **Reframe Problem**: Adapt the system to classify movies into revenue categories.
- **Real-Time Prediction**: Develop a web interface using Flask or FastAPI.

---

---

## **How to Run the Project**

### **1. Set Up Your Environment**

1. **Install Python**:
   - Ensure Python 3.8 or higher is installed. You can download it from [python.org](https://www.python.org/).

2. **Create a Virtual Environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/macOS
   env\Scripts\activate     # For Windows
   ```

3. **Install Dependencies**:
   Install all required libraries using `pip`:
   ```bash
   pip install pandas numpy scikit-learn tensorflow transformers matplotlib seaborn
   ```

---

### **2. Prepare the Dataset**

1. **Place the Dataset Files**:
   - Ensure your datasets (e.g., `movie_dataset.csv`, `cast_popularity.csv`, etc.) are in the same directory as the project or a specified `data/` folder.

2. **Verify Column Names**:
   - The dataset should include essential columns like:
     - `budget`
     - `runtime`
     - `genres`
     - `plot`
     - `revenue` (if available for training).

---

### **3. Run the Jupyter Notebook**

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the Project Notebook**:
   - Navigate to the project directory and open the `Projet.ipynb` file.

3. **Run All Cells**:
   - Execute the cells in sequence to preprocess the data, train the model, and evaluate predictions.

---

### **4. Execute the Script**

If a Python script (`script.py`) is provided for automating predictions:

1. **Run the Script**:
   ```bash
   python script.py
   ```

2. **Provide Input**:
   - Ensure the required input files (e.g., `movie_dataset.csv`) are present.
   - The output (predictions) will be saved in a file like `predictions.csv`.

---

### **5. Evaluate Results**

- Use the notebook's visualizations and metrics (e.g., RMSE, MAE, R²) to analyze model performance.
- Review the predictions file to see revenue forecasts.

---

### **6. Optional Enhancements**

- **Deploy the Model**:
   Use a web framework like Flask or FastAPI to create a real-time API for predictions.
- **Adjust Hyperparameters**:
   Modify the training configuration in the notebook or script to experiment with model performance.

---
