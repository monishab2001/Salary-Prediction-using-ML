# Salary-Prediction-using-ML
## Overview
This project leverages **Machine Learning techniques** to predict salaries based on key input features such as age, gender, education, experience, and job role. The notebook covers data preprocessing, EDA, visualizations, feature engineering, **model training**, **performance evaluation** to ensure accurate predictions.  The models were trained using a dataset with salary information and various preprocessing techniques. The best model is chosen by evaluation and deployed in **Streamlit** to allow users to input their details and get a salary prediction

## Repository Contents
- **Jupyter notebook** - `Salary Prediction.ipynb` # Main notebook with data processing & training
- **Python Code** - `app.py`  # Streamlit app for salary prediction
- **models Folder** # Saved trained models
    - `salary_prediction_model.pkl`  # Machine learning model
    - `job_titles.pkl`       # Encoded job titles
    - `feature_columns.pkl`   # Saved feature columns

## Installation
- To run this notebook, you need the following dependencies:
`pip install numpy pandas matplotlib seaborn scikit-learn joblib`
- For running the Streamlit app:
`pip install streamlit joblib`

## How to Use
1. Clone the repository
2. Open the Jupyter notebooks to run the cells step by step for data analysis and model training.
3. Make sure to have the dataset loaded is in the appropriate directory. 
4. The trained models will be saved after executing the notebook
5. Three models will be downloaded cells for model deployment by running the notebook 
6. Open `app.py` in VSCode/Terminal 
7. Ensure all dependencies are installed
8. Provide the correct paths and names for the model files to ensure smooth execution.
9. Run the code by `streamlit run app.py`
10. Open the provided URL in your web browser to interact with the app

## Process
### 📥 Data Acquisition & Exploration
- Loaded the dataset into a pandas DataFrame.
- Performed initial inspection of data types and summary statistics.

### Data Wrangling
- Handled missing values and inconsistencies in the dataset.
- Feature Engineering:
    1. Ordinal Encoding: Education Level
    2. Label Encoding: Gender
    3. One-Hot Encoding: Job Title

### Exploratory Data Analysis and Visualization
- Described summary statitics to understand feature distribution
- Visualized required  distributions using histograms 
- Visualized required compositions and comparisions using pie charts, bar charts
- Analyzed feature correlations using heatmaps and regression plots.

### Predictive Analytics
- Features used: Age, Gender, Education Level, Experience, Job Title
#### Hypertuning optimization
- Tuned hyperparameters to improve model performanc
#### Model Training and Evaluation
- Splited the dataset into training and testing sets.
- Appropriate models were trained, tested and evaluated to select the best-performing model
- Evaluated the model using  R² score and MSE
#### Model Prediction
- Model Used: RandomForestRegressor
- Predictions done with test data
#### Model Deployment
- For deployment, a Streamlit web application has been created locally in `app.py` to input details and get real-time salary predictions.

## Technologies Used
- Python - Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn
- Model Deployment - Streamlit (local hosted)

## Conclusion
This project successfully developed and deployed a salary prediction model using machine learning techniques. By leveraging features such as age, gender, education, experience, and job title, the model provides reasonably accurate salary estimates. The RandomForestRegressor model demonstrated strong performance, achieving a high R² score and low MSE, indicating its effectiveness in capturing the underlying patterns in the data.
The Streamlit application simplifies the user experience, allowing individuals to easily input their details and receive instant salary predictions. This deployment showcases the practical application of machine learning in real-world scenarios.
This project serves as a solid foundation for further development and refinement in salary prediction applications.

Refer this [project](https://github.com/monishab2001/Developer-Survey-Analysis-and-Visualization-using-Python) for a detailed **data wrangling, exploratory data analysis (EDA) and visualization**. Also includes visualization in **Dashboard** built in **PowerBI**








.

