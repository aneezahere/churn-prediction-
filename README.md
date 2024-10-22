Churn Prediction App
-
Welcome to the Churn Prediction App! This app is designed to predict customer churn using various machine learning models. It is built with Python, Streamlit, and various libraries like Scikit-Learn, Pandas, and XGBoost. The app enables users to visualize and predict the likelihood of customers leaving based on historical data.

Features
-
-Data Visualization: Interactive charts and visualizations to explore customer data.

-Model Comparisons: Multiple machine learning models (XGBoost, Random Forest, KNN, etc.) are compared for accuracy and performance.

-Real-Time Predictions: Make churn predictions based on new or existing customer data.

-Interactive User Interface: A simple and sleek user interface built with Streamlit for ease of use.

Deployed App
-
The app is deployed and accessible online: https://churning-prediction.replit.app

Setup Instructions
-
To run the app locally, follow these steps:

git clone https://github.com/aneezahere/churn-prediction-.git

cd churn-prediction-

pip install -r requirements.txt

streamlit run streamlit_app.py

Model Details
-
This app uses multiple machine learning models trained on the customer churn dataset, including:

-Random Forest: A powerful ensemble method.

-XGBoost: Known for high performance with large datasets.

-K-Nearest Neighbors (KNN): A simple yet effective classification algorithm.

-Support Vector Machines (SVM): Effective for high-dimensional data.

-Naive Bayes: Simple and probabilistic classifier.

Each model has been trained and fine-tuned to achieve the best performance on the kaggle dataset.

Technologies Used
-
-Streamlit: For building the web app interface.

-Scikit-learn: For machine learning algorithms.

-Pandas: For data manipulation and analysis.

-Plotly: For data visualization.

-XGBoost: For advanced gradient boosting models.

This app leverages the Groq API to enhance inference speed and efficiency. The models were pre-trained using traditional machine learning libraries like Scikit-Learn and XGBoost, but for deploying and running predictions at scale, the Groq API provides acceleration and optimized performance. This integration ensures fast real-time predictions, making the app suitable for larger datasets and production environments.

License
-
This project is licensed under the MIT License. See the LICENSE file for more details.












