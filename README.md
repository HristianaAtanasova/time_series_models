# Time Series Model for Airbnb Price Prediction

This project analyzes and models Airbnb listing prices using scraped data from Airbnb.com for properties. The dataset includes property details, historical prices, and availability over a two-year window. The goal is to perform exploratory data analysis (EDA), select relevant features, and develop a model to predict nightly listing prices.

# The project uses two CSV files:

properties.csv: Contains property-specific information such as number of bedrooms, bathrooms, max guests, property type, amenities, review score, and average nightly rate.

prices.csv: Includes daily price and availability data for each property over a two-year period.


# Steps in the Project

Exploratory Data Analysis (EDA): 

- Understanding distributions of key variables

- Identifying missing values and cleaning data
  
- Visualizing price trends and correlations
  
- Feature Selection
  
- Selecting relevant features impacting pricing
  
- Handling categorical variables with encoding
  
- Normalizing numerical features where necessary


# Modeling Approach
- Evaluating regression models (Random Forest, Gradient Boosting, etc.)


# Using time series models to capture price fluctuations
- Tuning hyperparameters for optimal performance


# Model Evaluation
- Defining key performance indicators (KPIs)

- Comparing models based on RMSE, MAE, and R² scores


# How to Run the Project

Clone the repository: git clone https://github.com/HristianaAtanasova/time_series_models.git
cd time_series_models

Install required dependencies: pip install -r requirements.txt

Run the EDA script: python exploratory_data_analysis.py

Train the model: python price_predictions.py

# Results and Insights

The analysis reveals strong correlations between pricing and factors such as property type, number of bedrooms, and occupancy rates.
Seasonal price variations were observed, with peak pricing in summer months.
The final model provides accurate predictions with a reasonable RMSE.

# Future Work

- Incorporating additional features like location-based data

- Exploring deep learning models for improved prediction accuracy

- Developing an interactive dashboard for price trend visualization

# Author

Hristiana Atanasova

For any questions or feedback, please reach out via GitHub issues or email.
