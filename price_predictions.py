# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Load the datasets
properties_path = "properties.csv"
prices_path = "prices.csv"

properties_df = pd.read_csv(properties_path)
prices_df = pd.read_csv(prices_path)

# --------------------------------------------------------------------
# Step 1: Data Cleaning & Merging
# --------------------------------------------------------------------
# Convert 'price' and 'adr_avg' to numeric, handling errors
properties_df['adr_avg'] = pd.to_numeric(properties_df['adr_avg'], errors='coerce')
prices_df['price'] = pd.to_numeric(prices_df['price'], errors='coerce')

# Convert 'number_of_reviews' to numeric
properties_df['number_of_reviews'] = pd.to_numeric(properties_df['number_of_reviews'], errors='coerce')

review_columns = [
    'review_score', 'rating_accuracy', 'rating_cleanliness', 'rating_checkin',
    'rating_location', 'rating_value', 'rating_communication'
]

# Impute missing ratings with the median of each review column
for col in review_columns:
    if col in properties_df.columns:
        # Compute the median rating for the column
        median_rating = properties_df[col].median()
        
        # Replace 0.0 ratings with the median for entries with no reviews (number_of_reviews == 0)
        properties_df.loc[(properties_df[col] == 0.0) & (properties_df['number_of_reviews'] == 0), col] = median_rating

# Standardize 'property_id' format by removing 'airbnb_' prefix in prices dataset
prices_df['property_id'] = prices_df['property_id'].str.replace(r'airbnb_', '', regex=True)

# Convert 'property_id' in both datasets to string to prevent mismatches
properties_df["property_id"] = properties_df["property_id"].apply(lambda x: f"{int(float(x))}" if pd.notna(x) else x)
### prices_df["property_id"] = prices_df["property_id"].astype(str) # this line is redundant 

# Merge both datasets on 'property_id'
merged_df = pd.merge(prices_df, properties_df, on='property_id', how='left', indicator=True)

# Convert 'date' to datetime format
merged_df["date"] = pd.to_datetime(merged_df["date"])

# Handle missing values: Drop rows where 'price' or 'adr_avg' is NaN
merged_df.dropna(subset=['price', 'adr_avg'], inplace=True)

# --------------------------------------------------------------------
# Step 3: Feature Engineering 
# --------------------------------------------------------------------
# Extract time-based features
merged_df["year"] = merged_df["date"].dt.year
merged_df["month"] = merged_df["date"].dt.month
merged_df["day_of_week"] = merged_df["date"].dt.dayofweek

# Sine and cosine transformations for cyclical features
merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month'] / 12)
merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['month'] / 12)

merged_df['day_of_week_sin'] = np.sin(2 * np.pi * merged_df['day_of_week'] / 7)
merged_df['day_of_week_cos'] = np.cos(2 * np.pi * merged_df['day_of_week'] / 7)

# Define the categorical columns 
# categorical_columns = ['property_type', 'superhost', 'city'] ### city is redundant because there is only one city...
categorical_columns = ['property_type', 'superhost', 'available'] ### city is redundant because there is only one city...

# Handle missing categorical values by replacing them with 'unknown' category
for col in categorical_columns:
    merged_df[col] = merged_df[col].astype(str).fillna('unknown')

# Compute price deviation as (price - base price)
merged_df["price_deviation"] = merged_df["price"] - merged_df["adr_avg"]

# Define cutoff date: Use the current date as the cutoff date
# This ensures that training data includes only past data and future data is used for predictions.
cutoff_date = datetime.now()  # Current date as the cutoff point

### I am not sure if this splitting is correct. Where does the future data come from? 
# Split into train (past dates only) and future (for prediction)
train_df = merged_df[merged_df["date"] < cutoff_date]
future_df = merged_df[merged_df["date"] >= cutoff_date]  # Reserved for final predictions

# Feature Selection
numeric_df = merged_df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()

# Set correlation threshold
corr_threshold = 0.2

# Select only highly correlated features with price
selected_features = corr_matrix.loc['price', (corr_matrix.loc['price']) > corr_threshold].index.tolist()

# Remove 'price' from feature list if it appears
if 'price' in selected_features:
    selected_features.remove('price')

print("Initial selected features:\n", selected_features)

# Set up the feature matrix and target
features = selected_features + ['year', 'month', 'day_of_week'] + categorical_columns
target = "price_deviation"

print("\nFinal selected features:\n", features)

# Drop rows with missing values in selected features for training
train_df = train_df.dropna(subset=features + [target])

# --------------------------------------------------------------------
# Step 4: Update Feature Selection Pipeline with Encoding and Scaling
# --------------------------------------------------------------------
# Define the preprocessor: 
# - OneHotEncoder for categorical variables
# - StandardScaler for numerical variables (scaling)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), selected_features + ['year', 'month', 'day_of_week']),
        ('cat', OneHotEncoder(), categorical_columns)
    ])
### It was probably wrong to include 'year' into the standard scaler because it distorts its meaning..

# Create a pipeline with preprocessing and model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1))
])

#--------------------------------------------------------------------
# Step 5: Train-test split
#--------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    train_df[features], train_df[target], test_size=0.2, random_state=42
)

# --------------------------------------------------------------------
# Step 6: Hyperparameter Tuning
# --------------------------------------------------------------------
# Define a simple range for hyperparameters
# Chose a basic grid search due to time and computational limitations.
param_grid = {
    'n_estimators': [50, 100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30, 40],  # Maximum depth of the tree
    'min_samples_split': [5, 10, 15, 20]  # Minimum samples required to split a node
}

# Define the categorical columns that need encoding
categorical_columns = ['property_type', 'superhost', 'available']

# Manually perform hyperparameter tuning (simple grid search)
best_model = None
best_score = -np.inf  # Initialize best score

# Loop over the parameter grid and evaluate each combination
for n_estimators in param_grid['n_estimators']:
    for max_depth in param_grid['max_depth']:
        for min_samples_split in param_grid['min_samples_split']:
            # Create the model with the current hyperparameters
            model = RandomForestRegressor(n_estimators=n_estimators, 
                                          max_depth=max_depth, 
                                          min_samples_split=min_samples_split, 
                                          random_state=42, n_jobs=1)
            
            # Create the full pipeline
            model_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            
            # Fit the model
            model_pipeline.fit(X_train, y_train)
            
            # Evaluate the model on the test set
            y_pred = model_pipeline.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            # If this combination gives a better R² score, save the model
            if r2 > best_score:
                best_score = r2
                best_model = model_pipeline

# Display the best hyperparameters and R² score
print(f"Best Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")
print(f"Best R² Score: {best_score:.2f}")

# --------------------------------------------------------------------
# Step 7: Model Evaluation
# --------------------------------------------------------------------
# Use the best model for prediction
y_pred = best_model.predict(X_test)

# Compute evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display evaluation results
evaluation_results = {
    "Mean Absolute Error (MAE)": mae,
    "Mean Squared Error (MSE)": mse,
    "Root Mean Squared Error (RMSE)": rmse,
    "R² Score": r2
}

print("\nModel Evaluation Results:")
for key, value in evaluation_results.items():
    print(f"{key}: {value:.2f}")

# --------------------------------------------------------------------
# Step 8: Feature Importance 
# -------------------------------------------------------------------- 
# Get the feature importances from the trained RandomForestRegressor
model = model_pipeline.named_steps['regressor']
importances = model.feature_importances_

# Get feature names after preprocessing (including categorical features after encoding)
encoded_columns = preprocessor.transformers_[1][1].get_feature_names_out(categorical_columns)
all_feature_names = selected_features + ['year', 'month', 'day_of_week'] + list(encoded_columns)

# Create a DataFrame to display the feature importances
feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importances in the console
print("\nFeature Importance:")
print(feature_importance_df)

# Final Step
# --------------------------------------------------------------------
# Step 9: Future Price Predictions 
# --------------------------------------------------------------------
future_df = future_df.copy()

# Make predictions on future data
future_df["predicted_price_deviation"] = best_model.predict(future_df[features])
future_df["predicted_price"] = future_df["adr_avg"] + future_df["predicted_price_deviation"]

# Display sample future predictions
print("\nFuture Price Predictions:")
print(future_df[["property_id", "date", "adr_avg", "predicted_price"]].head(10))
