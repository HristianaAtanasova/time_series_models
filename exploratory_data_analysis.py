# Import necessary libraries
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
prices_df["property_id"] = prices_df["property_id"].astype(str)

# Merge both datasets on 'property_id'
merged_df = pd.merge(prices_df, properties_df, on='property_id', how='left', indicator=True)

# Convert 'date' to datetime format
merged_df["date"] = pd.to_datetime(merged_df["date"])

# Handle missing values: Drop rows where 'price' or 'adr_avg' is NaN
merged_df.dropna(subset=['price', 'adr_avg'], inplace=True)

print('Cleaning & Merging done.')

# Display cleaned data summary
from IPython.display import display

cleaned_summary = merged_df.describe()
print('Merged data file: ')
display(merged_df)
print('Cleaned summary: ')
cleaned_summary

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# --------------------------------------------------------------------
# Step 2: Exploratory Data Analysis (EDA)
# --------------------------------------------------------------------

# Ensure that the time-based features (year, month, day_of_week) are extracted
merged_df["year"] = merged_df["date"].dt.year
merged_df["month"] = merged_df["date"].dt.month
merged_df["day_of_week"] = merged_df["date"].dt.dayofweek

# Plot price distribution
plt.figure(figsize=(12, 6))
sns.histplot(merged_df['price'], bins=50, kde=True, color='royalblue', edgecolor='black')
plt.title("Distribution of Airbnb Listing Prices", fontsize=16, fontweight='bold')
plt.xlabel("Price (USD)", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("plots/Price_destr.png", dpi=300)
plt.show()

# Availability distribution
availability_counts = merged_df["available"].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=availability_counts.index, y=availability_counts.values, palette=['lightcoral', 'royalblue'])
plt.xlabel("Availability", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.title("Availability Distribution", fontsize=16, fontweight='bold')
plt.xticks([0, 1], ["Not Available", "Available"], fontsize=12)
plt.tight_layout()
plt.savefig("plots/Availability.png", dpi=300)
plt.show()

# Boxplot for price distribution by number of bedrooms
plt.figure(figsize=(12, 6))
sns.boxplot(x=merged_df['bedrooms'], y=merged_df['price'], hue=merged_df['bedrooms'], palette='coolwarm', fliersize=4, legend=False)
plt.title("Price Distribution by Number of Bedrooms", fontsize=16, fontweight='bold')
plt.xlabel("Number of Bedrooms", fontsize=14)
plt.ylabel("Price (USD)", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("plots/Price_distr_by_number_of_bedrooms.png", dpi=300)

plt.show()

# Price distribution by property type
plt.figure(figsize=(12, 6))
property_order = merged_df.groupby('property_type')['price'].median().sort_values().index
sns.boxplot(x=merged_df['property_type'], y=merged_df['price'], hue=merged_df['property_type'], order=property_order, palette='viridis', fliersize=4, legend=False)
plt.xticks(rotation=90, fontsize=12)
plt.title("Price Distribution by Property Type", fontsize=16, fontweight='bold')
plt.xlabel("Property Type", fontsize=14)
plt.ylabel("Price (USD)", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("plots/Price_distr_by_property_type.png", dpi=300)
plt.show()

# Correlation heatmap of numerical features
plt.figure(figsize=(12, 8))
numeric_df = merged_df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask to improve readability
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title("Correlation Heatmap of Features", fontsize=16, fontweight='bold')
plt.savefig("plots/Correlation_heatmap_features.png", dpi=300)
plt.tight_layout()
plt.show()

# Scatter plot of ADR vs. Occupancy Rate
plt.figure(figsize=(10, 6))
sns.scatterplot(x=merged_df['adr_avg'], y=merged_df['occ_rate_avg'], alpha=0.7, color='darkorange')
plt.title("ADR vs. Occupancy Rate", fontsize=16, fontweight='bold')
plt.xlabel("Average Daily Rate (USD)", fontsize=14)
plt.ylabel("Occupancy Rate (%)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("plots/Avg_daily_rate.png", dpi=300)
plt.show()

# Create Monthly Price and Occupancy Rate Trends on the Same Plot
plt.figure(figsize=(12, 6))
monthly_avg_price = merged_df.groupby('month')['price'].mean()
monthly_avg_occ_rate = merged_df.groupby('month')['occ_rate_avg'].mean()

sns.lineplot(x=monthly_avg_price.index, y=monthly_avg_price.values, marker='o', label='Average Price (USD)', color='royalblue')
plt.title("Monthly Price Trends", fontsize=16, fontweight='bold')
plt.xlabel("Month", fontsize=14)
plt.ylabel("Values", fontsize=14)
plt.xticks(np.arange(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("plots/Monthly_price_trends.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(x=monthly_avg_occ_rate.index, y=monthly_avg_occ_rate.values, marker='o', label='Average Occupancy Rate (%)', color='darkorange')
plt.title("Occupancy Rate Trends", fontsize=16, fontweight='bold')
plt.xlabel("Month", fontsize=14)
plt.ylabel("Values", fontsize=14)
plt.xticks(np.arange(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("plots/Occ_rate_trends.png", dpi=300)
plt.show()

# Create Pie Chart for Monthly Price Distribution
price_bins = [0, 100, 200, 300, 400, 500, 1000, 2000]
price_labels = ['0-100', '100-200', '200-300', '300-400', '400-500', '500-1000', '1000+']
merged_df['price_category'] = pd.cut(merged_df['price'], bins=price_bins, labels=price_labels)

monthly_price_dist = merged_df.groupby('month')['price_category'].value_counts().unstack().fillna(0)
plt.figure(figsize=(10, 8))
monthly_price_dist.T.plot(kind='pie', subplots=True, layout=(4, 3), figsize=(12, 12), autopct='%1.1f%%', legend=False, cmap='Set3')
plt.suptitle("Monthly Price Distribution", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("plots/Month_price_destr.png", dpi=300)
plt.show()

# Create Pie Chart for Day of the Week Price Distribution
merged_df['day_of_week'] = merged_df['date'].dt.dayofweek  # Monday=0, Sunday=6
merged_df['day_of_week_category'] = pd.cut(merged_df['price'], bins=price_bins, labels=price_labels)

day_of_week_price_dist = merged_df.groupby('day_of_week')['day_of_week_category'].value_counts().unstack().fillna(0)
plt.figure(figsize=(12, 8))
day_of_week_price_dist.T.plot(kind='pie', subplots=True, layout=(2, 4), figsize=(12, 8), autopct='%1.1f%%', legend=False, cmap='coolwarm')
plt.suptitle("Price Distribution by Day of the Week", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
