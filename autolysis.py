import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Step 1: Load and Prepare Data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    # Check basic statistics for numerical columns
    description = df.describe().rename_axis("Statistic").reset_index()

    # Format column names to make them more readable
    description.columns = [col.replace("_", " ").title() for col in description.columns]

    # Print the description as a beautifully styled table
    print("\nDataset Description:")
    print(tabulate(description, headers="keys", tablefmt="double_outline", floatfmt=".2f"))
    
    return df

# Step 2: Data Cleaning
def clean_data(df):
    # 1. Check for missing values
    missing_summary = df.isnull().sum()
    missing_percentage = (missing_summary / len(df)) * 100

    # Generate summary of missing values
    missing_info = pd.DataFrame({
        'Missing Count': missing_summary,
        'Missing Percentage (%)': missing_percentage
    })

    # 2. Check for duplicate rows
    duplicate_row_count = df.duplicated().sum()

    # 3. Check for duplicate `book_id`s
    duplicate_book_id_count = df['book_id'].duplicated().sum()

    # 4. Check for invalid `original_publication_year`
    invalid_year_count = 0
    if 'original_publication_year' in df.columns:
        invalid_year_count = (df['original_publication_year'] < 0).sum()

    # 5. Final Summary Report
    print("\n=== Data Cleaning Summary ===")
    print(f"Total rows in dataset: {len(df)}")
    print(f"\n1. Missing Values Summary:\n{missing_info[missing_info['Missing Count'] > 0].to_string()}")
    print(f"\n2. Duplicate Rows: {duplicate_row_count}")
    print(f"3. Duplicate book_id Count: {duplicate_book_id_count}")
    if 'original_publication_year' in df.columns:
        print(f"4. Invalid Publication Years (Negative): {invalid_year_count}")

    # Add Flags to Dataset (Optional for Further Processing)
    df['missing_flag'] = df.isnull().any(axis=1)
    df['duplicate_row_flag'] = df.duplicated()
    df['duplicate_book_id_flag'] = df['book_id'].duplicated()
    if 'original_publication_year' in df.columns:
        df['invalid_year_flag'] = df['original_publication_year'] < 0

    # Display overall issues summary
    total_flagged_rows = df[['missing_flag', 'duplicate_row_flag', 'duplicate_book_id_flag']].any(axis=1).sum()
    if 'invalid_year_flag' in df.columns:
        total_flagged_rows += df['invalid_year_flag'].sum()

    print(f"\n5. Total Flagged Rows for Review: {total_flagged_rows}")
    print("\n================================")
    
    return df

# Step 3: Visualization and Plotting
def visualize_data(df):
    # 1. Distribution of Average Ratings
    plt.figure(figsize=(10, 6))
    sns.histplot(df['average_rating'], kde=True, bins=30, color='blue')
    plt.title('Distribution of Average Ratings')
    plt.xlabel('Average Rating')
    plt.ylabel('Frequency')
    plt.show()

    # 2. Correlation Heatmap for Numeric Columns
    plt.figure(figsize=(12, 8))
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

    # 3. Boxplot of Ratings Count vs Average Rating
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='ratings_count', y='average_rating', data=df)
    plt.title('Ratings Count vs Average Rating')
    plt.xlabel('Ratings Count')
    plt.ylabel('Average Rating')
    plt.show()

    # 4. Visualizing the Top 10 Books by Ratings Count
    top_books = df.nlargest(10, 'ratings_count')
    plt.figure(figsize=(12, 6))
    sns.barplot(x='ratings_count', y='title', data=top_books, palette='viridis')
    plt.title('Top 10 Books by Ratings Count')
    plt.xlabel('Ratings Count')
    plt.ylabel('Book Title')
    plt.show()

    # 5. Number of Ratings per Rating (1 to 5)
    rating_columns = ['ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']
    rating_data = df[rating_columns].sum()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=rating_data.index, y=rating_data.values, palette='coolwarm')
    plt.title('Distribution of Ratings (1-5)')
    plt.xlabel('Rating Value')
    plt.ylabel('Total Count of Ratings')
    plt.show()

    # 6. Rating Distribution by Language Code
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='language_code', y='average_rating', data=df)
    plt.title('Average Rating Distribution by Language')
    plt.xlabel('Language Code')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)
    plt.show()

    # 7. Scatterplot: Number of Reviews vs Ratings Count
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['ratings_count'], y=df['work_text_reviews_count'], color='green')
    plt.title('Ratings Count vs Text Reviews Count')
    plt.xlabel('Ratings Count')
    plt.ylabel('Text Reviews Count')
    plt.show()

# Step 4: Predict Ratings
def predict_ratings(df):
    # Select relevant columns
    data = df[['title', 'authors', 'language_code', 'books_count', 'original_publication_year',
               'average_rating', 'ratings_count', 'work_ratings_count', 'work_text_reviews_count', 
               'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']]

    # Drop rows with missing values
    data = data.dropna()

    # Normalize ratings distributions
    data['rating_percentage_5'] = data['ratings_5'] / (data['work_ratings_count'] + 1e-9)
    data['rating_percentage_4'] = data['ratings_4'] / (data['work_ratings_count'] + 1e-9)
    data['rating_percentage_3'] = data['ratings_3'] / (data['work_ratings_count'] + 1e-9)
    data['rating_percentage_2'] = data['ratings_2'] / (data['work_ratings_count'] + 1e-9)
    data['rating_percentage_1'] = data['ratings_1'] / (data['work_ratings_count'] + 1e-9)

    # Drop individual rating columns after creating percentages
    data = data.drop(columns=['ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5'])

    # Encode categorical columns
    label_encoders = {}
    for col in ['authors', 'language_code']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le  # Save the encoders for future use

    # Define Features (X) and Target (y)
    X = data.drop(columns=['average_rating', 'title'])
    y = data['average_rating']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Model
    model = RandomForestRegressor(random_state=42, n_estimators=200)
    model.fit(X_train, y_train)

    # Predict Ratings for the Original Books
    titles = data['title']  # Extract the titles
    predicted_ratings = model.predict(X)  # Predict ratings for the existing data

    # Styled Output
    print("\nPredicted Growth in Average Ratings for Books in Dataset:\n")
    print(f"{'Book Title':<50} {'Predicted Rating':>20}")
    print("-" * 70)

    for title, rating in zip(titles, predicted_ratings):
        print(f"{title:<50} {rating:>20.2f}")

# Main Function to Execute the Entire Process
def main():
    file_path = "C:\\Users\\SULAKSH\\Downloads\\goodreads.csv"
    
    # Step 1: Load and Prepare Data
    df = load_and_prepare_data(file_path)

    # Step 2: Data Cleaning
    cleaned_df = clean_data(df)

    # Step 3: Visualization and Plotting
    visualize_data(cleaned_df)

    # Step 4: Predict Ratings
    predict_ratings(cleaned_df)

# Run the main function
if __name__ == "__main__":
    main()




#CODE FOR HAPPINESS.CSV DATASET:-
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Step 1: Description (Descriptive Statistics)
def describe_data(file_path):
    # Read the CSV file with a specified encoding
    df = pd.read_csv(file_path, encoding="ISO-8859-1")

    # Check basic statistics for numerical columns
    description = df.describe().rename_axis("Statistic").reset_index()

    # Format column names to make them more readable
    description.columns = [col.replace("_", " ").title() for col in description.columns]

    # Print the description as a beautifully styled table
    print(tabulate(description, headers="keys", tablefmt="double_outline", floatfmt=".2f"))
    
    return df

# Step 2: Data Cleaning
def clean_data(df):
    # 1. Check for missing values
    missing_summary = df.isnull().sum()
    missing_percentage = (missing_summary / len(df)) * 100

    # Generate summary of missing values
    missing_info = pd.DataFrame({
        'Missing Count': missing_summary,
        'Missing Percentage (%)': missing_percentage
    })

    # 2. Check for duplicate rows
    duplicate_row_count = df.duplicated().sum()

    # 3. Check for invalid `year` entries (e.g., negative years)
    invalid_year_count = 0
    if 'year' in df.columns:
        invalid_year_count = (df['year'] < 0).sum()

    # 4. Final Summary Report
    print("\n=== Data Cleaning Summary ===")
    print(f"Total rows in dataset: {len(df)}")
    print(f"\n1. Missing Values Summary:\n{missing_info[missing_info['Missing Count'] > 0].to_string()}")
    print(f"\n2. Duplicate Rows: {duplicate_row_count}")
    if 'year' in df.columns:
        print(f"3. Invalid Year Entries (Negative): {invalid_year_count}")

    # Add Flags to Dataset (Optional for Further Processing)
    df['missing_flag'] = df.isnull().any(axis=1)
    df['duplicate_row_flag'] = df.duplicated()
    if 'year' in df.columns:
        df['invalid_year_flag'] = df['year'] < 0

    # Display overall issues summary
    total_flagged_rows = df[['missing_flag', 'duplicate_row_flag']].any(axis=1).sum()
    if 'invalid_year_flag' in df.columns:
        total_flagged_rows += df['invalid_year_flag'].sum()

    print(f"\n4. Total Flagged Rows for Review: {total_flagged_rows}")
    print("\n================================")

    return df

# Step 3: Data Visualization
def visualize_data(df):
    # 1. Distribution of Life Ladder (Happiness Score)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Life Ladder'], kde=True, bins=30, color='blue')
    plt.title('Distribution of Life Ladder (Happiness Score)')
    plt.xlabel('Happiness Score')
    plt.ylabel('Frequency')
    plt.show()

    # 2. Correlation Heatmap for Numeric Columns
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

    # 3. Boxplot of Log GDP per Capita vs Life Ladder
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=pd.qcut(df['Log GDP per capita'], q=5, duplicates='drop'), y='Life Ladder', data=df)
    plt.title('Log GDP per Capita vs Life Ladder')
    plt.xlabel('Log GDP per Capita (Quintiles)')
    plt.ylabel('Life Ladder (Happiness Score)')
    plt.xticks(rotation=45)
    plt.show()

    # 4. Visualizing the Top 10 Countries by Healthy Life Expectancy
    top_countries = df.nlargest(10, 'Healthy life expectancy at birth')
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Healthy life expectancy at birth', y='Country name', data=top_countries, palette='viridis')
    plt.title('Top 10 Countries by Healthy Life Expectancy')
    plt.xlabel('Healthy Life Expectancy at Birth')
    plt.ylabel('Country Name')
    plt.show()

    # 5. Average Positive and Negative Affect
    affect_data = df[['Positive affect', 'Negative affect']].mean()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=affect_data.index, y=affect_data.values, palette='coolwarm')
    plt.title('Average Positive and Negative Affect')
    plt.xlabel('Affect Type')
    plt.ylabel('Average Value')
    plt.show()

# Step 4: Predicting Happiness Score
def predict_happiness_score(df):
    # Encode categorical variables if necessary
    if 'Country name' in df.columns:
        encoder = LabelEncoder()
        df['Country name Encoded'] = encoder.fit_transform(df['Country name'])

    # Define feature columns and target variable
    features = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                'Freedom to make life choices', 'Generosity', 'Perceptions of corruption',
                'Positive affect', 'Negative affect']
    target = 'Life Ladder'

    # Drop rows with missing values in selected columns
    data = df[features + [target]].dropna()

    # Split the data into training and testing sets
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Evaluation:\n")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R^2): {r2:.2f}")

    # Predict happiness score for the entire dataset
    df['Predicted_Life_Ladder'] = model.predict(df[features].fillna(0))

    # Display country and predicted happiness score
    display_data = df[['Country name', 'Predicted_Life_Ladder']]
    print("\nPredicted Happiness Scores by Country:\n")
    print(tabulate(display_data, headers=['Country', 'Predicted Happiness Score'], tablefmt='fancy_grid', floatfmt=".2f"))

    # Save the dataset with predictions to a new CSV file
    df.to_csv("happiness_with_predictions.csv", index=False)

    print("\nHappiness score predictions have been added to the dataset and saved as 'happiness_with_predictions.csv'.")
    
    return df

# Main function to call all the steps
def main():
    file_path = "C:\\Users\\SULAKSH\\Downloads\\happiness.csv"  # Replace with actual path

    # Step 1: Describe the data
    df = describe_data(file_path)

    # Step 2: Clean the data
    df = clean_data(df)

    # Step 3: Visualize the data
    visualize_data(df)

    # Step 4: Predict the happiness score
    df = predict_happiness_score(df)

# Run the main function
if __name__ == "__main__":
    main()






#CODE FOR MEDIA.CSV FOR GENERIC DATA ANALYSIS:-


import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

def step1_description(file_path):
    # Read the CSV file with a specified encoding
    df = pd.read_csv(file_path, encoding="ISO-8859-1")

    # Check basic statistics for numerical columns
    description = df.describe().rename_axis("Statistic").reset_index()

    # Format column names to make them more readable
    description.columns = [col.replace("_", " ").title() for col in description.columns]

    # Print the description as a beautifully styled table
    print(tabulate(description, headers="keys", tablefmt="double_outline", floatfmt=".2f"))

    return df

def step2_cleaning(df):
    # 1. Check for missing values
    missing_summary = df.isnull().sum()
    missing_percentage = (missing_summary / len(df)) * 100

    # Generate summary of missing values
    missing_info = pd.DataFrame({
        'Missing Count': missing_summary,
        'Missing Percentage (%)': missing_percentage
    })

    # 2. Check for duplicate rows
    duplicate_row_count = df.duplicated().sum()

    # 3. Check for invalid entries (e.g., negative or nonsensical values for numerical columns)
    invalid_overall_count = (df['overall'] < 0).sum() if 'overall' in df.columns else 0
    invalid_quality_count = (df['quality'] < 0).sum() if 'quality' in df.columns else 0
    invalid_repeatability_count = (df['repeatability'] < 0).sum() if 'repeatability' in df.columns else 0

    # 4. Final Summary Report
    print("\n=== Data Cleaning Summary for media.csv ===")
    print(f"Total rows in dataset: {len(df)}")
    print(f"\n1. Missing Values Summary:\n{missing_info[missing_info['Missing Count'] > 0].to_string()}")
    print(f"\n2. Duplicate Rows: {duplicate_row_count}")
    print(f"3. Invalid Entries:")
    if 'overall' in df.columns:
        print(f"   - Invalid 'overall' Entries (Negative): {invalid_overall_count}")
    if 'quality' in df.columns:
        print(f"   - Invalid 'quality' Entries (Negative): {invalid_quality_count}")
    if 'repeatability' in df.columns:
        print(f"   - Invalid 'repeatability' Entries (Negative): {invalid_repeatability_count}")

    # Add Flags to Dataset (Optional for Further Processing)
    df['missing_flag'] = df.isnull().any(axis=1)
    df['duplicate_row_flag'] = df.duplicated()
    if 'overall' in df.columns:
        df['invalid_overall_flag'] = df['overall'] < 0
    if 'quality' in df.columns:
        df['invalid_quality_flag'] = df['quality'] < 0
    if 'repeatability' in df.columns:
        df['invalid_repeatability_flag'] = df['repeatability'] < 0

    # Display overall issues summary
    total_flagged_rows = df[['missing_flag', 'duplicate_row_flag']].any(axis=1).sum()
    if 'overall' in df.columns:
        total_flagged_rows += df['invalid_overall_flag'].sum()
    if 'quality' in df.columns:
        total_flagged_rows += df['invalid_quality_flag'].sum()
    if 'repeatability' in df.columns:
        total_flagged_rows += df['invalid_repeatability_flag'].sum()

    print(f"\n4. Total Flagged Rows for Review: {total_flagged_rows}")
    print("\n================================")

    return df

def step3_visualization(df):
    # 1. Distribution of Quality Ratings
    if 'quality' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['quality'], kde=True, bins=30, color='purple')
        plt.title('Distribution of Quality Ratings')
        plt.xlabel('Quality Rating')
        plt.ylabel('Frequency')
        plt.show()

    # 2. Correlation Heatmap for Numeric Columns
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap for Numeric Features in Media Dataset')
    plt.show()

    # 3. Boxplot of Repeatability by Language
    if 'repeatability' in df.columns and 'language' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='language', y='repeatability', data=df, palette='Set2')
        plt.title('Repeatability by Language')
        plt.xlabel('Language')
        plt.ylabel('Repeatability')
        plt.xticks(rotation=45)
        plt.show()

    # 4. Top 10 Media Entries by Overall Rating
    if 'overall' in df.columns:
        top_media = df.nlargest(10, 'overall')
        plt.figure(figsize=(12, 6))
        sns.barplot(x='overall', y='title', data=top_media, palette='magma')
        plt.title('Top 10 Media Entries by Overall Rating')
        plt.xlabel('Overall Rating')
        plt.ylabel('Title')
        plt.show()

    # 5. Average Quality, Repeatability, and Overall Scores
    if {'quality', 'repeatability', 'overall'}.issubset(df.columns):
        avg_scores = df[['quality', 'repeatability', 'overall']].mean()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=avg_scores.index, y=avg_scores.values, palette='coolwarm')
        plt.title('Average Quality, Repeatability, and Overall Scores')
        plt.xlabel('Metric')
        plt.ylabel('Average Value')
        plt.show()

# Example Usage
# file_path = "C:\\Users\\SULAKSH\\Downloads\\media.csv"
# df = step1_description(file_path)
# df = step2_cleaning(df)
# step3_visualization(df)

