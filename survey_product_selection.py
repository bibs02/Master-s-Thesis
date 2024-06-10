import pandas as pd
import numpy as np


df=pd.read_csv('compiled_useful_data.csv')

# Step 1: Get frequency counts of ASINs
asin_counts = df['asin'].value_counts()

# Step 2: Sort ASIN counts in descending order
sorted_counts = asin_counts.sort_values(ascending=False)

# Step 3: Calculate the threshold for the top 5%
threshold_index = int(len(sorted_counts) * 0.05)
threshold_count = sorted_counts.iloc[threshold_index]

# Step 4: Select ASINs with counts greater than or equal to the threshold
top_asins = sorted_counts[sorted_counts >= threshold_count].index.tolist()


selected_rows = df[df['asin'].isin(top_asins)]

selected_rows.to_csv('top_products.csv')

# top_asins now contains the ASINs that appear in the top 5% most frequently


np.random.seed(5445)

survey_ids = np.random.choice(top_asins, 5, False)
survey_ids