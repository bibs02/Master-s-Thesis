#import relevant libraries
import pandas as pd
import numpy as np

# read data
df=pd.read_csv('compiled_data.csv')

# get frequency counts of ASINs (product identifiers)
asin_counts = df['asin'].value_counts()

# sort ASIN counts in descending order
sorted_counts = asin_counts.sort_values(ascending=False)

# calculate the threshold for the top 5%
threshold_index = int(len(sorted_counts) * 0.05)
threshold_count = sorted_counts.iloc[threshold_index]

# select ASINs with counts greater than or equal to the threshold
top_asins = sorted_counts[sorted_counts >= threshold_count].index.tolist()


selected_rows = df[df['asin'].isin(top_asins)]

selected_rows.to_csv('top_products.csv')

# top_asins now contains the ASINs that appear in the top 5% most frequently

#randomly select 5 of those products
np.random.seed(5445)

survey_ids = np.random.choice(top_asins, 5, False)
survey_ids