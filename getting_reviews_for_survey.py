import pandas as pd




df=pd.read_csv('top_products.csv') 


prod = df[df['asin'] == 'B004MMEHOM']
random_rows = prod.sample(n=10,random_state=3011)

for index, row in random_rows.iterrows():
    print("Review Text:", row['reviewText'])
    print("Overall:", row['overall'])
    print()


prod = df[df['asin'] == 'B000FX81V2']
random_rows = prod.sample(n=10,random_state=3011)

for index, row in random_rows.iterrows():
    print("Review Text:", row['reviewText'])
    print("Overall:", row['overall'])
    print()

prod = df[df['asin'] == 'B00H1B5E3E'] 
random_rows = prod.sample(n=10,random_state=3011)
for index, row in random_rows.iterrows():
    print("Review Text:", row['reviewText'])
    print("Overall:", row['overall'])
    print()

prod = df[df['asin'] == 'B00T62RK1U']
random_rows = prod.sample(n=10,random_state=3011)
for index, row in random_rows.iterrows():
    print("Review Text:", row['reviewText'])
    print("Overall:", row['overall'])
    print()

prod = df[df['asin'] == 'B002IVHQ5Q']
random_rows = prod.sample(n=10,random_state=3011)
for index, row in random_rows.iterrows():
    print("Review Text:", row['reviewText'])
    print("Overall:", row['overall'])
    print()

