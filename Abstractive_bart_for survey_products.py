import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import pandas as pd
from rouge import Rouge
#import seaborn as sns


#from concurrent.futures import ThreadPoolExecutor
#import evaluate
#rouge = evaluate.load('rouge')

# Load pre-trained BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

df=pd.read_csv('top_products.csv')


#prod = df[df['asin'] == 'B004MMEHOM']

#prod = df[df['asin'] == 'B000FX81V2']

#prod = df[df['asin'] == 'B00H1B5E3E'] 

#prod = df[df['asin'] == 'B00T62RK1U']

#prod = df[df['asin'] == 'B002IVHQ5Q']



# Encode input text
all_reviews=""
all_reviews = " ".join(prod['reviewText'].astype(str))
inputs = tokenizer(all_reviews, return_tensors="pt", max_length = 500, truncation=True )

# Generate summary using BART model
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=500, early_stopping=True)

# Decode the generated summary
summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary_text)



results = pd.read_csv("bart_results.csv")
columns = ['prod', 'reviews', 'summary', 'rouge_1','rouge_2','rouge_l' ]
results = results[columns]
###################
# Define the batch size
batch_size = 1500
len(ids)
ids = df['asin'].unique()
# Split ids into batches
id_batches = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]
len(id_batches) #22

one_at_a_time = id_batches[7] #ja pronto para proximo
sys.setrecursionlimit(1500)


for id in one_at_a_time:
    all_reviews=""
    all_reviews = " ".join(df.loc[df['asin'] == id, 'reviewText'].astype(str))
    inputs = tokenizer(all_reviews, return_tensors="pt", max_length = 500, truncation=True )
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=500, early_stopping=True)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    rouge = Rouge()
    rouge_metrics = rouge.get_scores(summary_text, all_reviews)
    rouge1 = rouge_metrics[0]['rouge-1']['f']
    rouge2 = rouge_metrics[0]['rouge-2']['f']
    rougel = rouge_metrics[0]['rouge-l']['f']
    new_row = {'prod': id, 'reviews': all_reviews, 'summary': summary_text, 'rouge_1': rouge1, 
               'rouge_2': rouge2, 'rouge_l': rougel}
    results.loc[len(results)] = new_row

results.to_csv('bart_results.csv') 



import matplotlib.pyplot as plt

# Plotting distributions using KDE plots
plt.plot(results['prod'],results['rouge_1'], label='Rouge 1')
plt.plot(results['prod'],results['rouge_2'], label='Rouge 2')
plt.plot(results['prod'],results['rouge_l'], label='Rouge L')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Distributions of Three Columns')
plt.legend()

# Show plot
plt.show()


#results for analysis in excel because pretty colors and same graph style
cols = ['prod', 'rouge_1','rouge_2','rouge_l' ]
small_results = results[cols]

small_results.to_csv('bart_small_results.csv') 