from summarizer import TransformerSummarizer
from transformers import XLNetTokenizer
import pandas as pd
from rouge import Rouge
import seaborn as sns
import matplotlib.pyplot as plt


xlnet_model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")


df=pd.read_csv('top_products.csv')



results = pd.read_csv("xlnet_results.csv")
columns = ['prod', 'reviews', 'summary', 'rouge_1','rouge_2','rouge_l' ]
#results = pd.DataFrame(columns=columns)
results = results[columns]

ids = df['asin'].unique()
len(ids)
batch_size = 1000


# Split ids into batches
id_batches = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]
len(id_batches) #22

one_at_a_time = id_batches[10] #ja pronto par o proximo
sys.setrecursionlimit(1500)


import warnings

warnings.filterwarnings("ignore")

for id in one_at_a_time:
    all_reviews = " ".join(df.loc[df['asin'] == id, 'reviewText'].astype(str))

    if not all_reviews.strip():
        print(f"No reviews found for id: {id}")
        continue

    try:
        summary_text = ''.join(xlnet_model(all_reviews, min_length=105, max_length=110))
    except Exception as e:
        print(f"Error generating summary for id: {id} - {e}")
        continue

    if not summary_text.strip():
        print(f"Generated summary is empty for id: {id}")
        continue

    rouge = Rouge()
    
    try:
        rouge_metrics = rouge.get_scores(summary_text, all_reviews)
    except ValueError as e:
        print(f"Error calculating ROUGE score for id: {id} - {e}")
        continue

    rouge1 = rouge_metrics[0]['rouge-1']['f']
    rouge2 = rouge_metrics[0]['rouge-2']['f']
    rougel = rouge_metrics[0]['rouge-l']['f']

    new_row = {
        'prod': id,
        'reviews': all_reviews,
        'summary': summary_text,
        'rouge_1': rouge1,
        'rouge_2': rouge2,
        'rouge_l': rougel
    }
    results.loc[len(results)] = new_row



results.to_csv('xlnet_results.csv') #current has 9500

#manually add rows for which summaries weren't created 

new_row = {
    'prod': 'B00LXCS0OM',
    'reviews': 'all_reviews',
    'summary': 'summary_text',
    'rouge_1': 0,
    'rouge_2': 0,
    'rouge_l': 0
}
results.loc[len(results)] = new_row



results.to_csv('xlnet_results.csv')
cols = ['prod', 'rouge_1','rouge_2','rouge_l' ]
small_results = results[cols]

small_results.to_csv('xlnet_small_results.csv') 

