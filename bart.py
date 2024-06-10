#import necessary libraries
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import pandas as pd
from rouge import Rouge


# Load pre-trained BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')


#read review data
df=pd.read_csv('top_products.csv')

#create dataframe to store results
columns = ['prod', 'reviews', 'summary', 'rouge_1','rouge_2','rouge_l' ]
results = pd.DataFrame(columns=columns)

#create batches to minimize running time
batch_size = 1500
ids = df['asin'].unique()
# Split ids into batches
id_batches = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]

batch = id_batches[7]
sys.setrecursionlimit(1500)


for id in batch:
    #initialize empty string for review text
    all_reviews=""
    #combine all reviews in one string
    all_reviews = " ".join(df.loc[df['asin'] == id, 'reviewText'].astype(str))
    #tokenize input
    inputs = tokenizer(all_reviews, return_tensors="pt", max_length = 500, truncation=True)
    #use BART for summary generation
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=500, early_stopping=True)
    #decode output
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True) 
    #obtain rouge-c scores
    rouge = Rouge() 
    rouge_metrics = rouge.get_scores(summary_text, all_reviews)
    #extract relevant metrics 
    rouge1 = rouge_metrics[0]['rouge-1']['f'] 
    rouge2 = rouge_metrics[0]['rouge-2']['f']
    rougel = rouge_metrics[0]['rouge-l']['f']
    #create new row
    new_row = {'prod': id, 'reviews': all_reviews, 'summary': summary_text, 'rouge_1': rouge1, 
               'rouge_2': rouge2, 'rouge_l': rougel}
    #add to results data frame
    results.loc[len(results)] = new_row

#store results in csv
results.to_csv('bart_results.csv') 

#create new dataframe with only columns relevant for analysis and plot creation
cols = ['prod', 'rouge_1','rouge_2','rouge_l' ]
small_results = results[cols]

small_results.to_csv('bart_small_results.csv') 