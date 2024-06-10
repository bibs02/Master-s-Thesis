import numpy as np
import pandas as pd
import nltk
#nltk.download('punkt') 
#nltk.download('stopwords')
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from rouge import Rouge
import matplotlib.pyplot as plt
from tqdm import tqdm

#word embeddings
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

# function to remove stopwords
stop_words = stopwords.words('english')
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

#read_data
df=pd.read_csv('top_products.csv') 

prod = df[df['asin'] == 'B004MMEHOM']

prod = df[df['asin'] == 'B000FX81V2']

prod = df[df['asin'] == 'B00H1B5E3E'] 

prod = df[df['asin'] == 'B00T62RK1U']

prod = df[df['asin'] == 'B002IVHQ5Q']


##################################
# Initialize results DataFrame
results = pd.DataFrame(columns=['prod', 'reviews', 'summary', 'rouge_1', 'rouge_2', 'rouge_l'])
columns=['prod', 'reviews', 'summary', 'rouge_1', 'rouge_2', 'rouge_l']
results = pd.read_csv("textrank_results.csv")
results = results[columns]


ids = df['asin'].unique()
len(ids)
batch_size = 100

# Split ids into batches
id_batches = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]
len(id_batches) #22

one_at_a_time = id_batches[107] 
one_at_a_time[15]

def process_product(id, df, word_embeddings):
    all_reviews = ""
    all_reviews = " ".join(df.loc[df['asin'] == id, 'reviewText'].dropna().astype(str))
    
    # Tokenize sentences and clean them
    sentences = []
    sentences = df.loc[df['asin'] == id, 'reviewText'].dropna().astype(str).apply(sent_tokenize).explode().tolist()
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ", regex=True).str.lower()
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    # Use ThreadPoolExecutor to handle the timeout for computing sentence vectors
    sentence_vectors = [np.mean([word_embeddings.get(w, np.zeros(100)) for w in s], axis=0) 
                            if s else np.zeros(100) for s in clean_sentences]

    sim_mat = cosine_similarity(sentence_vectors)  # Enforce timeout

    # Create graph and compute PageRank
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    
    # Rank sentences
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Extract top 3 sentences as the summary
    summary_text = ' '.join([sentence[1] for sentence in ranked_sentences[:3]])
    
    # Compute ROUGE scores
    rouge = Rouge()
    rouge_metrics = rouge.get_scores(summary_text, all_reviews)
    rouge1 = rouge_metrics[0]['rouge-1']['f']
    rouge2 = rouge_metrics[0]['rouge-2']['f']
    rougel = rouge_metrics[0]['rouge-l']['f']
    return {'prod': id, 'reviews': all_reviews, 'summary': summary_text, 'rouge_1': rouge1, 'rouge_2': rouge2, 'rouge_l': rougel}

# Process each product id


for id in tqdm(one_at_a_time, desc="Processing Products"):
    if id in ['B000DZGN7Q', 'B000NJDRDA', 'B000NJJ1N0', 'B000X05G1A', 'B0011UIPIW', 'B0013ASG3E', 'B0013Q0S4S',
              'B00143JZ08', 'B0014KMDZ0','B0014X7B54','B0015UC17E', 'B0016OI52E' , 'B00178CS4K', 'B000I05TVW',
              'B002JWSNIS', 'B002P6EQPW', 'B00396S1Q2','B004AP92N2', 'B009GDHYPQ', 'B00BCCNZ98', 'B00DQELHBS',
              'B000CITK8S', 'B000ET525K','B000KPU8F2', 'B000W20LKK', 'B0037V0EW8', 'B00480GYSA', 'B000YDDF6O',
              'B000ZKPOTM', 'B001543YEY', 'B000X34GFE','B000YBAKU0','B0012Q2S4W', 'B0013092CS', 'B0013G8OMG',
              'B00167TGII', 'B0017IFSIS','B004X55L9I', 'B00FA2RLX2']:
        new_row = {'prod': id,
                   'reviews': 'all_reviews',
                   'summary': 'summary_text',
                   'rouge_1': 0,
                   'rouge_2': 0,
                   'rouge_l': 0}
    else:
        new_row = process_product(id, df, word_embeddings)
    results.loc[len(results)] = new_row
    



results.to_csv('textrank_results.csv') 
cols = ['prod', 'rouge_1','rouge_2','rouge_l' ]
small_results = results[cols]

small_results.to_csv('textrank_small_results.csv') 

