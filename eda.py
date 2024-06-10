#import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#read data
df1 = pd.read_json('data\All_Beauty_5.json.gz', lines=True, compression = 'gzip') #85 products 
df2 = pd.read_json('data\Musical_Instruments_5.json.gz', lines=True, compression = 'gzip') #10620 prodcuts 
df3 = pd.read_json('data\Tools_and_Home_Improvement_5.json.gz', lines=True, compression = 'gzip') #73649 products 
df4 = pd.read_json('data\Cell_Phones_and_Accessories_5.json.gz', lines=True, compression = 'gzip') #48186 products 
df5 = pd.read_json('data\Automotive_5.json.gz', lines=True, compression = 'gzip') #79437 products 
df6 = pd.read_json('data\Toys_and_Games_5.json.gz', lines=True, compression = 'gzip') #78772 products 
df7 = pd.read_json('data\Sports_and_Outdoors_5.json.gz', lines=True, compression = 'gzip') #104687 products 

#Word Cloud for all beauty
text = " ".join(str(review) for review in df1.reviewText)
wordcloud = WordCloud(collocations = False, stopwords=STOPWORDS,background_color = 'white').generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Word Cloud for instruments
text = " ".join(str(review) for review in df2.reviewText)
wordcloud = WordCloud(collocations = False, stopwords=STOPWORDS,background_color = 'white').generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Word Cloud for tools and home
text = " ".join(str(review) for review in df3.reviewText)
wordcloud = WordCloud(collocations = False, stopwords=STOPWORDS,background_color = 'white').generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Word Cloud for phone
text = " ".join(str(review) for review in df4.reviewText)
wordcloud = WordCloud(collocations = False, stopwords=STOPWORDS,background_color = 'white').generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Word Cloud for automotive
text = " ".join(str(review) for review in df5.reviewText)
wordcloud = WordCloud(collocations = False, stopwords=STOPWORDS,background_color = 'white').generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Word Cloud for toys
text = " ".join(str(review) for review in df6.reviewText)
wordcloud = WordCloud(collocations = False, stopwords=STOPWORDS,background_color = 'white').generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#Word Cloud for sports
text = " ".join(str(review) for review in df7.reviewText)
wordcloud = WordCloud(collocations = False, stopwords=STOPWORDS,background_color = 'white').generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#combine data in a single dataframe
df = pd.concat([df1, df2, df3,df4, df5,df6,df7])
#df.__len__() #9816359 reviews

df = df.drop_duplicates()
#products = df['asin'].nunique(dropna=True)
#print(products) #395436 products

df_useful = df[['asin','reviewText','vote','overall']]

#keep only those where asin shows up at least fifty times
asin_counts = df_useful['asin'].value_counts()
valid_asins = asin_counts[asin_counts >= 50].index

# Keep only rows with 'asin' values in valid_asins
df_useful= df_useful[df_useful['asin'].isin(valid_asins)]
df_useful['asin'].value_counts()

df_useful .to_csv('compiled_data.csv') 

