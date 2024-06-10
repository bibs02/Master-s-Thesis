#import relevant libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#read results data
textrank = pd.read_csv('textrank_small_results.csv')
bart = pd.read_csv('bart_small_results.csv')
xlnet = pd.read_csv('xlnet_small_results.csv')

### average scores
#rouge-c-1
textrank['rouge_1'].mean()
bart['rouge_1'].mean()
textrank['rouge_1'].mean()

#rouge-c-2
textrank['rouge_2'].mean()
bart['rouge_2'].mean()
xlnet['rouge_2'].mean()

#rouge-c-l
textrank['rouge_l'].mean()
bart['rouge_l'].mean()
xlnet['rouge_l'].mean()


### plots
#rouge-c-1
sns.set_theme()

sns.kdeplot(textrank['rouge_1'], label='TextRank')
sns.kdeplot(bart['rouge_1'], label='BART')
sns.kdeplot(xlnet['rouge_1'], label='XLNet')

# Adding labels and legend
plt.xlabel('ROUGE-C-1')
plt.ylabel('Density')
plt.legend()
plt.title('Distribution of ROUGE-C-1 scores')
plt.show()

#rouge-c-2
sns.set_theme()

sns.kdeplot(textrank['rouge_2'], label='TextRank')
sns.kdeplot(bart['rouge_2'], label='BART')
sns.kdeplot(xlnet['rouge_2'], label='XLNet')

# Adding labels and legend
plt.xlabel('ROUGE-C-2')
plt.ylabel('Density')
plt.legend()
plt.title('Distribution of ROUGE-C-2 scores')
plt.show()


#rouge-c-l
sns.set_theme()

sns.kdeplot(textrank['rouge_l'], label='TextRank')
sns.kdeplot(bart['rouge_l'], label='BART')
sns.kdeplot(xlnet['rouge_l'], label='XLNet')

# Adding labels and legend
plt.xlabel('ROUGE-C-L')
plt.ylabel('Density')
plt.legend()
plt.title('Distribution of ROUGE-C-L scores')
plt.show()