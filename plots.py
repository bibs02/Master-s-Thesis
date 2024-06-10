import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



textrank = pd.read_csv('textrank_small_results.csv')
bart = pd.read_csv('bart_small_results.csv')
xlnet = pd.read_csv('xlnet_small_results.csv')



#rouge-c-1
sns.set_theme()

sns.kdeplot(textrank['rouge_1'], label='TextRank')
sns.kdeplot(bart['rouge_1'], label='BART')
sns.kdeplot(xlnet['rouge_1'], label='XLNet')

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