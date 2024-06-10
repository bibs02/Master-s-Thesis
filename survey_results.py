import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#read data
cols = ['product1', 'product2','product3','product4','product5' ]
control = pd.read_csv("survey_control.csv", )
control = control[cols]
textrank = pd.read_csv("survey_textrank.csv")
textrank = textrank[cols]
bart = pd.read_csv("survey_bart.csv")
bart = bart[cols]
xlnet = pd.read_csv("survey_xlnet.csv")
xlnet = xlnet[cols]


####### tests across all products
control_all_products= control.to_numpy().flatten()

textrank_all_products= textrank.to_numpy().flatten()
bart_all_products= bart.to_numpy().flatten()
xlnet_all_products= xlnet.to_numpy().flatten()

#average trust level per status and ATE
avg_control = np.mean(control_all_products)
avg_textrank = np.mean(textrank_all_products)
avg_bart = np.mean(bart_all_products)
avg_xlnet = np.mean(xlnet_all_products)
print('Average trust control:', avg_control )
print('Average trust textrank:', avg_textrank )
print('Average trust bart:', avg_bart  )
print('Average trust xlnet:', avg_xlnet )


ATE_textrank = avg_textrank - avg_control
ATE_bart = avg_bart - avg_control
ATE_xlnet = avg_xlnet - avg_control

print('ATE textrank:', ATE_textrank )
print('ATE bart:', ATE_bart  )
print('ATE xlnet:', ATE_xlnet )



# Perform t-test for textrank
t_statistic, p_value = stats.ttest_ind(textrank_all_products, control_all_products)

print("T-Statistic:", t_statistic)
print("P-Value:", p_value)

# Perform t-test for bart
t_statistic, p_value = stats.ttest_ind(bart_all_products, control_all_products)

print("T-Statistic:", t_statistic)
print("P-Value:", p_value)

# Perform t-test for xlnet
t_statistic, p_value = stats.ttest_ind(xlnet_all_products, control_all_products)

print("T-Statistic:", t_statistic)
print("P-Value:", p_value)

######### plot of trust across all products
sns.set_theme()

sns.kdeplot(textrank_all_products, label='TextRank')
sns.kdeplot(bart_all_products, label='BART')
sns.kdeplot(xlnet_all_products, label='XLNet')
sns.kdeplot(control_all_products, label='Control')



plt.hist([textrank_all_products, bart_all_products, xlnet_all_products,control_all_products ],
         bins=11, 
         label=['TextRank', 'BART', 'XLNet','Control' ])
plt.legend(loc='upper right')
plt.xlabel('Trust')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Trust Levels')

plt.show()




###### tests per product 

# Perform t-test for textrank
avg_control_pp =control.mean()
avg_textrank_pp =textrank.mean()
avg_bart_pp =bart.mean()
avg_xlnet_pp =xlnet.mean()

ATE_textrank_pp = avg_textrank_pp - avg_control_pp
ATE_bart_pp = avg_bart_pp - avg_control_pp
ATE_xlnet_pp = avg_xlnet_pp - avg_control_pp



t_statistic, p_value = stats.ttest_ind(textrank, control)

print("T-Statistic:", t_statistic)
print("P-Value:", p_value)

# Perform t-test for bart
t_statistic, p_value = stats.ttest_ind(bart, control)

print("T-Statistic:", t_statistic)
print("P-Value:", p_value)

# Perform t-test for xlnet
t_statistic, p_value = stats.ttest_ind(xlnet, control)

print("T-Statistic:", t_statistic)
print("P-Value:", p_value)



############# trust plot per product
sns.set_theme()

plt.hist([textrank['product1'], bart['product1'], xlnet['product1'],control['product1']],
         bins=11, 
         label=['TextRank', 'BART', 'XLNet','Control' ])
plt.legend(loc='upper right')
plt.xlabel('Trust')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Trust Levels')

plt.show()

plt.hist([textrank['product2'], bart['product2'], xlnet['product2'],control['product2']],
         bins=11, 
         label=['TextRank', 'BART', 'XLNet','Control' ])
plt.legend(loc='upper right')
plt.xlabel('Trust')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Trust Levels')

plt.show()

plt.hist([textrank['product3'], bart['product3'], xlnet['product3'],control['product3']],
         bins=11, 
         label=['TextRank', 'BART', 'XLNet','Control' ])
plt.legend(loc='upper right')
plt.xlabel('Trust')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Trust Levels')

plt.show()

plt.hist([textrank['product4'], bart['product4'], xlnet['product4'],control['product4']],
         bins=11, 
         label=['TextRank', 'BART', 'XLNet','Control' ])
plt.legend(loc='upper right')
plt.xlabel('Trust')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Trust Levels')

plt.show()

plt.hist([textrank['product5'], bart['product5'], xlnet['product5'],control['product5']],
         bins=11, 
         label=['TextRank', 'BART', 'XLNet','Control' ])
plt.legend(loc='upper right')
plt.xlabel('Trust')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Trust Levels')

plt.show()