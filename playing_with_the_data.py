import pandas as pd
import matplotlib.pyplot as plt



data = pd.read_csv('compiled_useful_data.csv')
data.head()



data.__len__()

data.asin.value_counts().plot(kind='bar')
plt.show()    

#keep only those with at least ten reviews