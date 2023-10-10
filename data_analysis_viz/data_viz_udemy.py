#%%
import pandas as pd

import seaborn as sns

iris = sns.load_dataset('iris')
iris.head()

#%%
# series methods
type(iris.sepal_length)

iris.sum(numeric_only=True)

names = iris.species
names.shape
names.index()

mins = iris.min(numeric_only=True)

mins.index

#%%

iris.sepal_length.nlargest(5, keep='all')

# sorts the columns based how columns are ordered
iris.nsmallest(10, ['sepal_length', 'sepal_width'])


print(iris.species.describe())

print(iris.sepal_length.describe())

#counts nan
iris.species.nunique(dropna=False)


#%%

# evens = [a for a in iris.sepal_length if a % 2 ==1]

#%%

iris.sepal_length.value_counts(ascending=True, bins=10)


#%%

iris[['sepal_length','sepal_width']].plot(kind='scatter', x='sepal_length', y='sepal_width')





# ratio = {l,w for l,w in dict({iris.sepal_length:iris.sepal_width})}

#%%

# section 46: Indexes & Columns:

dfp = sns.load_dataset('planets')

dfp

