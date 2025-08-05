#Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
#Uploading data from local machine (Using Google Colab)
from google.colab import files
uploaded = files.upload()
file_name = next(iter(uploaded))
data = pd.read_csv(file_name)

data.hist(figsize = (16 , 10), bins = 30 , edgecolor = 'black')
plt.suptitle('Distribution of numeric features ' , fontsize = 16)
plt.tight_layout ()
plt.show ()

#Boxplot for all numeric features
numeric_cols = data.select_dtypes (include = 'number').columns
for col in numeric_cols : 
  plt.figure(figsize = (8 , 6))
  sns.boxplot (x = data[col])
  plt.title (f'Boxplot of {col}')
  plt.show ()

#correlation matrix with heatmap
plt.figure (figsize= (10 , 8))
corr = data.corr(numeric_only = True)
sns.heatmap(corr , annot = True  , cmap = 'coolwarm' , fmt = '2f')
plt.title ('Correlation heatmap')
plt.show ()

#Pair plot for numeric features (use only if too many cols)
sns.pairplot(data[numeric_cols])
plt.suptitle('pairwise feature relationships' , y = 1.020)
plt.show ()

#Interactive plotly visual (Example : scatter matrix)
fig = px.scatter_matrix(data[numeric_cols])
fig.update_layout(title = 'Interactive scatter matrix' , width = 1000, height = 800)
fig.show () 

#Finding missing values and data imbalance 
print ('\n Missing values' , data.isnull().sum())
print ('\n Values count of categorical coluns : ')
cat_cols = data.select_dtypes(include = 'object').columns
for col in cat_cols :
  print (f'\n{col} : \n{data[col].value_counts()}')

#Categorical feature analysis
for col in cat_cols :
  plt.figure(figsize=(12,4))
  sns.countplot(x = data[col], palette = 'Set2')
  plt.title(f'count plot - {col}')
  plt.xticks(rotation = 45)
  plt.tight_layout ()
  plt.show ()

#Identify strong correlations
print("\nHighly Correlated Features:")
print(corr[corr > 0.7])
