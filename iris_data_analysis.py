## Python3
## Statistical Analysis on Iris dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.stats as stats
import plotnine as p9
#import statsmodels as sm



# Load some data
# iris: 3 different species: Setosa, Versicolour, and Virginica

iris = datasets.load_iris()


# transform to dataframe
iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

#print (iris_df.head(5))
#print (iris_df.shape)
#print (iris_df.info())
#print (iris_df.describe())

iris_df['species'] = iris['target']


## Scatter matrix
#pd.scatter_matrix(iris_df, alpha=0.2, figsize=(10, 10))
#plt.show()

print("unique labels in the dataset: {}".format(iris_df.species.unique()))


## If need to re-label the target variable into numerical values
'''
mapping_dict = {0: 1, 1: 2, 2: 3}
iris_df['species'].map(mapping_dict)
print("unique labels in the dataset after mapping: {}".format(iris_df.species.unique()))
print('done mapping')
input()
'''


# Print density plot, mean, median, and mode
print(p9.ggplot(iris_df)+ p9.aes(x='sepal length (cm)')+ p9.geom_density(alpha=0.2))
print(iris_df.mean())
print(iris_df.median())
print(iris_df.mode())


# Calculate theoretical quantiles
tq = stats.probplot(iris_df['sepal length (cm)'], dist="norm")

# Create Dataframe
df_temp = pd.DataFrame(data= {'Theoretical Quantiles': tq[0][0], 
                         "Ordered Values": iris_df['sepal length (cm)'].sort_values() })

# Create Q-Q plot
print(p9.ggplot(df_temp)+ p9.aes('Theoretical Quantiles','Ordered Values') +p9.geom_point())



# Extract data with particular label (target)
#a = iris_df[['sepal length (cm)','species']]
a_sl = iris_df.loc[iris_df['species']== 0,'sepal length (cm)']
a_pl = iris_df.loc[iris_df['species']== 0,'petal length (cm)']


print(iris_df['sepal length (cm)'])

#Extract entire dataframe based on one type of flower
num_species = 0
a_df = iris_df.loc[iris_df['species']== num_species,]


print("dataframe info of species {}".format(num_species))
a_df.info()


## Scatter plot
import matplotlib.pyplot as plt

cdict = {0: 'red', 1: 'blue', 2: 'green'}
fig, axs = plt.subplots(1,2)
for g in iris_df.species.unique():
	a_sl = iris_df.loc[iris_df['species']== g,'sepal length (cm)']
	a_pl = iris_df.loc[iris_df['species']== g,'petal length (cm)']
	axs[0].scatter(a_sl, a_pl, c=cdict[g], label=g, s = 50)
axs[0].set_xlabel("sepal length (cm)")
axs[0].set_ylabel("petal length (cm)")
axs[0].legend()

for g in iris_df.species.unique():
	a_sl = iris_df.loc[iris_df['species']== g,'sepal length (cm)']
	a_sw = iris_df.loc[iris_df['species']== g,'sepal width (cm)']
	axs[1].scatter(a_sl, a_sw, c=cdict[g], label=g, s = 50)
axs[1].set_xlabel("sepal length (cm)")
axs[1].set_ylabel("sepal width (cm)")
axs[1].legend()


## Check to see if there's a linear correlation between...

# Sepal and Petal Legnth
a_sl = iris_df.loc[iris_df['species']== 0,'sepal length (cm)']
a_pl = iris_df.loc[iris_df['species']== 0,'petal length (cm)']

pearcorr_1 = stats.pearsonr(a_sl,a_pl)
print("Flower 0: Correlation between Sepal Length and Petal Length: {}".format(pearcorr_1[0]))
print("p-value is: {}".format(pearcorr_1[1]))

# Sepal Legnth and Width
a_sl = iris_df.loc[iris_df['species']== 0,'sepal length (cm)']
a_sw = iris_df.loc[iris_df['species']== 0,'sepal width (cm)']

pearcorr_2 = stats.pearsonr(a_sl, a_sw)
print("Flower 0: Correlation between Sepal Length and Sepal Width: {}".format(pearcorr_2[0]))
print("p-value is: {}".format(pearcorr_2[1]))



## Classifications:
import numpy as np
from sklearn.model_selection import train_test_split

y = iris_df['species'].values
X = iris_df[['sepal length (cm)','petal length (cm)']].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
	random_state=7, stratify=y)


## Support Vector Classifier
print("Support Vector Classifier:")

from sklearn.svm import SVC
model1 = SVC(kernel='linear', C=0.5, decision_function_shape='ovr')
model1.fit(X_train,y_train)
print('Accuracy: {}'.format(model1.score(X_test,y_test)))

y_pred = model1.predict(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print("Confusion Matrix: \n{}\n".format(confusion_matrix(y_test, y_pred)))
print("Classification Report: \n{}\n".format(classification_report(y_test, y_pred)))

'''output:
Confusion Matrix:
[[10  0  0]
 [ 0 10  0]
 [ 0  1  9]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       0.91      1.00      0.95        10
           2       1.00      0.90      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
'''


## Cross-validation
from sklearn.model_selection import cross_val_score

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(model1,X_train,y_train,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
'''output:
Average 5-Fold CV Score: 0.9666666666666668
'''


## KMeans Clustering
print("KMeans Clustering, using sepal length and petal length:")

points = np.concatenate( (iris_df['sepal length (cm)'].values.reshape(-1,1), 
	iris_df['petal length (cm)'].values.reshape(-1,1)),axis=1 )

from sklearn.cluster import KMeans
model2 = KMeans(n_clusters=3)
model2.fit(points)

# Determine the cluster labels of new_points: labels
labels = model2.predict(points)


# Assign the cluster centers: centroids
centroids = model2.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]


## Scatter plot
import matplotlib.pyplot as plt

fig2, ax2 = plt.subplots()
ax2.scatter(points[:,0],points[:,1],c=iris_df.species.values,alpha=0.5)
ax2.scatter(centroids_x,centroids_y,marker='D',s=100)
ax2.set_xlabel("sepal length (cm)")
ax2.set_ylabel("petal length (cm)")
#ax2.legend()

plt.show()


# Create a DataFrame with labels and iris_df['species'] as columns
df = pd.DataFrame({'labels': labels, 'targets': iris_df['species'] })

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['targets'])
print(ct)

'''output:
targets   0   1   2
labels             
0         0  45  13
1        50   1   0
2         0   4  37
'''

