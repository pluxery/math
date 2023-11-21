import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_excel('data.xlsx')
df.fillna('0', inplace=True)

df = df.astype(str)
df = df.map(lambda x: x.replace('\xa0', ''))

df['x4'] = pd.to_numeric(df['x4'], errors='coerce').fillna(0)
average_value = int(df['x4'].mean())
df['x4'].replace(0, average_value, inplace=True)

x_features = ['x1', 'x2', 'x3', 'x4', 'x5']
x_train = df[x_features]
y_train = df['Округ']

n_neighbors = 3
classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
classifier.fit(x_train, y_train)

df_test = df[df['Округ'] == 'dv']
x_test = df_test[x_features]

y_predict = classifier.predict(x_test)
y_real = df_test['Округ']

print(y_predict)
print(f"Точность модели: {accuracy_score(y_real, y_predict)}")


dfx = df[x_features]
n_clusters = 5
clust = KMeans(n_clusters=n_clusters, n_init=200, max_iter=200, random_state=1)
clust_labels = clust.fit_predict(dfx)
df_new = dfx.copy()
df_new["Cluster"] = clust_labels
model = TSNE(random_state=1)
transformed = model.fit_transform(df_new)

plt.title(f'Flattened Graph of {n_clusters} Clusters')
params = dict(
        x=transformed[:, 0],
        y=transformed[:, 1],
        hue=clust_labels,
        style=clust_labels,
        palette="Set1"
    )
sns.scatterplot(**params)
plt.show()


dfn = dfx.copy()
cmap = sns.diverging_palette(230, 20, as_cmap=True)
fig, ax, = plt.subplots(figsize=(12, 12))
mask = np.triu(np.ones_like(dfn.corr(), dtype=bool))
sns.heatmap(round(dfn.corr(), 2),
                    mask=mask,
                    cmap=cmap,
                    square=True,
                    linewidths=.5,
                    center=0,
                    cbar_kws={"shrink": .5},
                    xticklabels=dfn.corr().columns,
                    yticklabels=dfn.corr().columns,
                    annot=True,
                    ax=ax)
plt.show()

