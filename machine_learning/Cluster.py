import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from pandas import DataFrame
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, BisectingKMeans
import sklearn.metrics as metrics
from sklearn.metrics import silhouette_samples

warnings.filterwarnings("ignore")


class Cluster:
    @staticmethod
    def fit_visualize_kmeans(dfn: DataFrame, n_clusters: int, show: bool = True) -> DataFrame:
        clust = KMeans(n_clusters=n_clusters, n_init=200, max_iter=200, random_state=1)
        clust_labels = clust.fit_predict(dfn)
        df_new = dfn.copy()
        df_new["Cluster"] = clust_labels
        model = TSNE(random_state=1)
        transformed = model.fit_transform(df_new)

        if show:
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

        return df_new

    @staticmethod
    def normalize_minmax(dfn: DataFrame) -> DataFrame:
        return (dfn - dfn.min()) / (dfn.max() - dfn.min())

    @staticmethod
    def normalize_mean(dfn: DataFrame) -> DataFrame:
        return (dfn - dfn.mean()) / (dfn.std())

    @staticmethod
    def drop_clusters(dfn: DataFrame, clusters: list) -> None:
        for c in clusters:
            dfn.drop(dfn[dfn['Cluster'] == c].index, inplace=True)

    @staticmethod
    def show_snake_plot(dfn: DataFrame) -> None:
        nrm_df = dfn.copy()
        nrm_df['Id'] = dfn.index

        # "Расплавляем" данные в длинный формат
        df_melt = pd.melt(nrm_df.reset_index(),
                          id_vars=['Id', 'Cluster'],
                          value_vars=dfn.columns[0:len(dfn.columns) - 1],
                          var_name='Metric',
                          value_name='Value')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        sns.pointplot(data=df_melt, x='Metric', y='Value', hue='Cluster')
        plt.show()

    @staticmethod
    def show_matrix_corr(dfn: DataFrame) -> None:
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

    @staticmethod
    def show_elbow_method(dfn: DataFrame) -> None:
        K = range(1, 10)
        sum_of_squared_distances = []

        for k in K:
            model = KMeans(n_clusters=k, n_init=200, max_iter=200).fit(dfn)
            sum_of_squared_distances.append(model.inertia_)
        plt.plot(K, sum_of_squared_distances, 'bx-')
        plt.xlabel('K values')
        plt.ylabel('Sum of Squared Distances')
        plt.title('Elbow Method')
        plt.show()

    @staticmethod
    def show_silhouette(dfn: DataFrame) -> None:
        SK = range(2, 10)
        sil_score = []
        for i in SK:
            labels = KMeans(n_clusters=i, init="k-means++", random_state=200).fit(dfn).labels_
            score = metrics.silhouette_score(dfn, labels, metric="euclidean", sample_size=1000, random_state=200)
            sil_score.append(score)
            # print("Silhouette score for k(clusters) = " + str(i) + " is "
            #       + str(
            #     metrics.silhouette_score(df_Short, labels, metric="euclidean", sample_size=1000, random_state=200)))

        sil_centers = pd.DataFrame({'Clusters': SK, 'Sil Score': sil_score})
        sns.lineplot(x='Clusters', y='Sil Score', data=sil_centers, marker="+")

        plt.plot(SK, sil_score, 'bx-')
        plt.xlabel('Clusters')
        plt.ylabel('Silhouette score')
        plt.title('Silhouette Method')
        plt.show()

    @staticmethod
    def show_silhouette_analysis(dfn: DataFrame, n_clusters: int = 10):
        SK = range(2, n_clusters)
        for k in SK:
            fig, (ax1) = plt.subplots(1)
            fig.set_size_inches(7, 7)

            # Run the Kmeans algorithm
            km = KMeans(n_clusters=k)
            labels = km.fit_predict(dfn)

            # Get silhouette samples
            silhouette_vals = silhouette_samples(dfn, labels)

            # Silhouette plot
            y_lower, y_upper = 0, 0
            for i, cluster in enumerate(np.unique(labels)):
                cluster_silhouette_vals = silhouette_vals[labels == cluster]
                cluster_silhouette_vals.sort()
                y_upper += len(cluster_silhouette_vals)
                ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
                ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
                y_lower += len(cluster_silhouette_vals)

            # Get the average silhouette score and plot it
            avg_score = np.mean(silhouette_vals)
            ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
            ax1.set_yticks([])
            ax1.set_xlim([-0.1, 1])
            ax1.set_xlabel('Silhouette coefficient values')
            ax1.set_ylabel('Cluster labels')
            ax1.set_title(f'Silhouette plot for the s clusters {k}', y=1.02);

            plt.tight_layout()
            plt.suptitle(f'Silhouette analysis using k = {k}',
                         fontsize=16, fontweight='semibold', y=1.05)
            plt.show()
