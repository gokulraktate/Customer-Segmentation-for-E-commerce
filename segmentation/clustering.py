from sklearn.cluster import KMeans
import pandas as pd

def perform_kmeans(data, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = model.fit_predict(data)
    return data, model
