from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import minkowski

def compute_cosine_similarity(data):
    similarity = cosine_similarity(data)
    return similarity

def compute_minkowski_distance(x, y, p=3):
    return minkowski(x, y, p)
