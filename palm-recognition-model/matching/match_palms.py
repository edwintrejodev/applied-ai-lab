import numpy as np

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes the cosine similarity between two feature vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return float(dot_product / (norm_a * norm_b))

def is_match(similarity_score: float, threshold: float = 0.75) -> bool:
    """Returns True if the similarity score exceeds the authentication threshold."""
    return similarity_score >= threshold
