import torch
import numpy as np
from numpy import linalg as LA

def normalize_tensor(a: torch.Tensor, dim: int) -> torch.Tensor:
    """
      Normalize the tensor as unit vector
    """
    norm = torch.norm(a, dim=dim, keepdims=True)
    a = a / norm.expand_as(a)

    return a

def normalize_ndarray(a: np.ndarray, axis):
    """
      Normalize the ndarray as unit vector
    """
    norm = LA.norm(a, axis=axis, keepdims=True)
    a = a / np.repeat(norm, a.shape[axis], axis=axis)
    
    return a

def cosine_similarity(cast_feature, cast_name, candidate_feature, candidate_name) -> list:
    """
      Using cosine_similarity to sorting the query priorities.

      Return:
      - result: {'Id': 'Rank'} dicts in list
    """
    result = []

    norm = LA.norm(cast_feature, axis=1, keepdims=True)
    cast_feature = cast_feature / np.repeat(norm, 2048, axis=1)

    norm = LA.norm(candidate_feature, axis=1, keepdims=True)
    candidate_feature = candidate_feature / np.repeat(norm, 2048, axis=1)

    # Return the index with the descending priority 
    # np.argsort() only support ascending sorting
    distance = np.dot(cast_feature, np.transpose(candidate_feature))
    distance = distance * (-1)
    index = np.argsort(distance, axis=1)

    for i in range(index.shape[0]):
        result.append({
            'Id': cast_name[i],
            'Rank': ' '.join(candidate_name[index[i]])
        })

    return result
