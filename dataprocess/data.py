import pandas as pd
import numpy as np

def load_and_process_data(file_path):
    data = pd.read_csv(file_path)
    data = data.apply(pd.to_numeric, errors='coerce')

    features = data.iloc[:, :-1].values.astype(np.float32)
    labels = data.iloc[:, -1].values.astype(np.int64)  # Assuming labels are integers

    # Normalize features
    features = features / np.max(features)

    return features, labels

def create_batches(features, labels, batch_size):
    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, features.shape[0] - batch_size + 1, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield features[batch_indices], labels[batch_indices]
