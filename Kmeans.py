import warnings
from dataprocess import load_and_process_data
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    # Load and process data
    train_features, train_labels = load_and_process_data('train_audio_features.csv')
    test_features, test_labels = load_and_process_data('test_audio_features.csv')

    # Initialize PCA
    pca = PCA(n_components=0.95)  # Retain 95% of the variance

    # Standardize the data
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    # Apply PCA on the standardized data
    train_features_pca = pca.fit_transform(train_features_scaled)
    test_features_pca = pca.transform(test_features_scaled)

    # Initialize KMeans model
    num_clusters = len(np.unique(train_labels))  # Assuming the number of clusters is the same as the number of unique labels
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # Fit the model to the PCA-transformed training data
    kmeans.fit(train_features_pca)

    # Predict clusters for the PCA-transformed training data
    train_clusters = kmeans.predict(train_features_pca)

    # Map clusters to actual labels
    cluster_to_label = {}
    for cluster in range(num_clusters):
        mask = (train_clusters == cluster)
        most_common_label = mode(train_labels[mask], axis=None)[0][0]  # mode returns a 2D array, so [0][0] to get the actual mode
        cluster_to_label[cluster] = most_common_label

    # Convert clusters to labels for training data
    train_predictions = np.array([cluster_to_label[cluster] for cluster in train_clusters])

    # Evaluate on training data
    train_accuracy = accuracy_score(train_labels, train_predictions)
    print('Training Accuracy (with PCA): {:.2f}%'.format(train_accuracy * 100))

    # Predict clusters for the PCA-transformed test data
    test_clusters = kmeans.predict(test_features_pca)
    test_predictions = np.array([cluster_to_label[cluster] for cluster in test_clusters])

    # Evaluate on test data
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print('Testing Accuracy (with PCA): {:.2f}%'.format(test_accuracy * 100))
