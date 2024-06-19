import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from dataprocess.data import load_and_process_data

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    # Load and process data
    train_features, train_labels = load_and_process_data('train_audio_features.csv')
    test_features, test_labels = load_and_process_data('test_audio_features.csv')

    # Standardize the data
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    # Apply PCA
    pca = PCA(n_components=0.95)  # 保留解释方差的95%
    train_features_pca = pca.fit_transform(train_features_scaled)
    test_features_pca = pca.transform(test_features_scaled)

    # Initialize KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Train the model on PCA-transformed data
    knn.fit(train_features_pca, train_labels)
    
    # Predict on training data
    train_predictions = knn.predict(train_features_pca)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    print('Training Accuracy (with PCA): {:.2f}%'.format(train_accuracy * 100))
    
    # Predict on test data
    test_predictions = knn.predict(test_features_pca)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print('Testing Accuracy (with PCA): {:.2f}%'.format(test_accuracy * 100))
