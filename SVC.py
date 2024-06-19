from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dataprocess.data import load_and_process_data

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
    
    # Initialize SVM model
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    
    # Train the model on PCA-transformed data
    svm.fit(train_features_pca, train_labels)
    
    # Evaluate on training data
    train_predictions = svm.predict(train_features_pca)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    print('Training Accuracy: {:.2f}%'.format(train_accuracy * 100))
    
    # Evaluate on test data
    test_predictions = svm.predict(test_features_pca)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print('Testing Accuracy: {:.2f}%'.format(test_accuracy * 100))
