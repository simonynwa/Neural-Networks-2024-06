from dataprocess.data import load_and_process_data
from models import FullyConnected, Conv, Pooling
from utils import train_model, evaluate

if __name__ == '__main__':
    # Load and process data
    train_features, train_labels = load_and_process_data('train_audio_features.csv')
    test_features, test_labels = load_and_process_data('test_audio_features.csv')
    
    train_features = train_features.reshape(-1, 13, 13, 1)
    test_features = test_features.reshape(-1, 13, 13, 1)

    # Initialize model
    model = Pooling(optimizer='adam', init_method='he_normal')
    
    # Train the model
    train_model(model, train_features, train_labels, epochs=100, batch_size=34, lr=0.008)
    
    # Evaluate on training data
    train_accuracy = evaluate(model, train_features, train_labels, batch_size=32)
    print('Training Accuracy: {:.2f}%'.format(train_accuracy * 100))
    
    # Evaluate on test data
    test_accuracy = evaluate(model, test_features, test_labels, batch_size=32)
    print('Testing Accuracy: {:.2f}%'.format(test_accuracy * 100))
