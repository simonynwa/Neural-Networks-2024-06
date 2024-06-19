import numpy as np
from models import FullyConnected, Conv, Pooling
from utils import train_model, evaluate
from dataprocess.data import load_and_process_data
import matplotlib.pyplot as plt

def k_fold_split(X, Y, k):
    fold_size = len(X) // k
    X_folds = []
    Y_folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        X_folds.append(X[start:end])
        Y_folds.append(Y[start:end])
    return X_folds, Y_folds

def random_search_cv(train_features, train_labels, param_grid, n_iter=16, n_splits=2):
    results = []
    X_folds, Y_folds = k_fold_split(train_features, train_labels, n_splits)

    for _ in range(n_iter):

        lr = np.random.uniform(param_grid['lr'][0], param_grid['lr'][1])
        batch_size = np.random.randint(param_grid['batch_size'][0], param_grid['batch_size'][1])
        
        fold_accuracies = []
        
        for i in range(n_splits):
            
            X_val, Y_val = X_folds[i], Y_folds[i]
            X_train = np.concatenate(X_folds[:i] + X_folds[i+1:])
            Y_train = np.concatenate(Y_folds[:i] + Y_folds[i+1:])
            
            
            model = Pooling(optimizer='adam', init_method='he_normal')
            train_model(model, X_train, Y_train, epochs=20, batch_size=batch_size, lr=lr)
            
            val_accuracy = evaluate(model, X_val, Y_val, batch_size=batch_size)
            fold_accuracies.append(val_accuracy)
        
        mean_accuracy = np.mean(fold_accuracies)
        results.append((lr, batch_size, mean_accuracy))

    results.sort(key=lambda x: x[2], reverse=True)
    return results

def plot_results(results):
    lrs = [res[0] for res in results]
    batch_sizes = [res[1] for res in results]
    accuracies = [res[2] for res in results]

    # Scale accuracies to a larger range for better visibility
    min_size = 100
    max_size = 600
    sizes = min_size + (max_size - min_size) * (np.array(accuracies) - min(accuracies)) / (max(accuracies) - min(accuracies))

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(lrs, batch_sizes, c=accuracies, cmap='viridis', s=sizes)
    plt.colorbar(scatter, label='Mean Accuracy')
    plt.xlabel('Learning Rate')
    plt.ylabel('Batch Size')
    plt.title('Random Search CV Results')
    
    # Find the index of the maximum accuracy
    max_idx = accuracies.index(max(accuracies))
    
    # Annotate the point with the highest accuracy, adding offset
    plt.text(lrs[max_idx], batch_sizes[max_idx],
             f'({lrs[max_idx]:.3f}, {batch_sizes[max_idx]})',
             fontsize=12,
             ha='center',
             va='bottom',
             )
    
    plt.show()



train_features, train_labels = load_and_process_data('train_audio_features.csv')
test_features, test_labels = load_and_process_data('test_audio_features.csv')
    
train_features = train_features.reshape(-1, 13, 13, 1)
test_features = train_features.reshape(-1, 13, 13, 1)

param_grid = {
        'lr': [0.001, 0.05],
        'batch_size': [32, 64]
    }

results = random_search_cv(train_features, train_labels, param_grid)

best_lr, best_batch_size, _ = results[0]
print(best_lr, best_batch_size)

plot_results(results)