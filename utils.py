import numpy as np

def compute_loss(output, y):
    m = y.shape[0]
    log_probs = -np.log(output[np.arange(m), y])
    loss = np.sum(log_probs) / m
    return loss

def train_model(model, train_features, train_labels, epochs, batch_size, lr):
    for epoch in range(epochs):
        running_loss = 0.0
        batches = create_batches(train_features, train_labels, batch_size)
        for x_batch, y_batch in batches:
            output = model.forward(x_batch)
            loss = compute_loss(output, y_batch)
            model.backward(y_batch)
            model.update_params(lr)
            running_loss += loss
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / (train_features.shape[0] // batch_size)}")

def evaluate(model, features, labels, batch_size):
    correct = 0
    total = 0
    batches = create_batches(features, labels, batch_size)
    for x_batch, y_batch in batches:
        output = model.forward(x_batch)
        predictions = np.argmax(output, axis=1)
        correct += np.sum(predictions == y_batch)
        total += y_batch.shape[0]
    accuracy = correct / total
    return accuracy

def create_batches(features, labels, batch_size):
    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, features.shape[0] - batch_size + 1, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield features[batch_indices], labels[batch_indices]
