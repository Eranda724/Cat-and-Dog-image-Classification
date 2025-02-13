from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split

# Define hyperparameter grid
param_grid = {
    'learning_rate': [1e-3, 1e-4],
    'batch_size': [16, 32, 64],
    'num_layers': [3, 4, 5],
    'hidden_size': [32, 64, 128],
    'activation': ['relu', 'tanh']
}

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

best_accuracy = 0
best_params = None

# Perform grid search
for params in ParameterGrid(param_grid):
    layers = [X_train.shape[1]] + [params['hidden_size']] * params['num_layers'] + [1]
    model = NeuralNetwork(layers=layers,
                          activation=params['activation'],
                          learning_rate=params['learning_rate'])
    model.train(X_train, y_train, X_val, y_val, batch_size=params['batch_size'], epochs=10)
    val_accuracy = model.val_accuracy[-1]
    
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = params

print(f"Best Validation Accuracy: {best_accuracy:.4f}")
print(f"Best Parameters: {best_params}")