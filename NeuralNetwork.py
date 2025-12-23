import numpy as np
import matplotlib.pyplot as plt

# Hardcoded training data (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Hardcoded network: 2 inputs -> 2 hidden -> 1 output
W1 = np.random.randn(2, 2) * 0.5  # weights input->hidden
b1 = np.zeros((1, 2))              # bias hidden
W2 = np.random.randn(2, 1) * 0.5  # weights hidden->output
b2 = np.zeros((1, 1))              # bias output

learning_rate = 1.0
epochs = 50000
losses = []

# Training loop
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = 1 / (1 + np.exp(-z1))  # sigmoid
    
    z2 = np.dot(a1, W2) + b2
    a2 = 1 / (1 + np.exp(-z2))  # sigmoid
    
    # Backward pass
    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)
    
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * a1 * (1 - a1)  # sigmoid derivative
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)
    
    # Update weights
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    # Store loss
    loss = np.mean((a2 - y) ** 2)
    losses.append(loss)
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final predictions
z1 = np.dot(X, W1) + b1
a1 = 1 / (1 + np.exp(-z1))
z2 = np.dot(a1, W2) + b2
predictions = 1 / (1 + np.exp(-z2))

print("\nPredictions:")
print(predictions)
print("\nRounded:", np.round(predictions))
print("\nExpected:", y.flatten())
print("Correct:", np.allclose(np.round(predictions), y, atol=0.1))

# Plot XOR decision boundary
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Predict on grid
z1_grid = np.dot(grid_points, W1) + b1
a1_grid = 1 / (1 + np.exp(-z1_grid))
z2_grid = np.dot(a1_grid, W2) + b2
grid_pred = 1 / (1 + np.exp(-z2_grid))
grid_pred = grid_pred.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, grid_pred, levels=50, cmap='RdYlBu', alpha=0.6)
plt.colorbar(label='Prediction')
plt.contour(xx, yy, grid_pred, levels=[0.5], colors='black', linewidths=2)

# Plot training points
colors = ['red' if label == 0 else 'blue' for label in y.flatten()]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidths=2, zorder=5)
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('XOR Decision Boundary')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.grid(True, alpha=0.3)
plt.show()
