import matplotlib.pyplot as plt

# Example training and validation accuracy values (replace with your actual values)
train_accuracy = [0.6, 0.7, 0.8, 0.85, 0.9]
val_accuracy = [0.5, 0.65, 0.75, 0.8, 0.85]
epochs = range(1, len(train_accuracy) + 1)

# Plot training vs validation accuracy
plt.plot(epochs, train_accuracy, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.show()
