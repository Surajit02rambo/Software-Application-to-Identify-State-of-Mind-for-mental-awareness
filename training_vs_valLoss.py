import matplotlib.pyplot as plt

# Example training and validation loss values (replace with your actual values)
train_loss = [0.5, 0.4, 0.3, 0.25, 0.2]
val_loss = [0.6, 0.5, 0.4, 0.35, 0.3]
epochs = range(1, len(train_loss) + 1)

# Plot training vs validation loss
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.show()
