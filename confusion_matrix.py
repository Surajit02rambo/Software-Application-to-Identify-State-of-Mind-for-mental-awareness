from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Example true and predicted labels
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])  # Example true labels (0: Joyful, 1: Melancholic, 2: Neutral)
y_pred = np.array([0, 1, 2, 1, 1, 0, 0, 2, 2])  # Example predicted labels

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['Joyful', 'Melancholic', 'Neutral'], yticklabels=['Joyful', 'Melancholic', 'Neutral'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
