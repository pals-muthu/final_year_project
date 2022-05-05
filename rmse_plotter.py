import numpy as np
import matplotlib.pyplot as plt

# Define Data

model = ['Decision Tree', 'Random Forest', 'Linear SVM', 'KNN']
y = [70.646, 70.296, 64.141, 66.729]
train = [14.090, 14.246, 16.186, 15.665]
test = [14.426, 14.530, 16.247, 15.990]

x_axis = np.arange(len(model))

# Multi bar Chart

# plt.bar(x_axis - 0.2, train, width=0.4, label='Train')
# plt.bar(x_axis + 0.2, test, width=0.4, label='Test')

# Xticks

# plt.xticks(x_axis, model)

# Add legend

# plt.legend()
plt.bar(model, y, width=0.4)
# plt.show()
# Display
plt.title("Accuracy for different models")
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()
