import numpy as np
import matplotlib.pyplot as plt

# Define Data

model = ['Decision Tree', 'Random Forest', 'Linear SVM', 'KNN']
train = [15.535, 15.848, 16.506, 18.174]
test = [16.479, 16.310, 16.718, 18.594]

x_axis = np.arange(len(model))

# Multi bar Chart

plt.bar(x_axis - 0.2, train, width=0.4, label='Train')
plt.bar(x_axis + 0.2, test, width=0.4, label='Test')

# Xticks

plt.xticks(x_axis, model)

# Add legend

plt.legend()

# Display

plt.show()
