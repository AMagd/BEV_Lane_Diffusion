import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Creating a highly random array with multiple normal distributions
data1 = np.concatenate([np.random.normal(loc=np.random.randint(np.random.randint(-100, 0, 1),np.random.randint(1, 100, 1)), scale=np.random.randint(1,3), size=1000) for _ in range(55)])

# Creating a less random array with a single normal distribution
data2 = np.concatenate([np.random.normal(loc=np.random.randint(0,50), scale=np.random.randint(1,3), size=1000) for _ in range(5)])

# Creating the KDEs
kde1 = gaussian_kde(data1)
kde2 = gaussian_kde(data2)

# Values for the x-axis
x_values = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 1000)

# Creating the plot
plt.figure(figsize=(14, 6))

# Adding the first KDE plot
plt.subplot(1, 2, 1)
plt.plot(x_values, kde1(x_values), color='green')
plt.fill_between(x_values, kde1(x_values), color='green', alpha=0.5)
plt.title('Pedestrian Crossing Distribution')
plt.xlabel('Values')
plt.ylabel('Density')

# Adding the second KDE plot
plt.subplot(1, 2, 2)
plt.plot(x_values, kde2(x_values), color='green')
plt.fill_between(x_values, kde2(x_values), color='green', alpha=0.5)
plt.title('Pedestrian Crossing Distribution Conditioned on Current Scene')
plt.xlabel('Values')
plt.ylabel('Density')

# Setting the same y limit for both subplots
max_y = max(max(kde1(x_values)), max(kde2(x_values)))
max_y = 0.1
plt.subplot(1, 2, 1).set_ylim([0, max_y])
plt.subplot(1, 2, 2).set_ylim([0, max_y])

# Show the plot
plt.show()
