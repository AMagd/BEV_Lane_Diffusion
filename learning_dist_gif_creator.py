import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import imageio

def generate_frame(data, x_values, title, ax, ymax):
    kde = gaussian_kde(data)
    ax.clear()
    ax.plot(x_values, kde(x_values), color='green')
    ax.fill_between(x_values, kde(x_values), color='green', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('Values')
    ax.set_ylabel('Density')
    ax.set_ylim([0, ymax])
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

# X values range
x_values = np.linspace(-100, 100, 1000)

fig, ax = plt.subplots()

# Define centers for multiple peaks
centers = [-70, -20, 20, 70]

images = []
for i in range(100):
    # Generate uniform data and normal data
    uniform_data = np.random.uniform(-100, 100, 1000 - i*5)
    normal_data = np.concatenate([np.random.normal(center, 10, i*50) for center in centers])
    # Combine the data
    data = np.concatenate([uniform_data, normal_data])
    img = generate_frame(data, x_values, 'Model Learning - Frame {}'.format(i + 1), ax, 0.02)
    images.append(img)

# Save gif
imageio.mimsave('model_learning.gif', images, duration=100)
