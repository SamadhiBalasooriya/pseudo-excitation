import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# probabilitstic parameters
mean_pace = 2 #Hz  2005 pachi
pace_COV = 0.1
mean_alpha = 0.41 * (mean_pace - 0.95)
mean_mass= 70

# Calculate standard deviation
std_dev = pace_COV * mean_pace

# Define the normal distribution with the calculated mean and std_dev
normal_dist = norm(loc=mean_pace, scale=std_dev)

# Plot the PDF
x = np.linspace(mean_pace - 3*std_dev, mean_pace + 3*std_dev, 100)
pdf = normal_dist.pdf(x)

plt.plot(x, pdf, label=f'N({mean_pace}, {std_dev}^2)')
plt.title("Normal Distribution PDF")
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

psd_modal_force = 1/4*((mean_alpha*mean_mass*9.81)**2)*pdf
plt.plot(x, psd_modal_force, label=f'N({mean_pace}, {std_dev}^2)')
plt.title("power spectral density of modal force")
plt.xlabel('x')
plt.ylabel('PSD')
plt.legend()
plt.grid(True)
plt.show()