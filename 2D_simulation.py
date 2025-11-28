import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

k = 2
w = 2
t = 0
L = 1
#x1 = float(input())
#x2 = float(input())

def calculate_normalisation_constant(k_val, L_val):
  def integrand(x):
    return (np.sin(k_val * x))**2
  integral_result, _ = quad(integrand, -L, L)
  if integral_result == 0:
    # Handle this case if it's possible for the integral to be zero
    # For this specific integrand, it shouldn't be zero for non-zero k_val and L_val
    raise ValueError("Integral for normalization constant is zero.")
  return 1 / (integral_result)**0.5

# Calculate the normalization constant A
A = calculate_normalisation_constant(k, L)

def wave_fn(x,t):
  # Use the calculated A
  psi = A*np.sin(k*x)*np.exp(-1j*w*t)
  return psi

def prob_distribution(x,t):
  return np.abs(wave_fn(x, t))**2

def PDF(x1,x2,t=t):
  def f(x):
    return prob_distribution(x,t)
  integral, error = quad(f,x1,x2)
  return integral

wave_fn(4,6)
x_axis = np.linspace(-1,1,500)
psi_values = wave_fn(x_axis,t)
prob_values = prob_distribution(x_axis,t)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, psi_values.real, label="Re(ψ)")
plt.plot(x_axis, psi_values.imag, label="Im(ψ)")
plt.plot(x_axis, prob_values, label="|ψ|²", linestyle="--")
plt.legend()
plt.title("Wavefunction and Probability Distribution")
plt.xlabel("x")
plt.ylabel("Amplitude / Probability")

# Add vertical red lines at x1 and x2 (commented out as x1, x2 inputs are commented)
# plt.axvline(x1, color='red', linestyle='--', label='x1, x2 limits')
# plt.axvline(x2, color='red', linestyle='--')

# Highlight area under |ψ|² between x1 and x2 (commented out as x1, x2 inputs are commented)
# plt.fill_between(x_axis, 0, prob_values, where=(x_axis>=x1) & (x_axis<=x2),
#                  color='red', alpha=0.2, label='Integrated region')

plt.grid(True)
plt.show()

#PDF(x1,x2) # commented out as x1, x2 inputs are commented

def random_walk():
  position = 0
  positions = []
  delta = 0.05
  for i in range(1000):
    A1 = PDF(-1, position)
    A2 = PDF(position, 1)
    sum_A = A1 + A2
    if sum_A == 0:
      Pleft = 0.5 # Default to equal probability if integral is zero
    else:
      Pleft = A1 / sum_A

    random_step = np.random.uniform(0,1)

    if random_step > Pleft:
      position += delta
    else:
      position -= delta
    positions.append(position)
  return np.array(positions)


positions = random_walk()

plt.figure(figsize=(10, 5))
plt.plot(positions, color='blue')
plt.title("Random Walk Biased by Quantum Probability")
plt.xlabel("Step number")
plt.ylabel("Position")
plt.grid(True)
plt.show()
