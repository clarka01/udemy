# f(x) = x^2
# f'(x) = 2x
# evaluate at x = 5
h = 1e-10
x = 5
print("Estimate:", ((x + h)**2 - x**2) / h)
print("True:", 2*x)

# f(x) = x^3
# f'(x) = 3x^2
# evaluate at x = 2
h = 1e-10
x = 2
print("Estimate:", ((x + h)**3 - x**3) / h)
print("True:", 3 * x**2)

# f(x) = 1 / x
# f'(x) = -1 / x^2
# evaluate at x = 1
h = 1e-10
x = 1
print("Estimate:", (1 / (x + h) - 1 / x) / h)
print("True:", -1 / x**2)

import math
# f(x) = sqrt(x)
# f'(x) = 0.5 / sqrt(x)
# evaluate at x = 3
h = 1e-10
x = 3
print("Estimate:", (math.sqrt(x + h) - math.sqrt(x)) / h)
print("True:", 0.5 / math.sqrt(x))

# Exercise: try the 2-sided method






