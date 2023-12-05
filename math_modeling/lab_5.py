from typing import Final
import numpy as np
from sympy import symbols, integrate
import matplotlib.pyplot as plt

POW_NUMBERS: Final[list[int]] = [1, 3, 5]

DIMENSION: Final[int] = 3
BEGIN: Final[int] = 0
END: Final[int] = 1
COUNT: Final[int] = 11

C_SOLUTION = np.linalg.solve(np.array([
    [.7, -.15, -.1],
    [.009, 1.006, .0045],
    [-.000108, -.000081, .9999352]
]), [.3, -.009, .000108])


def u(x_point):
    return x_point + sum([C_SOLUTION[i] * x_point ** POW_NUMBERS[i] for i in range(DIMENSION)])


X = np.linspace(BEGIN, END, COUNT)
Y = [u(x) for x in X]

s_char = symbols('s')
x_char = symbols('x')
alpha = [x_char ** POW_NUMBERS[i] for i in range(DIMENSION)]
beta = [.6, -.036 * s_char ** 2, .0000648 * s_char ** 4]

integrate_formula = u(s_char) * sum([alpha[i] * beta[i] for i in range(DIMENSION)])
integrate_result_formula = integrate(integrate_formula, (s_char, BEGIN, END))


def integrate_result_function(x_point: float) -> float:
    return 1.52969448455074e-5 * x_point ** 5 - 0.0127568840284406 * x_point ** 3 + 0.425859663343685 * x_point


_y = [integrate_result_function(x_point) for x_point in X]

rn_y = [abs(y1 - y2) for y1, y2 in zip(Y, _y)]
Rn = [abs(x_point - r) for x_point, r in zip(X, rn_y)]

print(C_SOLUTION)
print(integrate_result_formula)
print(Rn)

plt.plot(X, Rn)
plt.show()
