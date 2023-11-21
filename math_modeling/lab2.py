import numpy as np
import matplotlib.pyplot as plt


def u(x: float) -> float:
    return 1 / (1 + x * x)


def u1(x: float) -> float:
    return 0.5 + (-2839 / 5880) * (x * x - 1) + (711 / 980) * (x ** 3 - x * x)


x = np.linspace(0, 1, 10)
y1 = [u(p) for p in x]
y2 = [u1(p) for p in x]
rn = [abs(p1 - p2) for p1, p2 in zip(y1, y2)]

print(f'Точное решение:\n{y1}')
print(f'Приближенное решение:\n{y2}')
print(f'Погрешность:\n{rn}')

plt.plot(x, rn)
plt.show()
