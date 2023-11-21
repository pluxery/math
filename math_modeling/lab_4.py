from typing import Final
import numpy as np


def f(x: float) -> float:
    return 1 / (x * x + 1)


def getA(x: float) -> float:
    return 10 / 3 * (1 - 1 / 5 * x / (x * x + 1) - 1 / 600 * (1 / (x * x + 1)))


def getC(x: float) -> float:
    return 10 / 3 * (1 + 1 / 5 * x / (x * x + 1) - 1 / 600 * (1 / (x * x + 1)))


def getB(x: float) -> float:
    return -20 / 3 + 2 / 180 * 1 / (x * x + 1) - 1 / 30 * 1 / (x * x + 1)


def getF(x: float) -> float:
    return -1 / 10 * 1 / (x * x + 1) ** 2


BEGIN: Final[int] = 0
END: Final[int] = 1
STEP: Final[float] = 0.1
COUNT: Final[int] = int((END - BEGIN) / STEP)

SOLVE_CONDITION: Final[dict] = {
    'A': (-3, 0.1),
    'B': (0, 0.4),
    'C': (3, 0.1),
    'F': (0, 0.3)
}
X = [BEGIN + i * STEP for i in range(COUNT + 1)]

a = [getA(_x) for _x in X]
b = [getB(_x) for _x in X]
c = [getC(_x) for _x in X]
fs = [getF(_x) for _x in X]

first_value = a[0] / SOLVE_CONDITION['A'][0]
last_value = c[10] / SOLVE_CONDITION['C'][1]

b0 = b[0] - SOLVE_CONDITION['B'][0] * first_value
c0 = c[0] - SOLVE_CONDITION['C'][0] * first_value
f0 = fs[0] - SOLVE_CONDITION['F'][0] * first_value

a10 = a[-1] - SOLVE_CONDITION['A'][1] * last_value
b10 = b[-1] - SOLVE_CONDITION['B'][1] * last_value
f10 = fs[-1] - SOLVE_CONDITION['F'][1] * last_value

system = np.array([
    [b0, c0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [a[1], b[1], c[1], 0, 0, 0, 0, 0, 0, 0, 0],
    [0, a[2], b[2], c[2], 0, 0, 0, 0, 0, 0, 0],
    [0, 0, a[3], b[3], c[3], 0, 0, 0, 0, 0, 0],
    [0, 0, 0, a[4], b[4], c[4], 0, 0, 0, 0, 0],
    [0, 0, 0, 0, a[5], b[5], c[5], 0, 0, 0, 0],
    [0, 0, 0, 0, 0, a[6], b[6], c[6], 0, 0, 0],
    [0, 0, 0, 0, 0, 0, a[7], b[7], c[7], 0, 0],
    [0, 0, 0, 0, 0, 0, 0, a[8], b[8], c[8], 0],
    [0, 0, 0, 0, 0, 0, 0, 0, a[9], b[9], c[9]],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, a10, b10],
])

rightPart = np.append([f0], [*fs[1:-1], f10])
answer = np.linalg.solve(system, rightPart)
y = [f(_x) for _x in X]

rn = [np.abs(val1 - val2) for val1, val2 in zip(answer, y)]
print(rn)
