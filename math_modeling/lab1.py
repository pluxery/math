import math as m


def function(x: float, y: float) -> float:
    return x / 3 + 2 * y


def exact_solve(x: float) -> float:
    return m.exp(2 * x) / 12 - x / 6 - 1 / 12


def remainder(y):
    return [abs(y_point - y_exact) for y_point, y_exact in zip(y, EXACT_SOLVE)]


class BaseStrategy:
    name = 'Не определенно'

    @staticmethod
    def solve(x: float, y: float) -> float:
        raise 'child must implement!'


class EulerStrategy(BaseStrategy):
    name = 'Метод Эйлера'

    @staticmethod
    def solve(x: float, y: float) -> float:
        return y + STEP * function(x, y)


class ImprovedEulerStrategy(BaseStrategy):
    name = 'Улучшенный Метод Эйлера'

    @staticmethod
    def solve(x: float, y: float) -> float:
        return y + (function(x, y) + function(x + STEP, EulerStrategy.solve(x, y))) * STEP * 0.5


class RungeKuttaStrategy(BaseStrategy):
    name = 'Метод Рунге-Кутта'

    @staticmethod
    def solve(x: float, y: float) -> float:
        k1 = STEP * function(x, y)
        k2 = STEP * function(x + 0.5 * STEP, y + 0.5 * k1)
        k3 = STEP * function(x + 0.5 * STEP, y + 0.5 * k2)
        k4 = STEP * function(x + STEP, y + k3)
        return y + (1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class DifferentialEquationSolver:
    def __init__(self, strategy):
        self.strategy = strategy

    def __solve(self, x: float, y: float) -> float:
        return self.strategy.system(x, y)

    def __compute(self):
        y = [START_SOLVE, ]
        x = BEGIN
        for i in range(COUNT_STEPS):
            y.append(self.strategy.system(x, y[i]))
            x += STEP
        return y

    def print_result(self):

        y = self.__compute()
        rn = remainder(y)

        pad_x = max(len(str(element)) for element in X)
        pad_y_ = max(len(str(element)) for element in EXACT_SOLVE)
        pad_y = max(len(str(element)) for element in y)

        line = '-' * 2 * (pad_x + pad_y + pad_y_)
        centered = ' ' * int(len(line) / 2 - len(self.strategy.name))

        print(line, centered + self.strategy.name, line, sep='\n')

        print(f"{'xi':<{pad_x}}  {'yi*':<{pad_y_}} "
              f" {'yi':<{pad_y}} {'rn'}")

        print(line)

        for i in range(len(X)):
            print(f"{X[i]:<{pad_x}} | {EXACT_SOLVE[i]:<{pad_y_}} | "
                  f"{y[i]:<{pad_y}} | {rn[i]}")


BEGIN, END = 0, 1
STEP = 0.1
COUNT_STEPS = int((END - BEGIN) / STEP)

START_SOLVE = 0
X = [BEGIN + i * STEP for i in range(COUNT_STEPS + 1)]
EXACT_SOLVE = [exact_solve(x) for x in X]

solver = DifferentialEquationSolver(EulerStrategy)
solver.print_result()

solver = DifferentialEquationSolver(ImprovedEulerStrategy)
solver.print_result()

solver = DifferentialEquationSolver(RungeKuttaStrategy)
solver.print_result()
