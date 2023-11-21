import math as m


def F(x, y):
    # return m.sin(2 * x) + y * m.cos(x)
    return x / 3 + 2 * y


def exact_method(x):
    # return m.exp(m.sin(x)) - 2 * (1 + m.sin(x))
    return m.exp(2 * x) / 12 - x / 6 - 1 / 12


def euler(x, y):
    return y + H * F(x, y)


def improved_euler(x, y):
    return y + (F(x, y) + F(x + H, euler(x, y))) * H * 0.5


def runge_kutta(x, y):
    k1 = H * F(x, y)
    k2 = H * F(x + 0.5 * H, y + 0.5 * k1)
    k3 = H * F(x + 0.5 * H, y + 0.5 * k2)
    k4 = H * F(x + H, y + k3)
    return y + (1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def apply_method(method):
    y = [Y0]
    xi = A
    for i in range(N):
        y.append(method(xi, y[i]))
        xi += H
    return y


def print_table(y, rn, method_name):
    if len(X) != len(y) or len(X) != len(Y_) or len(X) != len(rn):
        print(f"x={len(X)}, y={len(y)}, y_={len(Y_)}, rn={len(rn)}")
        raise "array len not equals"

    pad_x = max(len(str(element)) for element in X)
    pad_y_ = max(len(str(element)) for element in Y_)
    pad_y = max(len(str(element)) for element in y)
    pad_rn = max(len(str(element)) for element in rn)

    line = '-' * (pad_x + pad_y + pad_y_ + pad_rn)
    centered = ' ' * int(len(line) / 2 - len(method_name))

    print(line, centered + method_name, line, sep='\n')

    print(f"{'xi':<{pad_x}}  {'yi*':<{pad_y_}} "
          f" {'yi':<{pad_y}} {'rn':<{pad_rn}}")

    print(line)

    for i in range(len(X)):
        print(f"{X[i]:<{pad_x}} | {Y_[i]:<{pad_y_}} | "
              f"{y[i]:<{pad_y}} | {rn[i]:<{pad_rn}}")


def Rn(y):
    return [abs(_y - y_exact) for _y, y_exact in zip(y, Y_)]


A, B = 0, 1
H = 0.1
N = int((B - A) / H)

Y0 = 0
X = [A + i * H for i in range(N + 1)]
Y_ = [exact_method(x) for x in X]

solve = apply_method(euler)
print_table(solve, Rn(solve), 'Метод Эйлера')

solve = apply_method(improved_euler)
print_table(solve, Rn(solve), 'Улучшенный Метод Эйлера')

solve = apply_method(runge_kutta)
print_table(solve, Rn(solve), 'Метод Рунге-Кутта')
