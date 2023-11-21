import numpy as np

h = .1
tau = .5

c = [0, 1]
N = 10


def Nu_n_0(y):
    res = []
    for i in range(1, N):
        if i == 0:
            t = (.5 - np.exp(y[i]) + .5 * np.exp(y[i + 1]))
        else:
            t = (.5 * np.exp(y[i - 1]) - np.exp(y[i]) + .5 * np.exp(y[i + 1]))
        res.append(t)

    return res


def Nu_n_1(y):
    res = []
    for i in range(N):
        if i == 0:
            t = (.5 * np.exp(1) - np.exp(1 + y[i]) + .5 * np.exp(1 + y[i + 1]))
        else:
            t = (.5 * np.exp(1 + y[i - 1]) - np.exp(1 + y[i]) + .5 * np.exp(1 + y[i + 1]))
        res.append(t)

    return res


def Nu_0(y):
    nu_0 = Nu_n_0(y)
    res = []
    for i in range(1, N):
        t = .5 * (np.exp(y[i - 1]) - np.exp(y[i]) + .5 * np.exp(y[i + 1])) + .5 / 4 * .1 * .1 * nu_0[i - 1]
        res.append(t)

    return res


def Nu_1(y):
    nu_1 = Nu_n_1(y)
    res = []
    for i in range(1, N):
        t = .5 * (np.exp(1 + y[i - 1]) - np.exp(1 + y[i]) + .5 * np.exp(1 + y[i + 1])) + .5 / 4 * .1 * .1 * nu_1[i - 1]
        res.append(t)

    return res


def Vi(xi, yi):
    return .5 ** 2 * np.exp(xi + yi)


y = np.linspace(0, 1, N + 1)

nu_0 = Nu_0(y)
nu_1 = Nu_1(y)

print(nu_0)

print(nu_1)
