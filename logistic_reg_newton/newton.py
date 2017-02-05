import numpy as np
import asyncio
import aiohttp


def sigmoid(x, y):
    return 1 / (1 + np.exp(x * y))


def first_derivative(x, y, theta):
    m = x.shape[0]
    xts = x@theta
    sigs = sigmoid(xts, y)
    sigs_y = sigs * y
    derivative = x.T @ sigs_y
    return -1/m * derivative


def second_derivative(x, y, theta):
    m = x.shape[0]
    xts = x@theta
    sigs = sigmoid(xts, y)
    return 1/m * x.T@np.diag(sigs*(1-sigs))@x


def logistic_regression_newton(x, y, max_iter=100):
    num_samples = x.shape[0]
    intecept = np.ones(num_samples)

    x = np.c_[intecept, x]
    num_features = x.shape[1]

    theta = np.zeros(num_features)
    for i in range(max_iter):
        first = first_derivative(x, y, theta)
        second = second_derivative(x, y, theta)
        theta = theta - np.linalg.inv(second)@first
    return theta


async def fetch(session, url):
    with aiohttp.Timeout(10, loop=session.loop):
        async with session.get(url) as response:
            return await response.text()

async def main(loop):
    async with aiohttp.ClientSession(loop=loop) as session:
        train_x = await fetch(session, 'http://cs229.stanford.edu/ps/ps1/logistic_x.txt')
        train_x = [float(x) for x in train_x.split()]
        train_x = np.array(train_x).reshape(-1, 2)
        train_y = await fetch(session, 'http://cs229.stanford.edu/ps/ps1/logistic_y.txt')
        train_y = [float(x) for x in train_y.split()]
        train_y = np.array(train_y)
    theta = logistic_regression_newton(train_x, train_y)
    print(theta)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))

#
# x = np.arange(6).reshape(3, 2)
# y = np.array([1, 2, 3])
# theta = np.array([2, 3])
#
# first = first_derivative(x, y, theta)
# second = 1/np.linalg.inv(second_derivative(x, y, theta))
#
#
# #theta = logistic_regression_newton(x, y)
#print(theta)