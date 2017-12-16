import numpy as np
from keras.models import Sequential
from keras.layers import Dense


np.random.seed(123)

def keras_net(n, h):

        model = Sequential()
        model.add(Dense(8, input_dim=n, activation='sigmoid'))
        model.add(Dense(1))

        model.compile(loss='mse', optimizer='sgd', metrics=['mse'])

        return model


def sigmoid(x, s=1):
        return 1 / (1 + np.exp(- s * x))


def net(n, h, m):
        # 0 * some inputs, the rest are added + or -
        w1 = np.tile(m.astype(int), (8,1)).T
        sign = np.random.choice([1,-1], 8).reshape(1,-1)
        w1 *= sign
        b1 = -1 * sign * 4 * np.arange(-h/2, h/2) - sign
        
        w2 = np.ones((8,1))
        b2 = -2*h - 1

        signature = 'ij,ki->kj'
        return lambda x, s: b2 + np.einsum(signature, w2,
                        sigmoid(b1/s + np.einsum(signature, w1, x), s))


def psi(x, s=1):

        m = 8
        p = -(2 * m + 1)
        for k in range(-m, m+1):
                p += sigmoid(x - (4*k - 1)/s, s)
                p += sigmoid((4*k + 1)/s - x, s)

        return p


if __name__ == '__main__':
 
        n = 1000 # n samples
        n_d = 32 # dimensionality
        n_h = 8  # number of hidden units

        s = np.random.choice([True, False], 32)
        nn = net(n_d, n_h, s)
        x = np.random.multivariate_normal(np.zeros(n_d), np.eye(n_d), n)

        m = keras_net(n_d, n_h)
        x_i = x[:, s].sum(1)

        e = []
        for i in range(1, 50, 2):

                si = i / np.sqrt(n_d)
                y = psi(x_i, si)

                split = 2 * len(x) // 3
                m.fit(x[:split], y[:split], verbose=0, epochs=10)
                e.append(m.evaluate(x[split:], y[split:])[0])
