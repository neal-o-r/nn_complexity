import numpy as np
from keras.models import Sequential
from keras.layers import Dense


np.random.seed(123)

def keras_net(n, h):

        model = Sequential()
        model.add(Dense(100, input_dim=n, activation='sigmoid'))
        model.add(Dense(50, activation='sigmoid'))
        model.add(Dense(1))

        model.compile(loss='mse', optimizer='sgd', metrics=['mse'])

        return model


def sigmoid(x, s=1):
        return 1 / (1 + np.exp(- s * x))


def random_weights(n, h, s):
        w1, w2 = np.sign(np.random.randn(n, h)), np.random.randn(h, 1)
        w1[s, :] = 0
        return w1, w2


def net(n, h, m):
        w1, w2 = random_weights(n, h, m)
        signature = 'ij,ki->kj'
        return lambda x, s: np.einsum(signature, w2,
                        sigmoid(1 + np.einsum(signature, w1, x), s))


def psi(x, s=1):

        m = 8
        p = -(2 * m + 1)
        for k in range(-m, m):
                p += sigmoid(x - (4*k - 1), s)
                p += sigmoid((4*k + 1) - x, s)

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
        '''
        e = []
        for i in range(1, 50, 2):

                y = psi(x_i, i/5)
                y = (y - y.mean()) / y.std()

                split = 2 * len(x) // 3
                m.fit(x[:split], y[:split], verbose=0, epochs=50)
                e.append(m.evaluate(x[split:], y[split:])[0])
        '''
