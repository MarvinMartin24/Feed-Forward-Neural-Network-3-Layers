import numpy as np

def sse(y_true, y_pred):
    return 0.5 * np.sum(np.power(y_pred - y_true, 2))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

class FFNN:
    def __init__(self, input_layer, hidden_layer, output_layer):
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.input_layer = input_layer
        self.W1 = np.random.rand(input_layer, hidden_layer)
        self.W2 = np.random.rand(hidden_layer, output_layer)
        self.B1 = np.random.rand(hidden_layer)
        self.B2 = np.random.rand(output_layer)


    def fit(self, X_train, y_train, learning_rate=0.1, epochs=500):
        for epoch in range(epochs):
            error = 0
            for x, y_true in zip(X_train, y_train):
                h1_ = np.dot(x, self.W1) + self.B1
                h1 = tanh(h1_)
                y_ = np.dot(h1, self.W2) + self.B2
                y = tanh(y_)
                error += sse(y_true, y)

                dW1 = np.zeros((self.input_layer, self.hidden_layer))
                dW2 = np.zeros((self.hidden_layer, self.output_layer))
                dB1 = np.zeros(self.hidden_layer)
                dB2 = np.zeros(self.output_layer)

                for k in range(self.output_layer):
                    dB2[k] = (y[k] - y_true[k]) * tanh_prime(y_[k])

                for j in range(self.hidden_layer):
                    for k in range(self.output_layer):
                        dW2[j][k] =  dB2[k] * h1[j]

                for j in range(self.hidden_layer):
                    dB1[j] = sum([dB2[k] * self.W2[j][k] * tanh_prime(h1_[j]) for k in range(self.output_layer)])

                for i in range(self.input_layer):
                    for j in range(self.hidden_layer):
                        dW1[i][j] = dB1[j] * x[i]

                self.B1 -= learning_rate * dB1
                self.W1 -= learning_rate * dW1
                self.B2 -= learning_rate * dB2
                self.W2 -= learning_rate * dW2

            error /= X_train.shape[0]
            print(f"[INFO]: epoch = {epoch + 1} | error = {error}")

    def predict(self, X_test):
        h1_ = np.dot(X_test, self.W1) + self.B1
        h1 = tanh(h1_)
        y_ = np.dot(h1, self.W2) + self.B2
        y = tanh(y_)
        return y

x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
y_train = np.array([[0], [1], [1], [0]])
fnn = FFNN(2, 4, 1)
fnn.fit(x_train, y_train)
print('Input',x_train)
print()
print('Predition', np.around(fnn.predict(x_train)))
print('Label', y_train)
