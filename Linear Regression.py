# In[0]
# import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

# In[1]
slots = [['horsepower', 'Red'],
         ['carwidth', 'Green'],
         ['curbweight', 'Blue'],
         ['enginesize', 'yellow']]

theta = np.zeros(len(slots) + 1)
iterations = int(1e6)
alpha = 0.005

# In[2]
# Scatter plots for feature selection
# Find best 4 of the numerical features
# horsepower, carwidth, curb weight, engine size

# import file with data
data = pd.read_csv(".\\car_data.csv")


def scatter_plot(slot, color):
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(data[slot], data['price'], c=color, label='Training Data')
    axis.legend(loc=2)
    axis.set_xlabel(slot)
    axis.set_ylabel('Price')
    axis.set_title(slot + ' vs Price')
    plt.show()


for i in slots:
    scatter_plot(i[0], i[1])

# In[3]
# Normalizing (z = (x – min) / (max – min)), shuffling, and splitting the data

# shuffling to the same random state for debugging
data = data.sample(frac=1).reset_index()
data_size = len(data['ID'])

# Normalizing (z = (x – min) / (max – min))
# best 4 of the numerical features X1,X2,X3,X4 and X0 always equal 1

X0 = np.ones(data_size)

X1 = data['horsepower'].to_numpy()
X1 = ((X1 - min(X1)) / (max(X1) - min(X1)))

X2 = data['carwidth'].to_numpy()
X2 = ((X2 - min(X2)) / (max(X2) - min(X2)))

X3 = data['enginesize'].to_numpy()
X3 = ((X3 - min(X3)) / (max(X3) - min(X3)))

X4 = data['curbweight'].to_numpy()
X4 = ((X4 - min(X4)) / (max(X4) - min(X4)))

y_full = data['price'].to_numpy()

# splitting the data %85 training, %15 testing
M = int(data_size * 0.85)
X_train = np.array([X0[:M], X1[:M], X2[:M], X3[:M], X4[:M]]).transpose()
y_train = np.array(y_full[:M]).transpose()

X_test = np.array([X0[M:], X1[M:], X2[M:], X3[M:], X4[M:]]).transpose()
y_test = np.array(y_full[M:]).transpose()


# print("Shape of X_train :", X_train.shape)
# print("Shape of Y_train :", y_train.shape)
# print("Shape of X_test :", X_test.shape)
# print("Shape of Y_test :", y_test.shape)

# In[4]
# Linear regression (GD)
# cost function h(x) = theta0 X0 + theta1 X1 + theta2 X2 + theta3 X3 + theta4 X4 == X.dot(theta)
def compute_cost(x, y, Theta):
    hypothesis = x.dot(Theta)
    # print('hypothesis= ', hypothesis[:5])
    errors = np.subtract(hypothesis, y)
    # print('errors= ', errors[:5])
    # print('sqrErrors= ', sqrErrors[:5])
    J = 1 / (2 * M) * errors.T.dot(errors)
    return J


def gradient_descent(x, y, Theta, Alpha, iteration):
    costHistory = np.zeros(iteration)

    for i in range(iteration):
        hypothesis = x.dot(Theta)
        # print('hypothesis= ', hypothesis[:5])
        errors = np.subtract(hypothesis, y)
        # print('errors= ', errors[:5])
        sum_delta = (Alpha / M) * x.transpose().dot(errors)
        # print('sum_delta= ', sum_delta[:5])
        Theta = Theta - sum_delta

        costHistory[i] = compute_cost(x, y, Theta)

    return Theta, costHistory


def plot_history(ch, i):
    # Plot the history
    plt.plot(range(1, i + 1), ch, color='blue')
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.grid()
    plt.xlabel("Number of iterations")
    plt.ylabel("cost (J)")
    plt.title("Convergence of gradient descent")
    plt.show()


theta, cost_history = gradient_descent(X_train, y_train, theta, alpha, iterations)
plot_history(cost_history, iterations)

print('Final value of theta =', theta)
print('First 5 values from cost_history =', cost_history[:5])
print('Last 5 values from cost_history =', cost_history[-5:])


# In[5]
# MSE (calculation and plot)

def test(xt, yt, Theta):
    hypothesis = xt.dot(Theta)
    errors = np.subtract(hypothesis, yt)
    sqrErrors = np.square(errors)
    plt.plot(range(1, len(y_test) + 1), sqrErrors, color='Red')
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.grid()
    plt.xlabel("Test Number")
    plt.ylabel("MSE")
    plt.title("Accuracy")
    plt.show()
    errors = abs(errors)
    threshold = 400
    for i in range(len(errors)):
        if errors[i] < threshold:
            errors[i] = 0
        else:
            errors[i] = 1

    error = (1 / len(errors)) * np.sum(errors)
    print("Test error is :", error * 100, "%")
    print("Test Accuracy is :", (1 - error) * 100, "%")


test(X_test, y_test, theta)


# In[6]
def training_epoch_test():
    fig, ax = plt.subplots(figsize=(7, 7))
    learning_rate = [0.1,0.01,0.001, 0.9, 0.03]
    colors = ['red', 'green', 'blue', 'black', 'yellow']
    for i in range(len(learning_rate)):
        best_theta, cost = gradient_descent(X_train, y_train, np.zeros(5), learning_rate[i], iterations)
        ax.plot(np.arange(iterations), cost, colors[i], label="alpha = " + str(learning_rate[i]))

    ax.legend(loc=1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs Training Epoch')
    plt.show()


training_epoch_test()
