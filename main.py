# In[1]
# import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing

theta = np.zeros(5)
iterations = int(5e6)
alpha = 0.25

# In[2]
# Scatter plots for feature selection
# Find best 4 of the numerical features
# horsepower, carwidth, curb weight, engine size

# import file with data
data = pd.read_csv(".\\car_data.csv")
"""
print(data.corr())

# plotting correlation heatmap~
dataPlot = sb.heatmap(data.corr(), cmap="YlGnBu", annot=True)

# displaying heatmap
plt.show()
"""

# In[3]
# Normalizing (z = (x – min) / (max – min)), shuffling, and splitting the data

# shuffling to the same random state for debugging
data = data.sample(frac=1, random_state=1).reset_index()

# Normalizing (z = (x – min) / (max – min))
scaler = preprocessing.MinMaxScaler()

# best 4 of the numerical features X1,X2,X3,X4 and X0 always equal 1
data_size = len(data['ID'])

X0 = np.ones(data_size)
X1 = data['horsepower'].to_numpy()
X2 = data['carwidth'].to_numpy()
X3 = data['enginesize'].to_numpy()
X4 = data['curbweight'].to_numpy()
y_full = data['price'].to_numpy()

# splitting the data %80 training, %20 testing
m = int(data_size * 0.8)
X = scaler.fit_transform([X1, X2, X3, X4])

X_train = np.array([X0[:m], X[0][:m], X[1][:m], X[2][:m], X[3][:m]]).transpose()
y_train = np.array(y_full[:m]).transpose()

X_test = np.array([X0[m:], X[0][m:], X[1][m:], X[2][m:], X[3][m:]]).transpose()
y_test = np.array(y_full[m:]).transpose()


# In[4]
# Linear regression (GD)
# cost function h(x) = theta1 + theta2 X1 + theta3 X2 + theta4 X3 + theta5 X4 == X.dot(theta)
def compute_cost(x, y, Theta):
    hypothesis = x.dot(Theta)
    # print('hypothesis= ', hypothesis[:5])
    errors = np.subtract(hypothesis, y)
    # print('errors= ', errors[:5])
    # print('sqrErrors= ', sqrErrors[:5])
    J = 1/(2 * m) * errors.T.dot(errors)
    return J


def gradient_descent(x, y, Theta, Alpha, i):
    costHistory = np.zeros(i)

    for i in range(i):
        hypothesis = x.dot(Theta)
        # print('hypothesis= ', hypothesis[:5])
        errors = np.subtract(hypothesis, y)
        # print('errors= ', errors[:5])
        sum_delta = (Alpha / m) * x.transpose().dot(errors)
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

def test(x, y, Theta):
    hypothesis = x.dot(Theta)
    errors = np.subtract(hypothesis, y)
    sqrErrors = np.square(errors)
    plt.plot(range(1, len(y_test) + 1), sqrErrors, color='Red')
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.grid()
    plt.xlabel("Test Number")
    plt.ylabel("MSE")
    plt.title("Accuracy")
    plt.show()


test(X_test, y_test, theta)
