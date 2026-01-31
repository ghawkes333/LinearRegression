import matplotlib.pyplot as plt
import numpy as np

# I used ChatGPT to understand the formulas given in the Linear Regression Optional Reading better and to generate examples for my understanding

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

# plt.scatter(x_train, y_train)
# plt.show()

# plt.scatter(x_test, y_train)
# plt.show()


def learn(num_iter):
    alpha = 0.001
    slope = 10
    slope_history = []
    error_history = []
    bias = 1
    params = np.array([slope, bias])

    i = 0

    while i < num_iter:
        x_i_input = x_train[i % len(x_train)]
        input = np.array([x_i_input, 1]) # Use 1 as the x input val for bias
        y_i = y_train[i % len(x_train)]

        # Calculate sum of loss
        loss_sum = 0
        for n in range(len(x_train)):
            pred_val = np.sum(params * input) 
            loss = (y_train[n] - pred_val) # TODO - should I square this?
            loss_sum = loss_sum + loss
            loss_sum_scaled = loss_sum * input

        error_history.append(loss_sum)
        # Update theta
        params = params - alpha * loss_sum_scaled
        slope_history.append(params[0])

        i = i + 1
    
    return params[0], params[1], slope_history, error_history


learned_slope, bias, slope_history, error_history = learn(100)

print(f"Equation is slope = {learned_slope} and bias is {bias}")

plt.scatter(x_train, y_train)
plt.plot([-2, 0,2], [bias - 2 * learned_slope, bias, bias + 2 * learned_slope], color="yellow")
plt.show()

print(slope_history)
plt.scatter([i for i in range(0, len(slope_history))], slope_history)
plt.title("Slope history")
plt.show()

print(error_history)
plt.title("Loss History")
plt.scatter([i for i in range(0, len(error_history))], error_history)
plt.show()