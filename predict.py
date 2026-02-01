import matplotlib.pyplot as plt
import numpy as np

# I used ChatGPT to understand the formulas given in the Linear Regression Optional Reading better and to generate examples for my understanding

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")


def learn(num_iter):
    alpha = 0.001
    slope = 1 # Seed value
    slope_history = []
    bias_history = []
    error_history = []
    bias = 1 # Seed value
    params = np.array([slope, bias])

    j = 0

    while j < num_iter:
        x_j_input = x_train[j % len(x_train)]
        input_j = np.array([x_j_input, 1]) # Use 1 as the x input val for bias

        # Calculate sum of loss
        loss_sum = 0
        for n in range(len(x_train)):
            pred_val = np.sum(params * input_j) 
            loss = (y_train[n] - pred_val)
            loss_sum = loss_sum + loss
            loss_sum_scaled = loss_sum * input_j

        error_history.append(loss_sum)
        # Update theta
        params = params + alpha * loss_sum_scaled
        slope_history.append(params[0])
        bias_history.append(params[0])

        j = j + 1
    
    return params[0], params[1], slope_history, error_history


learned_slope, bias, slope_history, error_history = learn(200)

print(f"Learned slope: {learned_slope}")
print(f"Learned bias: {bias}")
print(f"Equation: h(x) = {learned_slope}x + {bias}")

plt.scatter(x_train, y_train, color="blue")
plt.scatter(x_test, y_test, color="orange")
plt.title("Training and Test Data")
plt.show()

plt.scatter(x_train, y_train)
plt.plot([-2, 0,2], [bias - 2 * learned_slope, bias, bias + 2 * learned_slope], color="yellow")
plt.title("Model with Training Data")
plt.show()

plt.scatter(x_test, y_test)
plt.plot([-1, 0,1], [bias - learned_slope, bias, bias + learned_slope], color="yellow")
plt.title("Model with Test Data")
plt.show()

plt.scatter([i for i in range(0, len(slope_history))], slope_history)
plt.title("Slope history")
plt.show()

plt.scatter([i for i in range(0, len(slope_history))], slope_history)
plt.title("Bias history")
plt.show()

plt.title("Loss History")
plt.scatter([i for i in range(0, len(error_history))], error_history)
plt.show()