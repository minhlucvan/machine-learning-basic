from numpy import *


def load_data(file):
    points = genfromtxt(file, delimiter=",")

    return points


def cost_function(b, m, points):
    total_cost = 0
    n = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (y - (m * x + b)) ** 2

    return total_cost / n


def gradient_step(b_current, m_current, points, learning_rate):
    n = float(len(points))
    m_gradient = 0
    b_gradient = 0

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        b_gradient += -(2/n) * (y - ((m_current * x) + b_current))
        m_gradient +=  -(2/n) * x * (y - ((m_current * x) + b_current))

    b_new = b_current - (learning_rate * b_gradient)
    m_new = m_current - (learning_rate * m_gradient)

    return b_new, m_new


def gradient_runner(starting_b, starting_m, points, learning_rate, number_iterations):
    b = starting_b
    m = starting_m

    for i in range(number_iterations):
        # print("gradient step {0}:".format(i))
        # print("current b {0}:".format(b))
        # print("current m {0}:".format(m))

        b, m = gradient_step(b, m, points, learning_rate)

    return b, m


def single_variable_linear_regresstion(points, starting_b, starting_m, number_iterations, learning_rate):
    b, m = gradient_runner(starting_b, starting_m, points, learning_rate, number_iterations)

    return b, m


def main():
    data_file = 'data.csv'
    initial_b = 0
    initial_m = 0
    learning_rate = 0.0001
    number_iterations = 1000

    points = load_data(data_file)

    b, m = single_variable_linear_regresstion(points, initial_b, initial_m, number_iterations, learning_rate)

    print("================RESULT===========================")
    print("number iterations {0}:".format(number_iterations))
    print("optimized b {0}:".format(b))
    print("optimized m {0}:".format(m))


if __name__ == '__main__':
    main()
