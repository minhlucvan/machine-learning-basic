import unittest
from sklearn import datasets, linear_model
import numpy as np
import os
from . import single_variable as sv

curdir=os.path.dirname(__file__)

data_file = curdir + '/data.csv'
initial_b = 0
initial_m = 0
learning_rate = 0.0001
number_iterations = 1000

points = sv.load_data(data_file)
x = np.array(points[:, 0])
y = np.array(points[:, 1])



class SingleVariableTest(unittest.TestCase):

    def test_scikit_learn(self):
        x_data = np.matrix(points[:, 0]).T
        y_data = np.matrix(points[:, 1]).T.A
        regr = linear_model.LinearRegression(fit_intercept=False)  # fit_intercept = False for calculating the bias
        regr.fit(x_data, y_data)

        b, m = sv.single_variable_linear_regresstion(points, initial_b, initial_m, number_iterations, learning_rate)
        res = np.array([[b, m]])

        # self.assertTrue((np.array([[0, 0]]) == np.array([[0, 0]])).all())
        x = np.random.random()
        self.assertTrue(regr.predict(x) - (x * m + b) < learning_rate)

    # def test_tensor_flow(self):
    # def test_tensor_flow(self):
    #     self.failIf(IsOdd(2))


def main():
    unittest.main()


if __name__ == '__main__':
    main()