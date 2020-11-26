import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    def regress(self, x, y):
        # precompute values
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_var = np.sum( (x-x_mean)**2 )

        # compute least squares estimators
        self.beta1 = np.sum( (x-x_mean) * (y-y_mean) ) / x_var
        self.beta0 = y_mean - self.beta1 * x_mean

        # precompute more values
        yhat = self.beta0 + self.beta1 * x
        SSR = np.sum((y - yhat)**2)
        std_eps = SSR / (len(x) - 2)

        # compute least squares estimator error estimates
        self.beta1_std = std_eps / np.sqrt(x_var)
        self.beta0_std = self.beta1_std * np.sqrt(np.sum(x**2) / len(x))

    def __call__(self, x):
        return self.beta0 + self.beta1 * x


if __name__ == '__main__':
    # prepare data
    x = np.linspace(0, 1, 20) * np.pi / 2
    y = np.sin(x)
    e = 0.1*np.random.randn(x.size)

    # perform regression
    SLR = SimpleLinearRegression()
    SLR.regress(x,y+e)

    # plot results
    plt.scatter(x, y+e, s=7)
    plt.plot(x, SLR(x), label="SLR")
    plt.plot(x, y, label="y(x)")
    plt.legend()
    plt.show()

    # print estimators
    print(f"beta0 = {SLR.beta0:.3e} +- {SLR.beta0_std:.3e}")
    print(f"beta1 = {SLR.beta1:.3e} +- {SLR.beta1_std:.3e}")

