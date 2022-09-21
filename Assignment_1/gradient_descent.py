import numpy as np
import pandas as pd

def gradient_descent(x,y,iterations,learning_rate):
    w0, w1, w2, w3 = 0

    xn1, xn2, xn3 = x

    n = len(y)

    for i in range(iterations):
        # calculate predicted y value
        y_hat = w0 + w1*xn1 + w2*xn2 + w3*xn3

        # calculate current loss/cost value
        loss = (1/n) * sum([calculated_error**2 for calculated_error in (y-y_hat)])

        # calculate the partial derivative for each weight coefficient
        w0d = -learning_rate(1/n)*sum([x*(y-y_hat)])
        w1d = -learning_rate(1/n)*sum([x*(y-y_hat)])
        w2d = -learning_rate(1/n)*sum([x*(y-y_hat)])
        w3d = -learning_rate(1/n)*sum([x*(y-y_hat)])

        w0 = w0 - w0d
        w1 = w1 - w1d
        w2 = w2 - w2d
        w2 = w2 - w2d
        w3 = w3 - w3d

        print("w0 {}, w1 {}, w2 {}, w3 {}, loss/cost {}, iteration {}".format(w0,w1,w2,w3,loss,i))
