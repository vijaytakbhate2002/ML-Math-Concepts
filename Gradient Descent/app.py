import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import cv2
from algos.gradient_descent import GradientDescent
warnings.filterwarnings("ignore")

m = 2.5  
c = 5    
noise_factor = 1.0  

np.random.seed(42)
X = np.random.rand(1000, 1) * 10  
y = m * X + c + np.random.randn(1000, 1) * noise_factor
data = pd.DataFrame(np.hstack((X, y)), columns=['X', 'y'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

def generateInterceptPlot(m, mse):
    plt.scatter(x=m, y=mse, color='b', label='Intercepts')
    plt.xlim(0, 4)  
    plt.ylim(0, 400)
    plt.xlabel(f'Slope {m[-1]}', color='red')
    plt.ylabel(f'Mean Squared Error {mse[-1]}', color='red')
    plt.title('Change in MSE with change in Slope', color='red')
    plt.draw()
    image = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    plt.close()
    cv2.imshow("Change in MSE with respect to Slope", image)

def generateSlopePlot(c, mse):
    plt.scatter(x=c, y=mse, color='b', label='Intercepts')
    plt.xlim(0, 4)  
    plt.ylim(0, 400)
    plt.xlabel(f'Intercept {c[-1]}', color='red')
    plt.ylabel(f'Mean Squared Error {mse[-1]}', color='red')
    plt.title('Change in MSE with change in Intercept', color='red')
    plt.draw()
    image = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    cv2.imshow("Change in MSE with respect to Intercept", image)


def generateScatter(y_pred, m, c):
    plt.scatter(X_test, y_test, color='blue', label='Data points') 

    x_values = np.linspace(min(X), max(X), 100) 
    y_values = m * x_values + c  
    plt.plot(x_values, y_values, color='red', label='Predicted line', linewidth=2)
    plt.xlabel('X', color='red')
    plt.ylabel('y', color='red')
    plt.title('Generated Dataset', color='red')
    plt.draw()
    image = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    cv2.imshow("Gradient Descent", image)

mse = []
m = []
c = []

gradient = GradientDescent(X=X_train, y=y_train, lr=0.0001, iterations=3000)
params = gradient.getGradients(X_test=X_test)

while params:
    mse.append(next(params)['mse'])
    m.append(next(params)['m'])
    c.append(next(params)['c'])
    generateScatter(next(params)['y_pred'], next(params)['m'], next(params)['c'])
    generateInterceptPlot(c, mse)
    generateSlopePlot(m, mse)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break