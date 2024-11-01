import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2 as cv
import logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    handlers=[
        logging.StreamHandler(),  # Output logs to console
        logging.FileHandler('app.log')  # Optionally, log to a file
    ]
)
from sklearn.metrics import mean_squared_error

import numpy as np
from abc import ABC, abstractmethod

class GradientAbstract(ABC):
    mse_ = None
    
    def __init__(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, m: float = 00.1, c: float = 0, iterations: int = 1000):
        self.X = X
        self.y = y
        self.lr = lr
        self.m = m
        self.c = c
        self.iterations = iterations

    @abstractmethod
    def MSE(self, y_pred: np.ndarray) -> float:
        pass

    @abstractmethod
    def slopeDelta(self) -> float:
        pass

    @abstractmethod
    def interceptDelta(self) -> float:
        pass

    @abstractmethod
    def predict_(self) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def getGradients(self) -> None:
        pass

class GradientDescent(GradientAbstract):
    
    def MSE(self, y_pred) -> float:
        self.mse_ = mean_squared_error(y_true=self.y, y_pred=y_pred)
        return self.mse_  
    
    def slopeDelta(self) -> float:
        return -2 / len(self.y) * np.sum(self.X * (self.y - (self.m * self.X + self.c)))
    
    def interceptDelta(self) -> float:
        return -2 / len(self.y) * np.sum(self.y - (self.m * self.X + self.c))
    
    def predict_(self) -> np.ndarray:
        y_pred = self.m * self.X + self.c
        return y_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        # logging.info(f"intercept used for predition = {self.c}")
        return self.m * X + self.c

    @staticmethod
    def score(y_true:np.ndarray, y_pred:np.ndarray):
        diff = []
        for i, j in zip(y_true, y_pred):
            diff.append(i -j)
        return sum(diff)
   
    def getGradients(self, X_test):
        for i in range(self.iterations):
            y_pred = self.predict_()
            mse = self.MSE(y_pred=y_pred)

            delta_m = self.slopeDelta()
            delta_c = self.interceptDelta()
            
            self.m -= self.lr * delta_m
            self.c -= self.lr * delta_c

            # logging.info({"c":self.c, "m":self.m, "mse":mse})
            yield {"c":self.c, "m":self.m, "mse":mse, "y_pred":self.predict(X_test)}