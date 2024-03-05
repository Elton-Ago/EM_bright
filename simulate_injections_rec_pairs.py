import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import pandas as pd


class SimulateInjRecPairs_1:

#         This method creates a fake distribution of recovered masses based on the injected masses

#         This method creates a fake distribution of recovered masses based on the injected masses within a random range and offset
# 	======
# 
# 	INPUT:
# 	======
# 	offset: Number difference from mass injection
# 	sigma: Standard Deviation of sample
# 
# 	RETURNS:
# 	=======
# 	Distribution of fake recovered masses based off injections reshaped as (-1, 1)
# 	"""
# 	METHOD:
# 	======
# 	This method produces a random distirbution of predicted masses based on the masses injected and masses recovered
# 	======
# 
# 	INPUT:
# 	======
# 	test_size:
# 	random_state:
# 	test_result:
# 
# 	RETURNS:
# 	=======
# 	Distribution of the models tested sample of injected masses and predicted masses
# 	"""
# 	"""
# 	METHOD:
# 	======
# 	This method produces a plot distribution of predicted masses to test mass points
# 	
# 	INPUT:
# 	======
# 	None
# 
# 	RETURNS:
# 	=======
# 	Plot figure of returned tested masses compared to predicted masses returned from mass_predictor module
# 	"""

    def __init__(self, N=100000, m_min=10, m_max=20):
        """
        METHOD:
        =======
        This method produces a random distribution of masses with a given range

        INPUT:
        ======
        N: Total number of samples
        m_min: Lower bound of the masses
        m_max: Upper bound of masses

        RETURNS:
        ========
        Distribution of injected masses reshaped as (-1, 1)   
        """
        M_inj = m_min + (m_max - m_min) * np.random.random(N)
        self.M_inj = M_inj.reshape(-1, 1)

    def fake_recovered_masses(self, offset, sigma):
        """
        METHOD:
        =======
        This method creates a fake distribution of recovered masses based on the injected masses
        """
        #now we will get a gaussian distribution of the M_rec with an offset to t
        M_rec = np.random.normal(self.M_inj + offset, sigma)
        #now we have both the mass injections and mass recovered in however the len the N sample is
        self.M_rec = M_rec.reshape(-1, 1)
        #reshaping is better for the machine learning model
        return self.M_rec
    
    def mass_predictor(self, test_size=0.2, random_state=42, test_result=True):
        X_train, self.X_test, y_train, y_test = train_test_split(self.M_inj, self.M_rec, 
                                                                 test_size=test_size,
                                                                 random_state=random_state)
        RandomForestRegressionModel = RandomForestRegressor()
        RandomForestRegressionModel.fit(X_train, y_train)
        if not test_result:
            return RandomForestRegressionModel
        self.y_pred = RandomForestRegressionModel.predict(self.X_test)
        return (self.X_test, self.y_pred, RandomForestRegressor)
    
    def plotter(self):
        plt.scatter(self.X_test, self.y_pred, s=2) 
        plt.xlabel('Test Masses')
        plt.ylabel('Pred Masses')
        plt.title("Predicting Injected Mass Values")
        plt.show()
