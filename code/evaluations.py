from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

def rmse_test(true_data, predicted_data):
    '''Calculate the root mean square error of predicted values against true values

    :params list true_data: list of true values
    :params list predicted_data: list of predicted values
    :returns: rmse values
    :rtype: float
    '''

    return sqrt(mean_squared_error(true_data, predicted_data))