# -*- coding: utf-8 -*-


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from scipy.optimize import minimize


def prediction_analysis(data):
    '''
        This function analyses the predicted heart rate
        INPUT:
            data frame with prediction named : heart_rate_predicted, and the true heart rate named: heart_rate, and time named as :time (make sure this starts at zero.)
        OUTPUTS:
            data frame updated with:
                mse - mean squared error for prediction
                mae - mean absolute error for prediction
                mae_shape - mean absolute error for the shape of the prediction
                heart_rate_predicted_reshape - reshaped version of prediction
                time_reshape - reshaped version of time
                mae_all - mae for each point in time series
    '''
    
    data['mse'] = np.nan
    data['mae'] = np.nan
    data['mae_shape'] = np.nan
    data['heart_rate_predicted_reshaped'] = np.nan
    data['time_reshaped'] = np.nan
    data['mae_all'] = np.nan
    
    for i, row in data.itterrows():
        y_true = row.heart_rate
        y_pred = row.heart_rate_predicted
        time = row.time
        
        # calculate mean squared error:
        data.loc['mse',i] = mean_squared_error(y_true, y_pred)
        
        # calculate mean absolute error:
        data.loc['mae',i] = mean_absolute_error(y_true, y_pred)
        
        # calcualte shape error:
        params_0 = np.zeros((4,1))
        res = minimize(fun = shape_cost, argms = (y_true, y_pred, time), x0 = params_0, options = {'maxiter':1000})

        params_best = res.x.reshape((-1,1))
        data.loc['mae_shape', i] = shape_cost(params_best, y_true, y_pred, time)
        y_pred_modified, time_modified =  modify_prediction(params_best, y_pred, time)
        data.loc['heart_rate_predicted_reshaped', i] = list(y_pred_modified)
        data.loc['time_reshaped', i] = list(time_modified)
        
        
        # calculate accumulation of errors:
        data.loc['mae_all',i] = list(np.linalg.norm(y_true-y_pred, axis = 1))
        
        
    return data

      
        
def shape_cost(params, y_true, y_pred, time):
    '''
        This is the cost function for finding the shape error
        INPUTS:
            params = [x_scale, y_scale, x_translate, y_translate]'
            y_true, y_pred - true and estimated hear rates
        OUTPUTS:
            error
    '''
        
    y_true = y_true.reshape((-1,1))
    y_pred = y_pred.reshape((-1,1))
    time = time.reshape((-1,1))
    
    # modify prediction:
    y_pred_modified, time_modified = modify_prediction(params, y_pred, time)
    
    # turn them into pairs:
    y_true = np.hstack((y_true, time))
    y_pred = np.hstack((y_pred_modified, time_modified))
    
    # calcualte error:
    err = np.mean(np.linalg.norm(y_true-y_pred, axis = 1))
    
    return err
    
def modify_prediction(params, y_pred, time):
    x_scale = params[0,0]
    y_scale = params[1,0]
    x_translate = params[2,0]
    y_translate = params[3,0]
    
    y_pred_modified = y_pred * y_scale + y_translate
    time_modified = time * x_scale + x_translate
    
    return y_pred_modified, time_modified
        