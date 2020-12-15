# -*- coding: utf-8 -*-


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from utils import rescale


def find_n_plot(input_speed, input_alt, input_gender, input_sport, input_user, input_time_last, prevData, targData,
                model, num, to_plot, best):
    '''
    This function finds the best num_best predictions from the data set, returns their indices, and plots them
    INPUTS
        input_speed, input_alt, input_gender, input_sport, input_user, input_time_last, prevData, targData
        model - ..well model duh
        num - number of best results that should be kept
        to_plot - if True, best will be plotted
        best - if true, the best ones are found. Otherwise, randomly selected
    OUTPUTS:
        idx_list, errors - index of selected points and their associated error
    '''
    input_temporal = np.dstack([input_speed, input_alt])
    print('here')
    if best:
        idx_list = np.arange(input_speed.shape[0])
        errors = np.zeros(idx_list.shape[0])
        for plot_ind, i in enumerate(idx_list):
            inputs = [input_sport[i].reshape(1, 300, 1),
                      input_gender[i].reshape(1, 300, 1),
                      input_temporal[i].reshape(1, 300, 2),
                      prevData[i].reshape(1, 300, 4)]
            pred = model.predict(inputs).reshape(-1)
            actual = targData[i]
            errors[plot_ind] = mean_squared_error(pred, actual)
            print(plot_ind)

        idx_best = np.flip(np.argsort(errors))[:num]
        idx_list = idx_list[idx_best]
        errors = errors[idx_best]
    else:
        idx_list = np.random.randint(1, input_speed.shape[0], num)
        errors = np.zeros(idx_list.shape[0])
        for plot_ind, i in enumerate(idx_list):
            inputs = [input_sport[i].reshape(1, 300, 1),
                      input_gender[i].reshape(1, 300, 1),
                      input_temporal[i].reshape(1, 300, 2),
                      prevData[i].reshape(1, 300, 3)]
            pred = model.predict(inputs).reshape(-1)
            actual = targData[i]
            errors[plot_ind] = mean_squared_error(pred, actual)
    print('here')
    if to_plot:
        for plot_ind, i in enumerate(idx_list):
            inputs = [input_sport[i].reshape(1, 300, 1),
                      input_gender[i].reshape(1, 300, 1),
                      input_temporal[i].reshape(1, 300, 2),
                      prevData[i].reshape(1, 300, 4)]

            sport = input_sport[i][0]
            gender = input_gender[i][0]

            pred = model.predict(inputs).reshape(-1)
            actual = targData[i]

            pred_rescaled = pred #rescale(pred, 21.11, 137.95)

            actual_rescaled = actual #rescale(actual, 21.11, 137.95)

            err = mean_absolute_error(pred_rescaled, actual_rescaled)
            #err = errors[plot_ind]
            plt.subplot(3, 4, plot_ind + 1)
            plt.plot(actual_rescaled, color='r', label='actual')
            plt.plot(pred_rescaled, color='b', label='pred')
            plt.title(f'Gender: {gender}, Sport: {sport}, Error: {round(err, 2)}')

        plt.legend()
        plt.xlim(0, 300)
        plt.ylim(-3, 3)
        plt.show()

    return idx_list, errors


def analyze_all_predictions(input_temporal, input_gender, input_sport, input_user, input_time_last, prevData, targData,
                            model):
    mse_all, mae_all, mae_shape_all, y_pred_modified_all, time_reshaped_all, mae_all_all = [], [], [], [], [], []
    for i in list(np.arange(input_gender.shape[0])):
        inputs = [input_sport[i].reshape(1, 300, 1),
                  input_gender[i].reshape(1, 300, 1),
                  input_temporal[i].reshape(1, 300, 2),
                  prevData[i].reshape(1, 300, 3)]

        sport = input_sport[i][0]
        gender = input_gender[i][0]

        pred = model.predict(inputs).reshape(-1)
        actual = targData[i]

        # input("going into prediction analysis - Press Enter to continue...")

        mse, mae, mae_shape, y_pred_modified, time_reshaped, mae_all_point = prediction_analysis(actual, pred)

        mse_all.append(mse)
        mae_all.append(mae)
        mae_shape_all.append(mae_shape)
        y_pred_modified_all.append(y_pred_modified)
        time_reshaped_all.append(time_reshaped)
        mae_all_all.append(mae_all_point)

    return mse_all, mae_all, mae_shape_all, y_pred_modified_all, time_reshaped_all, mae_all_all


def prediction_analysis(y_true, y_pred):
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
    time = np.arange(y_true.shape[0])

    # calculate mean squared error:
    mse = mean_squared_error(y_true, y_pred)

    # calculate mean absolute error:
    mae = mean_absolute_error(y_true, y_pred)

    # calcualte shape error:
    params_0 = np.zeros((4, 1))
    # y_true += 10
    # y_pred +=10
    res = minimize(fun=shape_cost, args=(y_true, y_pred, time), x0=params_0, options={'maxiter': 1000})

    params_best = res.x.reshape((-1, 1))
    mae_shape = shape_cost(params_best, y_true, y_pred, time)
    y_pred_modified, time_modified = modify_prediction(params_best, y_pred, time)

    y_pred_modified = list(y_pred_modified)

    time_reshaped = list(time_modified)

    '''
    plt.plot(time, y_true, color = 'r', label ='true')
    plt.plot(time, y_pred, color = 'b', label = 'predicted')
    plt.plot(time_reshaped, y_pred_modified, color = 'g', label = 'modified predicted')
    plt.legend()

    plt.show()
    '''

    # calculate accumulation of errors:
    mae_all = list(np.linalg.norm(y_true - y_pred, axis=1))

    return mse, mae, mae_shape, y_pred_modified, time_reshaped, mae_all


def shape_cost(params, y_true, y_pred, time):
    '''
        This is the cost function for finding the shape error
        INPUTS:
            params = [x_scale, y_scale, x_translate, y_translate]'
            y_true, y_pred - true and estimated hear rates
        OUTPUTS:
            error
    '''

    y_true = y_true.reshape((-1, 1))
    y_pred = y_pred.reshape((-1, 1))
    time = time.reshape((-1, 1))

    # modify prediction:
    y_pred_modified, time_modified = modify_prediction(params, y_pred, time)

    # turn them into pairs:
    y_true = np.hstack((y_true, time))
    y_pred = np.hstack((y_pred_modified, time_modified))

    # calcualte error:
    err = np.mean(np.linalg.norm(y_true - y_pred, axis=1))

    return err


def modify_prediction(params, y_pred, time):
    x_scale = params[0]
    y_scale = params[1]
    x_translate = params[2]
    y_translate = params[3]

    y_pred_modified = y_pred * y_scale + y_translate
    time_modified = time * x_scale + x_translate

    return y_pred_modified, time_modified
