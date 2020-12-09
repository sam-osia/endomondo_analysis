from collections import defaultdict
import json
import gzip
import pandas as pd
import numpy as np
import itertools
from utils import *
from sklearn import preprocessing


def create_time_series_data(df):
    '''

    :param df: dataframe with time-series
    :return: temporal sequences, target sequence
    '''
    df = df.reset_index(drop=True)

    data = np.dstack([np.array(df["tar_derived_speed"].tolist()),
                      np.array(df["altitude"].tolist())])
    targData = np.array(df["tar_heart_rate"].tolist()).reshape(-1, 300, 1)

    return data, targData

def create_time_series_1D(df, feature):
    '''

    :param df: dataframe with time-series
    :return: temporal sequences, target sequence
    '''
    df = df.reset_index(drop=True)
    targData = np.array(df[feature].tolist()).reshape(-1, 300, 1)

    return targData

def process_catData(df, feature):
    '''

    :param df: dataframe
    :param feature: (str) categorical feature to be processed
    :return: processed and reshaped feature
    '''
    df = df.reset_index(drop=True)
    le = preprocessing.LabelEncoder()
    le.fit(df[feature])
    transfrom_data = le.transform(df[feature])
    print(f'Feature: {feature}')
    print(transfrom_data.tolist()[:2])
    print(list(le.inverse_transform(transfrom_data.tolist()[:2])))
    print()
    return np.tile(transfrom_data, (300, 1)).T.reshape(-1, 300, 1)

def find_user_workouts(wid, df):

    w_df = df.loc[lambda df: df['id'] == wid]
    uid = w_df['userId'].tolist()[0]
    t = w_df['timestamp'].tolist()[0][0]

    u_df = df.loc[lambda df: df['userId'] == uid][:]
    u_df['start'] = u_df['timestamp'].apply(lambda x: x[0])

    myList = list(zip(u_df.start, u_df.id))
    myList = sorted(myList, key=lambda x: x[0])

    idx = myList.index((t, wid))

    if idx > 0:
        return myList[idx-1][1]
    else:
        return None

def time_since_last(wid, df):
    prevWid = df[df["id"] == wid]["prevId"].values[0]
    t = np.NaN
    if prevWid > 0:
        timePrev = np.array(df.loc[lambda df: df['id'] == prevWid]['timestamp'])[0][0]
        timeCurr = np.array(df.loc[lambda df: df['id'] == wid]['timestamp'])[0][0]
        t = timeCurr - timePrev
    return t


def prev_wid(df):
    return df['id'].apply(lambda x: find_user_workouts(x, df))

def scaling (row, mean, std, zMultiple=1):
    row = np.array(row)
    row -= mean
    row /= std
    row *= zMultiple
    return row.tolist()

def scaleData(df, feature):
    flat_data = list(itertools.chain.from_iterable(df[feature].values.flatten()))
    mean, std = np.mean(flat_data), np.std(flat_data)
    scaled_feat = df[feature].apply(scaling, args=(mean, std))
    return scaled_feat

def clean_time(row):
    row = np.array(row)
    row -= row[0]
    return row

def curr_preprocess(df):
    df['prevId'] = prev_wid(df)
    df['time_last'] = df['id'].apply(lambda x: time_since_last(x, df))
    df = prev_dataframe(df)

    for feature in ["tar_derived_speed", "altitude", "tar_heart_rate"]:
        df[feature] = scaleData(df, feature)
    print(len(df))
    df = remove_first_workout(df)
    print(len(df))
    df.reset_index(drop=True, inplace=True)

    #seqs, targData = create_time_series_data(df)
    input_speed = create_time_series_1D(df, 'tar_derived_speed')
    input_alt = create_time_series_1D(df, 'altitude')
    targData = create_time_series_1D(df, 'tar_heart_rate')

    input_gender = process_catData(df, 'gender')
    input_sport = process_catData(df, 'sport')
    input_user = process_catData(df, 'userId')
    input_time_last = np.tile(df.time_last, (300, 1)).T.reshape(-1, 300, 1)
    prevData = prev_time_series_data(df)
    return input_speed, input_alt, input_gender, input_sport, input_user, input_time_last, prevData, targData


def prev_dataframe(df):
    #df["prevID"] = prev_wid(df)
    df2 = df[["tar_derived_speed", "altitude", "tar_heart_rate", "id"]][:]
    df2.rename(columns={"tar_derived_speed": "prev_tar_speed",
                        "altitude": "prev_altitude",
                        "tar_heart_rate": "prev_tar_heart_rate",
                        "id": "id"}, inplace=True)
    prevDf = pd.DataFrame({"pid": df["prevId"]})
    prevDf = prevDf.merge(df2, left_on="pid", right_on="id")
    mergeDF = df.merge(prevDf, left_on="prevId", right_on="pid")
    mergeDF.rename(columns={"id_x": "id"}, inplace=True)
    return mergeDF

def prev_time_series_data(mergeDF):
    data = np.dstack([np.array(mergeDF["prev_tar_speed"].tolist()),
                      np.array(mergeDF["prev_altitude"].tolist()),
                      np.array(mergeDF["prev_tar_heart_rate"].tolist())])
    return data

def remove_first_workout(df):
    df_list = []
    uList = df['userId'].unique()
    for u in uList:
        u_df = df[df['userId'] == u]
        wid = u_df['id']
        t = u_df['timestamp']
        startT = t.apply(lambda x: x[0])

        myList = list(zip(startT, wid))
        myList = sorted(myList, key=lambda x: x[0])

        for i in myList[1:]:
            j = i[1]
            df_list.append(df[df['id'] == j][:])
    return pd.concat(df_list)


if __name__ == "__main__":
    set_path("saman")
    df = pd.read_json('./data/female_bike.json')
    #newDf = remove_first_workout(df)
    #print(newDf.shape)
    print(df.shape)
    print(len(df.userId.unique()))
    input_speed, input_alt, input_gender, input_sport, input_user, input_time_last, prevData, targData = curr_preprocess(df)
    print(input_speed.shape)
    print(input_gender.shape)
    print(input_sport.shape)
    print(input_time_last)
    print(prevData.shape)
