import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

# define standard dataframe format for multiple data generation functions
columns_global=['loggingTime(txt)', 'accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'c_fr', 'c_rr', 'timestep', 'gyroRotationY(rad/s)', 'gyroRotationX(rad/s)']

def custom_smooth(array, rounds):
    
    for all in range(rounds):
        array = np.convolve(array, [0.025, 0.025, 0.075, 0.075, 0.30, 0.30, 0.075, 0.075, 0.025, 0.025], 'same')
    
    return(array)

def from_sensor_log_iOS_app(path: str):

    print('Converting file to dataframe...')
    data_in = pd.read_csv(path)[['loggingTime(txt)', 'accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'gyroRotationY(rad/s)', 'gyroRotationX(rad/s)']]

    print('Parsing timesteps...')
    #Create datetime column to be interpolated
    data_in['datetime'] = pd.to_datetime(data_in['loggingTime(txt)'])
    data_in['datetime'] = pd.DatetimeIndex(data_in['datetime'])
    #drop redundant column
    data_in = data_in.drop(columns='loggingTime(txt)')
    #select interesting time range
    data_in = data_in[3100:5500]
    #set index to be picked up by interpolation function
    data_in = data_in.set_index('datetime')

    data_in['c_fr_array'] = 0
    data_in['c_rr_array'] = 0

    #resampling to time resolution, interpolate linearly then drop all nans
    data_in = data_in.resample('1ms')
    data_in = data_in.interpolate(method='linear')
    data_in = data_in.dropna(how='any')

    #resampling to 10ms, interpolate linearly then drop all nans
    data_in = data_in.resample('10ms')
    data_in = data_in.interpolate(method='linear')
    data_in = data_in.dropna(how='any')

    data_in['accelerometerAccelerationX(G)'] = data_in['accelerometerAccelerationX(G)'].rolling(window = 20).mean()
    data_in['accelerometerAccelerationY(G)'] = data_in['accelerometerAccelerationY(G)'].rolling(window = 20).mean()
    data_in['gyroRotationY(rad/s)'] = data_in['gyroRotationY(rad/s)'].rolling(window = 20).mean()
    data_in['gyroRotationX(rad/s)'] = data_in['gyroRotationX(rad/s)'].rolling(window = 20).mean()
    data_in = data_in.dropna(how='any')

    #create new time and timestep columns
    data_in['time'] = data_in.index
    data_in['timestep'] = data_in['time'].diff().dt.total_seconds()

    #create dataframe and drop nans one more time
    data = pd.DataFrame(list(zip(data_in['time'], data_in['accelerometerAccelerationX(G)'], data_in['accelerometerAccelerationY(G)'], data_in['c_fr_array'], data_in['c_rr_array'], data_in['timestep'], data_in['gyroRotationY(rad/s)'], data_in['gyroRotationX(rad/s)'])), \
        columns=columns_global)
    data = data.dropna(how='any')
    data = data.reset_index(drop=True)

    return data

def get_demo_G_function(
        timespan: int = 3,
        lat_magnitude: float = 1.4, lat_frequency: float = 0.5,
        long_magnitude: float = 0.6, long_frequency: float = 1
    ):

    #  Default time resolution is set to 100hz
    time_res = 100  # hz

    G_lat_array = np.array([math.sin(2 * math.pi * lat_frequency * x / time_res) * lat_magnitude for x in range(time_res * timespan)]) # 100 steps is 1s
    G_lat_array[200:] = 0

    G_long_array = np.array([math.sin(2 * math.pi * long_frequency * x / time_res) * long_magnitude for x in range(time_res * timespan)])  # 100 steps is 1s
    G_long_array[175:] = -long_magnitude

    c_fr_array = np.array([0.0 for x in range(time_res * timespan)])
    c_fr_array[130:160] = -0.008  # m
    c_fr_array = custom_smooth(c_fr_array, 3)

    c_rr_array = np.array([0.0 for x in range(time_res * timespan)])
    c_rr_array[150:180] = -0.010  # m
    c_rr_array = custom_smooth(c_rr_array, 3)

    time_array = [x/time_res for x in range(len(G_lat_array))]
    dt_array = [1/time_res for all in range(len(G_lat_array))]

    control_array_roll = np.array([0.0 for x in range(time_res * timespan)])
    control_array_pitch = np.array([0.0 for x in range(time_res * timespan)])

    data = pd.DataFrame(list(zip(time_array, G_lat_array, G_long_array, c_fr_array, c_rr_array, dt_array, control_array_roll, control_array_pitch)), \
        columns=columns_global)

    return data

def get_bump_function(timespan: int = 9, magnitude: float = 0.020, frequency: float = 0.5):

    #  Default time resolution is set to 100hz
    time_res = 100  # hz
    dt = 1/time_res

    c_array = [math.sin(2 * math.pi * frequency * x / time_res) * magnitude for x in range(time_res * timespan)]  # 100 steps is 1s
    time_array = [x/time_res for x in range(len(c_array))]

    return time_array, c_array, dt

'''
Begin functions for dev purposes, exploring telemetry data.
'''

def from_sensor_log_iOS_app_dev(path: str):

    print('Converting file to dataframe...')
    data_in = pd.read_csv(path)[['loggingTime(txt)', 'accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'gyroRotationY(rad/s)', 'motionPitch(rad)']]

    print('Parsing timesteps...')
    #Create datetime column to be interpolated
    data_in['datetime'] = pd.to_datetime(data_in['loggingTime(txt)'])
    data_in['datetime'] = pd.DatetimeIndex(data_in['datetime'])
    #drop redundant column
    data_in = data_in.drop(columns='loggingTime(txt)')
    #select interesting time range
    data_in = data_in[3100:5500]
    #set index to be picked up by interpolation function
    data_in = data_in.set_index('datetime')

    data_in['c_fr_array'] = 0
    data_in['c_rr_array'] = 0

    #resampling to time resolution, interpolate linearly then drop all nans
    data_in = data_in.resample('1ms')
    data_in = data_in.interpolate(method='linear')
    data_in = data_in.dropna(how='any')

    #resampling to 10ms, interpolate linearly then drop all nans
    data_in = data_in.resample('10ms')
    data_in = data_in.interpolate(method='linear')
    data_in = data_in.dropna(how='any')

    data_in['gyroRotationY(rad/s)'] = data_in['gyroRotationY(rad/s)'].rolling(window = 10).mean()
    data_in = data_in.dropna(how='any')

    #create new time and timestep columns
    data_in['time'] = data_in.index
    data_in['timestep'] = data_in['time'].diff().dt.total_seconds()

    #create dataframe and drop nans one more time
    data = pd.DataFrame(list(zip(data_in['time'], data_in['accelerometerAccelerationX(G)'], data_in['accelerometerAccelerationY(G)'], data_in['c_fr_array'], data_in['c_rr_array'], data_in['timestep'], data_in['gyroRotationY(rad/s)'], data_in['motionPitch(rad)'])), \
        columns=columns_global)
    data = data.dropna(how='any')
    data = data.reset_index(drop=True)

    print(data_in['gyroRotationY(rad/s)'])

    return data