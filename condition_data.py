import pandas as pd
import math
import numpy as np

# define standard dataframe format for multiple data generation functions
columns_global=['loggingTime(txt)', 'accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'c_fr', 'c_rr', 'timestep', 'motionRoll(rad)', 'motionPitch(rad)']

def custom_smooth(array, rounds):
    
    for all in range(rounds):
        array = np.convolve(array, [0.025, 0.025, 0.075, 0.075, 0.30, 0.30, 0.075, 0.075, 0.025, 0.025], 'same')
    
    return(array)

def from_sensor_log_iOS_app(path: str):

    print('Converting file to dataframe...')
    data = pd.read_csv(path)[['loggingTime(txt)', 'accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'motionRoll(rad)', 'motionPitch(rad)']]

    print('Parsing timesteps...')
    data['datetime'] = pd.to_datetime(data['loggingTime(txt)'])
    data['timestep'] = data['datetime'].diff()
    data['timestep'][0] = data['timestep'][1]
    for i, timestep in enumerate(data['timestep']):
        data['timestep'][i] = int(str(data['timestep'][i])[-6:]) / 1000000  # us to s
    data = data[3200:5400].reset_index()

    data['accelerometerAccelerationX(G)'] = custom_smooth(np.array(data['accelerometerAccelerationX(G)']), 500)
    data['accelerometerAccelerationY(G)'] = custom_smooth(np.array(data['accelerometerAccelerationY(G)']), 500)

    data['c_fr_array'] = 0
    data['c_rr_array'] = 0

    data = data.round({'accelerometerAccelerationX(G)': 3, 'accelerometerAccelerationY(G)': 3})

    data = pd.DataFrame(list(zip(data['loggingTime(txt)'], data['accelerometerAccelerationX(G)'], data['accelerometerAccelerationY(G)'], data['c_fr_array'], data['c_rr_array'], data['timestep'], data['motionRoll(rad)'], data['motionPitch(rad)'])), \
        columns=columns_global)

    print(data)

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
    c_fr_array[130:160] = -0.018  # m
    c_fr_array = custom_smooth(c_fr_array, 3)

    c_rr_array = np.array([0.0 for x in range(time_res * timespan)])
    c_rr_array[150:180] = -0.020  # m
    c_rr_array = custom_smooth(c_rr_array, 3)

    time_array = [x/time_res for x in range(len(G_lat_array))]
    dt_array = [1/time_res for all in range(len(G_lat_array))]

    control_array_roll = np.array([0.0 for x in range(time_res * timespan)])
    control_array_pitch = np.array([0.0 for x in range(time_res * timespan)])

    data = pd.DataFrame(list(zip(time_array, G_lat_array, G_long_array, c_fr_array, c_rr_array, dt_array, control_array_roll, control_array_pitch)), \
        columns=columns_global)

    print(len(data))

    return data

def get_bump_function(timespan: int = 9, magnitude: float = 0.020, frequency: float = 0.5):

    #  Default time resolution is set to 100hz
    time_res = 100  # hz
    dt = 1/time_res

    c_array = [math.sin(2 * math.pi * frequency * x / time_res) * magnitude for x in range(time_res * timespan)]  # 100 steps is 1s
    time_array = [x/time_res for x in range(len(c_array))]

    return time_array, c_array, dt