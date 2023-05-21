import pandas as pd
import math
import numpy as np

def from_sensor_log_iOS_app(path: str):

    print('Converting file to dataframe...')
    data = pd.read_csv(path)[['loggingTime(txt)', 'accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'motionRoll(rad)', 'motionPitch(rad)']]

    print('Parsing timesteps...')
    data['datetime'] = pd.to_datetime(data['loggingTime(txt)'])
    data['timestep'] = data['datetime'].diff()
    data['timestep'][0] = data['timestep'][1]
    for i, timestep in enumerate(data['timestep']):
        data['timestep'][i] = int(str(data['timestep'][i])[-6:]) / 1000000  # us to s

    return data[3200:5400].reset_index()

def get_G_function(
        timespan: int = 3,
        lat_magnitude: float = 1.4, lat_frequency: float = 0.5,
        long_magnitude: float = -0.6, long_frequency: float = 1
    ):

    #  Default time resolution is set to 100hz
    time_res = 100  # hz

    G_lat_array = np.array([math.sin(2 * math.pi * lat_frequency * x / time_res) * lat_magnitude for x in range(time_res * timespan)]) # 100 steps is 1s
    G_lat_array[200:] = 0
    G_long_array = np.array([math.sin(2 * math.pi * long_frequency * x / time_res) * long_magnitude for x in range(time_res * timespan)])  # 100 steps is 1s
    G_long_array[175:] = -long_magnitude
    time_array = [x/time_res for x in range(len(G_lat_array))]
    dt_array = [1/time_res for all in range(len(G_lat_array))]

    data = pd.DataFrame(list(zip(time_array, G_lat_array, G_long_array, dt_array)), columns=['loggingTime(txt)', 'accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'timestep'])

    print(len(data))

    return data

def get_bump_function(timespan: int = 9, magnitude: float = 0.020, frequency: float = 0.5):

    #  Default time resolution is set to 100hz
    time_res = 100  # hz
    dt = 1/time_res

    c_array = [math.sin(2 * math.pi * frequency * x / time_res) * magnitude for x in range(time_res * timespan)]  # 100 steps is 1s
    time_array = [x/time_res for x in range(len(c_array))]

    return time_array, c_array, dt