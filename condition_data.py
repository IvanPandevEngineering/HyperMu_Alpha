import pandas as pd
import math

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

def get_bump_function(timespan: int = 9, magnitude: float = 0.020, frequency: float = 0.5):

    #  Default time resolution is set to 100hz
    time_res = 100  # hz
    dt = 1/time_res

    c_array = [math.sin(2 * math.pi * frequency * x / 100) * magnitude for x in range(time_res * timespan)]  # 100 steps is 1s
    time_array = [x/time_res for x in range(len(c_array))]

    return time_array, c_array, dt