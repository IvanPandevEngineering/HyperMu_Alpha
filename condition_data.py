import pandas as pd
import math
import numpy as np
from scipy.signal import bessel, butter, filtfilt

# define standard dataframe format for multiple data generation functions
columns_global=['loggingTime(txt)',
                'accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)',
                'c_fr', 'c_rr',
                'timestep',
                'gyroRotationY(rad/s)', 'gyroRotationX(rad/s)', 'gyroRotationZ(rad/s)',
                'gyroRotationZ_diff(rad/s)', 'gyroRotationX_corrected(rad/s)']

def custom_smooth(array, rounds):
    
    for all in range(rounds):
        array = np.convolve(array, [0.025, 0.025, 0.075, 0.075, 0.30, 0.30, 0.075, 0.075, 0.025, 0.025], 'same')
    
    return(array)

def apply_filter(data, filter_type, smoothing_window_size):
    if filter_type == 'bidirectional_bessel':
        data['motionUserAccelerationX(G)'] = bidirectional_bessel_lowpass(data['motionUserAccelerationX(G)'])
        data['motionUserAccelerationY(G)'] = bidirectional_bessel_lowpass(data['motionUserAccelerationY(G)'])
        data['motionUserAccelerationZ(G)'] = bidirectional_bessel_lowpass(data['motionUserAccelerationZ(G)'])
        data['motionRotationRateY(rad/s)'] = bidirectional_bessel_lowpass(data['motionRotationRateY(rad/s)'])
        data['motionRotationRateX(rad/s)'] = bidirectional_bessel_lowpass(data['motionRotationRateX(rad/s)'])
        data['motionRotationRateZ(rad/s)'] = bidirectional_bessel_lowpass(data['motionRotationRateZ(rad/s)'])
        data = data.dropna(how='any')
        print('Applying lowpass filter (bessel_bidirectional)...')
    
    elif filter_type == 'bidirectional_butterworth_lowpass':
        data['motionUserAccelerationX(G)'] = bidirectional_butterworth_lowpass(data['motionUserAccelerationX(G)'])
        data['motionUserAccelerationY(G)'] = bidirectional_butterworth_lowpass(data['motionUserAccelerationY(G)'])
        data['motionUserAccelerationZ(G)'] = bidirectional_butterworth_lowpass(data['motionUserAccelerationZ(G)'])
        data['motionRotationRateY(rad/s)'] = bidirectional_butterworth_lowpass(data['motionRotationRateY(rad/s)'])
        data['motionRotationRateX(rad/s)'] = bidirectional_butterworth_lowpass(data['motionRotationRateX(rad/s)'])
        data['motionRotationRateZ(rad/s)'] = bidirectional_butterworth_lowpass(data['motionRotationRateZ(rad/s)'])
        print('Applying lowpass filter (butterworth_bidirectional)...')

    elif filter_type == 'pandas_rolling':
        data['motionUserAccelerationX(G)'] = data['motionUserAccelerationX(G)'].rolling(window = smoothing_window_size, center = False).mean()
        data['motionUserAccelerationY(G)'] = data['motionUserAccelerationY(G)'].rolling(window = smoothing_window_size, center = False).mean()
        data['motionUserAccelerationZ(G)'] = data['motionUserAccelerationZ(G)'].rolling(window = smoothing_window_size, center = False).mean()
        data['motionRotationRateY(rad/s)'] = data['motionRotationRateY(rad/s)'].rolling(window = smoothing_window_size, center = False).mean()
        data['motionRotationRateX(rad/s)'] = data['motionRotationRateX(rad/s)'].rolling(window = smoothing_window_size, center = False).mean()
        data['motionRotationRateZ(rad/s)'] = data['motionRotationRateZ(rad/s)'].rolling(window = smoothing_window_size, center = False).mean()
        print('Applying lowpass filter (pandas rolling)...')

    else:
        print('filter_type not recognized.')
        return

    return data.dropna(how='any')

def yaw_rate_correction(data, pitch_installation_angle_deg):

    installed_pitch_angle = pitch_installation_angle_deg*math.pi/180  # Convert pitch installation angle to rad
    data['gyroRotationX_corrected(rad/s)'] = data['motionRotationRateX(rad/s)'] + 2*np.sin(installed_pitch_angle)*abs(1-np.cos(data['motionRotationRateZ(rad/s)']))  # pitch rate correction by yaw and roll rates
    
    return data

def bidirectional_butterworth_lowpass(signal, order = 2, cutoff_freq = 0.7, sampling_freq = 1000):

    nyquist = sampling_freq / 2
    b, a = butter(order, cutoff_freq / nyquist, btype='low')

    return filtfilt(b, a, signal)

def bidirectional_bessel_lowpass(signal, order = 4, cutoff_freq = 0.7, sampling_freq = 1000):

    nyquist = sampling_freq / 2
    b, a = bessel(order, cutoff_freq / nyquist, btype='low', analog=False)

    return filtfilt(b, a, signal)

def from_sensor_log_iOS_app_unbiased(path: str, smoothing_window_size_ms:int):

    print('Converting file to dataframe...')
    data_in = pd.read_csv(path, low_memory=False)[['loggingTime(txt)',
                                                   'motionUserAccelerationX(G)',
                                                   'motionUserAccelerationY(G)',
                                                   'motionUserAccelerationZ(G)',
                                                   'motionRotationRateY(rad/s)',
                                                   'motionRotationRateX(rad/s)',
                                                   'motionRotationRateZ(rad/s)']]

    print('Parsing timesteps...')

    #Create datetime column to be interpolated
    data_in['datetime'] = pd.to_datetime(data_in['loggingTime(txt)'])
    data_in['datetime'] = pd.DatetimeIndex(data_in['datetime'])

    #drop redundant column
    data_in = data_in.drop(columns='loggingTime(txt)')

    #select interesting time range
    data_in = data_in[2400:4200]

    #set index to be picked up by interpolation function, drop duplicated time stamps
    data_in = data_in.set_index('datetime')
    data_in = data_in[~data_in.index.duplicated(keep='first')]

    data_in['c_fr_array'] = 0
    data_in['c_rr_array'] = 0

    #resampling to time resolution, interpolate linearly then drop all nans
    data_in = data_in.resample('1ms').interpolate(method='linear')
    data_in = data_in.dropna(how='any')

    #get derivative of yaw rate
    data_in['motionRotationRateZ_diff(rad/s)'] = data_in['motionRotationRateZ(rad/s)'].diff()/0.001
    data_in = data_in.dropna(how='any')

    data_in = apply_filter(
        data = data_in,
        filter_type = 'bidirectional_bessel',
        smoothing_window_size = smoothing_window_size_ms
    )

    #apply vertical-offset corrections
    data_in['motionUserAccelerationX(G)'] = data_in['motionUserAccelerationX(G)'] - 0.03
    data_in['motionUserAccelerationY(G)'] = data_in['motionUserAccelerationY(G)'] - 0.01

    data_in = yaw_rate_correction(
        data = data_in,
        installed_pitch_angle_deg = 5
    )

    #create new time and timestep columns
    data_in['time'] = data_in.index
    data_in['timestep'] = data_in['time'].diff().dt.total_seconds()

    #create dataframe and drop nans one more time
    data = pd.DataFrame(list(zip(data_in['time'],
                                 data_in['motionUserAccelerationX(G)'],
                                 data_in['motionUserAccelerationY(G)'],
                                 data_in['motionUserAccelerationZ(G)'],
                                 data_in['c_fr_array'],
                                 data_in['c_rr_array'],
                                 data_in['timestep'],
                                 data_in['motionRotationRateY(rad/s)'],
                                 data_in['motionRotationRateX(rad/s)'],
                                 data_in['motionRotationRateZ(rad/s)'],
                                 data_in['motionRotationRateZ_diff(rad/s)'],
                                 data_in['gyroRotationX_corrected(rad/s)'])), \
        columns=columns_global)
    
    return data.dropna(how='any').reset_index(drop=True)

def get_unit_test_Slalom_w_Curbs(
        timespan: int = 3,
        lat_magnitude: float = 1.2, lat_frequency: float = 0.5,
        long_magnitude: float = 0.5, long_frequency: float = 1
    ):

    #  Default time resolution is set to 100hz
    time_res = 1000  # hz

    G_lat_array = np.array([math.sin(2 * math.pi * lat_frequency * x / time_res) * lat_magnitude for x in range(time_res * timespan)]) # 100 steps is 1s
    G_lat_array[2000:] = 0

    G_long_array = np.array([math.sin(2 * math.pi * long_frequency * x / time_res) * long_magnitude for x in range(time_res * timespan)])  # 100 steps is 1s
    G_long_array[1750:] = -long_magnitude

    c_fr_array = np.array([0.0 for x in range(time_res * timespan)])
    c_fr_array[1300:1600] = -0.008  # m
    c_fr_array = custom_smooth(c_fr_array, 10)

    c_rr_array = np.array([0.0 for x in range(time_res * timespan)])
    c_rr_array[1500:1800] = -0.010  # m
    c_rr_array = custom_smooth(c_rr_array, 10)

    time_array = [x/time_res for x in range(len(G_lat_array))]
    dt_array = [1/time_res for all in range(len(G_lat_array))]

    control_array_Gz = np.array([0.0 for x in range(time_res * timespan)])
    control_array_roll = np.array([0.0 for x in range(time_res * timespan)])
    control_array_pitch = np.array([0.0 for x in range(time_res * timespan)])
    control_array_yaw = np.array([0.0 for x in range(time_res * timespan)])
    control_array_pitch_corrected = np.array([0.0 for x in range(time_res * timespan)])

    data = pd.DataFrame(list(zip(time_array, G_lat_array, G_long_array, control_array_Gz,
        c_fr_array, c_rr_array, dt_array, control_array_roll, control_array_pitch, control_array_yaw, control_array_pitch_corrected)),
        columns=columns_global)

    return data

def get_unit_test_Curbs(
        timespan: int = 10,
        lat_magnitude: float = 1.2, lat_frequency: float = 0.5,
        long_magnitude: float = 0.5, long_frequency: float = 1
    ):

    #  Default time resolution is set to 100hz
    time_res = 1000  # hz

    G_lat_array = np.array([0.0 for x in range(time_res * timespan)])
    G_long_array = np.array([0.0 for x in range(time_res * timespan)])

    c_fr_array = np.array([0.0 for x in range(time_res * timespan)])
    c_fr_array[1300:1600] = -0.008  # m
    c_fr_array[5300:5600] = -0.012  # m
    c_fr_array = custom_smooth(c_fr_array, 10)

    c_rr_array = np.array([0.0 for x in range(time_res * timespan)])
    c_rr_array[1500:1800] = -0.010  # m
    c_rr_array[7500:7800] = -0.020  # m
    c_rr_array = custom_smooth(c_rr_array, 10)

    time_array = [x/time_res for x in range(len(G_lat_array))]
    dt_array = [1/time_res for all in range(len(G_lat_array))]

    control_array_Gz = np.array([0.0 for x in range(time_res * timespan)])
    control_array_roll = np.array([0.0 for x in range(time_res * timespan)])
    control_array_pitch = np.array([0.0 for x in range(time_res * timespan)])
    control_array_yaw = np.array([0.0 for x in range(time_res * timespan)])
    control_array_pitch_corrected = np.array([0.0 for x in range(time_res * timespan)])

    data = pd.DataFrame(list(zip(time_array, G_lat_array, G_long_array, control_array_Gz,
        c_fr_array, c_rr_array, dt_array, control_array_roll, control_array_pitch, control_array_yaw, control_array_pitch_corrected)),
        columns=columns_global)

    return data

def get_unit_test_Roll_Harmonic_Sweep(
        timespan: int = 15,
        lat_magnitude: float = 0.5, lat_frequency: float = 0.5,
        long_magnitude: float = 0.5, long_frequency: float = 1
    ):

    #  Default time resolution is set to 100hz
    time_res = 1000  # hz

    G_lat_array = np.array([math.sin(0.000001 * t**2) * lat_magnitude for t in range(time_res * timespan)]) # 100 steps is 1s
    G_long_array = np.array([0.0 for t in range(time_res * timespan)])  # 100 steps is 1s

    c_fr_array = np.array([0.0 for x in range(time_res * timespan)])
    c_rr_array = np.array([0.0 for x in range(time_res * timespan)])

    time_array = [x/time_res for x in range(len(G_lat_array))]
    dt_array = [1/time_res for all in range(len(G_lat_array))]

    control_array_Gz = np.array([0.0 for x in range(time_res * timespan)])
    control_array_roll = np.array([0.0 for x in range(time_res * timespan)])
    control_array_pitch = np.array([0.0 for x in range(time_res * timespan)])
    control_array_yaw = np.array([0.0 for x in range(time_res * timespan)])
    control_array_pitch_corrected = np.array([0.0 for x in range(time_res * timespan)])

    data = pd.DataFrame(list(zip(time_array, G_lat_array, G_long_array, control_array_Gz,
        c_fr_array, c_rr_array, dt_array, control_array_roll, control_array_pitch, control_array_yaw, control_array_pitch_corrected)),
        columns=columns_global)

    return data


'''
Begin functions for dev purposes, exploring telemetry data.
'''
def get_bump_function(timespan: int = 9, magnitude: float = 0.020, frequency: float = 0.5):

    #  Default time resolution is set to 100hz
    time_res = 100  # hz
    dt = 1/time_res

    c_array = [math.sin(2 * math.pi * frequency * x / time_res) * magnitude for x in range(time_res * timespan)]  # 100 steps is 1s
    time_array = [x/time_res for x in range(len(c_array))]

    return time_array, c_array, dt

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