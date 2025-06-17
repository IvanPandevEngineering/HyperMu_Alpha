from collections import namedtuple
import math
import numpy as np
import pandas as pd
from scipy.signal import bessel, butter, filtfilt

import formulas as f
import visualizer as vis

# define standard dataframe format for multiple data generation functions
COLUMNS_GLOBAL=['loggingTime(txt)',
                'accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)',
                'motionPitch(rad)',
                'c_fr', 'c_fl', 'c_rr', 'c_rl',
                'timestep',
                'gyroRotationY(rad/s)', 'gyroRotationX(rad/s)', 'gyroRotationZ(rad/s)',
                'gyroRotationZ_diff(rad/s)', 'gyroRotationX_corrected(rad/s)', 'calc_speed_ms']

force_function_namedTuple = namedtuple('force_function_namedTuple',
    ['time',
    'G_lat', 'G_long', 'G_vert',
    'pitch_rad',
    'c_fr', 'c_fl', 'c_rr', 'c_rl',
    'timestep',
    'gyro_roll_radps', 'gyro_pitch_radps', 'gyro_yaw_radps',
    'gyro_yaw_diff_radps', 'gyro_pitch_corrected_radps', 'calc_speed_ms']
)

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
        print('filter_type argument not recognized. Returning unfiltered input.')

    return data.dropna(how='any')

def yaw_rate_correction(data, pitch_installation_angle_deg):

    live_pitch_angle = pitch_installation_angle_deg*math.pi/180 #- 10*data['motionPitch(rad)'] +   # Convert pitch installation angle to rad

    # live_pitch_angle = data['motionUserAccelerationY(G)'] # use long acceleration as a proxy for pitch angle
    #data['gyroRotationX_corrected(rad/s)'] = data['motionRotationRateX(rad/s)'] + abs(0.28*data['motionRotationRateZ(rad/s)'])**2
    
    data['gyroRotationX_corrected(rad/s)'] = data['motionRotationRateX(rad/s)'] - np.sin(live_pitch_angle)*abs(1-np.cos(data['motionRotationRateZ(rad/s)']))  # pitch rate correction by yaw and roll rates
    #TODO: APPLY LATERALLY AS WELL???

    return data

def bidirectional_butterworth_lowpass(signal, order = 2, cutoff_freq = 0.7, sampling_freq = f.FREQ_DATA):

    nyquist = sampling_freq / 2
    b, a = butter(order, cutoff_freq / nyquist, btype='low')

    return filtfilt(b, a, signal)

def bidirectional_bessel_lowpass(signal, order = 5, cutoff_freq = 1.2, sampling_freq = f.FREQ_DATA):

    nyquist = sampling_freq / 2
    b, a = bessel(order, cutoff_freq / nyquist, btype='low', analog=False)

    return filtfilt(b, a, signal)

def from_sensor_log_iOS_app_unbiased(path:str, filter_type:str, smoothing_window_size_ms:int, start_index:int, end_index:int):

    print('Converting file to dataframe...')
    data_in = pd.read_csv(path, low_memory=False)[['loggingTime(txt)',
                                                   'accelerometerAccelerationX(G)',
                                                   'accelerometerAccelerationY(G)',
                                                   'accelerometerAccelerationZ(G)',
                                                   'motionPitch(rad)',
                                                   'gyroRotationY(rad/s)',
                                                   'gyroRotationX(rad/s)',
                                                   'gyroRotationZ(rad/s)']]

    print('Parsing timesteps...')

    data_in['motionRotationRateY(rad/s)'] = data_in['gyroRotationY(rad/s)']
    data_in['motionRotationRateX(rad/s)'] = data_in['gyroRotationX(rad/s)']
    data_in['motionRotationRateZ(rad/s)'] = data_in['gyroRotationZ(rad/s)']
    data_in['motionUserAccelerationX(G)'] = data_in['accelerometerAccelerationX(G)']
    data_in['motionUserAccelerationY(G)'] = data_in['accelerometerAccelerationY(G)']
    data_in['motionUserAccelerationZ(G)'] = data_in['accelerometerAccelerationZ(G)']

    #Create datetime column to be interpolated
    data_in['datetime'] = pd.to_datetime(data_in['loggingTime(txt)'])
    data_in['datetime'] = pd.DatetimeIndex(data_in['datetime'])

    #drop redundant column
    data_in = data_in.drop(columns='loggingTime(txt)')

    #set index to be picked up by interpolation function, drop duplicated time stamps
    data_in = data_in.set_index('datetime')
    data_in = data_in[~data_in.index.duplicated(keep='first')]

    #select interesting time range
    data_in = data_in[start_index:end_index]

    data_in['c_fr_array'] = 0
    data_in['c_fl_array'] = 0
    data_in['c_rr_array'] = 0
    data_in['c_rl_array'] = 0

    #resampling to time resolution, interpolate linearly then drop all nans
    data_in = data_in.resample(f'{f.PERIOD_DATA_MS}ms').interpolate(method='cubic', order=3)
    data_in = data_in.dropna(how='any')

    #get derivative of yaw rate
    data_in['motionRotationRateZ_diff(rad/s)'] = data_in['motionRotationRateZ(rad/s)'].diff()/f.PERIOD_DATA
    data_in = data_in.dropna(how='any')

    data_in['calc_speed_ms'] = np.cumsum(
        0.5 * (data_in['motionUserAccelerationY(G)'] + data_in['motionUserAccelerationY(G)'].shift(1)) * (f.G / f.FREQ_DATA))

    data_in = apply_filter(
        data = data_in,
        filter_type = filter_type,
        smoothing_window_size = smoothing_window_size_ms
    )

    #apply vertical-offset corrections
    data_in['motionUserAccelerationX(G)'] = data_in['motionUserAccelerationX(G)'] - 0.03
    data_in['motionUserAccelerationY(G)'] = data_in['motionUserAccelerationY(G)'] - 0.01

    # data_in['motionPitch(rad)'] = 

    data_in = yaw_rate_correction(
        data = data_in,
        pitch_installation_angle_deg = 5
    )

    #create new time and timestep columns
    data_in['time'] = data_in.index
    data_in['timestep'] = data_in['time'].diff().dt.total_seconds()

    #print(f"ROLL RMS: {f.get_RMS(data_in['motionRotationRateY(rad/s)'])}")
    #print(f"PITCH RMS: {f.get_RMS(data_in['motionRotationRateX(rad/s)'])}")

    #create dataframe and drop nans one more time
    data = pd.DataFrame(list(zip(data_in['time'],
                                 data_in['motionUserAccelerationX(G)'],
                                 data_in['motionUserAccelerationY(G)'],
                                 data_in['motionUserAccelerationZ(G)'],
                                 data_in['motionPitch(rad)'],
                                 data_in['c_fr_array'],
                                 data_in['c_fl_array'],
                                 data_in['c_rr_array'],
                                 data_in['c_rl_array'],
                                 data_in['timestep'],
                                 data_in['motionRotationRateY(rad/s)'],
                                 data_in['motionRotationRateX(rad/s)'],
                                 data_in['motionRotationRateZ(rad/s)'],
                                 data_in['motionRotationRateZ_diff(rad/s)'],
                                 data_in['gyroRotationX_corrected(rad/s)'],
                                 data_in['calc_speed_ms'])), \
        columns=COLUMNS_GLOBAL).dropna(how='any').reset_index(drop=True)

    return force_function_namedTuple(
        time = np.array(data['loggingTime(txt)']),
        G_lat = np.array(data['accelerometerAccelerationX(G)']),
        G_long = np.array(data['accelerometerAccelerationY(G)']),
        G_vert = np.array(data['accelerometerAccelerationZ(G)']),
        pitch_rad = np.array(data['motionPitch(rad)']),
        c_fr = np.array(data['c_fr']),
        c_fl = np.array(data['c_fl']),
        c_rr = np.array(data['c_rr']),
        c_rl = np.array(data['c_rl']),
        timestep = np.array(data['timestep']),
        gyro_roll_radps = np.array(data['gyroRotationY(rad/s)']),
        gyro_pitch_radps = np.array(data['gyroRotationX(rad/s)']),
        gyro_yaw_radps = np.array(data['gyroRotationZ(rad/s)']),
        gyro_yaw_diff_radps = np.array(data['gyroRotationZ_diff(rad/s)']),
        gyro_pitch_corrected_radps = np.array(data['gyroRotationX_corrected(rad/s)']),
        calc_speed_ms = np.array(data['calc_speed_ms'])
    )

def from_RaceBox(path:str, filter_type:str, smoothing_window_size_ms:int, start_index:int, end_index:int):

    print('Converting file to dataframe...')
    data_in = pd.read_csv(path, low_memory=False)[['loggingTime(txt)',
                                                   'GForceX',
                                                   'GForceY',
                                                   'GForceZ',
                                                   'Speed',
                                                   'GyroX',
                                                   'GyroY',
                                                   'GyroZ']]

    print('Parsing timesteps...')

    data_in['motionRotationRateY(rad/s)'] = data_in['GyroX']*(np.pi/180)
    data_in['motionRotationRateX(rad/s)'] = data_in['GyroY']*(np.pi/180)
    data_in['motionRotationRateZ(rad/s)'] = data_in['GyroZ']*(np.pi/180)
    data_in['motionPitch(rad)'] = data_in['GyroX']
    data_in['motionUserAccelerationX(G)'] = data_in['GForceY']
    data_in['motionUserAccelerationY(G)'] = data_in['GForceX']
    data_in['motionUserAccelerationZ(G)'] = data_in['GForceZ']

    #Create datetime column to be interpolated
    data_in['datetime'] = pd.to_datetime(data_in['loggingTime(txt)'])
    data_in['datetime'] = pd.DatetimeIndex(data_in['datetime'])

    #drop redundant column
    data_in = data_in.drop(columns='loggingTime(txt)')

    #set index to be picked up by interpolation function, drop duplicated time stamps
    data_in = data_in.set_index('datetime')
    data_in = data_in[~data_in.index.duplicated(keep='first')]

    #select interesting time range
    data_in = data_in[start_index:end_index]

    #resampling to time resolution, interpolate linearly then drop all nans
    data_in = data_in.resample(f'{f.PERIOD_DATA_MS}ms').interpolate(method='cubic', order=3)
    data_in = data_in.dropna(how='any')

    data_in['c_fr_array'] = 0
    data_in['c_fl_array'] = 0
    data_in['c_rr_array'] = 0
    data_in['c_rl_array'] = 0

    #get derivative of yaw rate
    data_in['motionRotationRateZ_diff(rad/s)'] = data_in['motionRotationRateZ(rad/s)'].diff()/f.PERIOD_DATA
    data_in = data_in.dropna(how='any')

    data_in['calc_speed_ms'] = np.cumsum(
        -0.5 * (data_in['motionUserAccelerationY(G)'] + data_in['motionUserAccelerationY(G)'].shift(1)) * (f.G / f.FREQ_DATA))

    data_in = apply_filter(
        data = data_in,
        filter_type = filter_type,
        smoothing_window_size = smoothing_window_size_ms
    )

    #apply vertical-offset corrections
    data_in['motionUserAccelerationX(G)'] = data_in['motionUserAccelerationX(G)'] - 0.03
    data_in['motionUserAccelerationY(G)'] = data_in['motionUserAccelerationY(G)'] - 0.01

    #create new time and timestep columns
    data_in['time'] = data_in.index
    data_in['timestep'] = data_in['time'].diff().dt.total_seconds()

    #print(f"ROLL RMS: {f.get_RMS(data_in['motionRotationRateY(rad/s)'])}")
    #print(f"PITCH RMS: {f.get_RMS(data_in['motionRotationRateX(rad/s)'])}")

    # data_in['gyroRotationX_corrected(rad/s)'] = data_in['motionRotationRateX(rad/s)']

    data_in = yaw_rate_correction(
        data = data_in,
        pitch_installation_angle_deg = 6
    )

    #create dataframe and drop nans one more time
    data = pd.DataFrame(list(zip(data_in['time'],
                                 data_in['motionUserAccelerationX(G)'],
                                 data_in['motionUserAccelerationY(G)'],
                                 data_in['motionUserAccelerationZ(G)'],
                                 data_in['motionPitch(rad)'],
                                 data_in['c_fr_array'],
                                 data_in['c_fl_array'],
                                 data_in['c_rr_array'],
                                 data_in['c_rl_array'],
                                 data_in['timestep'],
                                 data_in['motionRotationRateY(rad/s)'],
                                 data_in['motionRotationRateX(rad/s)'],
                                 data_in['motionRotationRateZ(rad/s)'],
                                 data_in['motionRotationRateZ_diff(rad/s)'],
                                 data_in['gyroRotationX_corrected(rad/s)'],
                                 data_in['calc_speed_ms'])), \
        columns=COLUMNS_GLOBAL).dropna(how='any').reset_index(drop=True)

    return force_function_namedTuple(
        time = np.array(data['loggingTime(txt)']),
        G_lat = np.array(data['accelerometerAccelerationX(G)']),
        G_long = np.array(data['accelerometerAccelerationY(G)']),
        G_vert = np.array(data['accelerometerAccelerationZ(G)']),
        pitch_rad = np.array(data['motionPitch(rad)']),
        c_fr = np.array(data['c_fr']),
        c_fl = np.array(data['c_fl']),
        c_rr = np.array(data['c_rr']),
        c_rl = np.array(data['c_rl']),
        timestep = np.array(data['timestep']),
        gyro_roll_radps = np.array(data['gyroRotationY(rad/s)']),
        gyro_pitch_radps = np.array(data['gyroRotationX(rad/s)']),
        gyro_yaw_radps = np.array(data['gyroRotationZ(rad/s)']),
        gyro_yaw_diff_radps = np.array(data['gyroRotationZ_diff(rad/s)']),
        gyro_pitch_corrected_radps = np.array(data['gyroRotationX_corrected(rad/s)']),
        calc_speed_ms = np.array(data['calc_speed_ms'])
    )

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

    empty = np.array([0.0 for x in range(time_res * timespan)])

    data = pd.DataFrame(list(zip(time_array, G_lat_array, G_long_array, empty, empty,
        c_fr_array, empty, c_rr_array, empty,
        dt_array, empty, empty, empty, empty, empty)),
        columns=COLUMNS_GLOBAL)

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
    c_fr_array[1300:1600] = -0.022  # m
    c_fr_array[5300:5600] = -0.024  # m
    c_fr_array = custom_smooth(c_fr_array, 10)

    c_fl_array = np.array([0.0 for x in range(time_res * timespan)])
    c_fl_array[5400:6000] = -0.030  # m
    c_fl_array = custom_smooth(c_fl_array, 10)

    c_rr_array = np.array([0.0 for x in range(time_res * timespan)])
    c_rr_array[2500:2800] = -0.015  # m
    c_rr_array[7500:7800] = -0.020  # m
    c_rr_array = custom_smooth(c_rr_array, 10)

    c_rl_array = np.array([0.0 for x in range(time_res * timespan)])
    c_rl_array[6200:7200] = -0.030  # m
    c_rl_array = custom_smooth(c_rl_array, 10)

    time_array = [x/time_res for x in range(len(G_lat_array))]
    dt_array = [1/time_res for all in range(len(G_lat_array))]

    empty = np.array([0.0 for x in range(time_res * timespan)])

    data = pd.DataFrame(list(zip(time_array, G_lat_array, G_long_array, empty,
        empty,
        c_fr_array, c_fl_array, c_rr_array, c_rl_array,
        dt_array,
        empty, empty, empty,
        empty, empty, empty)),
        columns=COLUMNS_GLOBAL)

    return force_function_namedTuple(
        time = np.array(data['loggingTime(txt)']),
        G_lat = np.array(data['accelerometerAccelerationX(G)']),
        G_long = np.array(data['accelerometerAccelerationY(G)']),
        G_vert = np.array(data['accelerometerAccelerationZ(G)']),
        pitch_rad = np.array(data['motionPitch(rad)']),
        c_fr = np.array(data['c_fr']),
        c_fl = np.array(data['c_fl']),
        c_rr = np.array(data['c_rr']),
        c_rl = np.array(data['c_rl']),
        timestep = np.array(data['timestep']),
        gyro_roll_radps = np.array(data['gyroRotationY(rad/s)']),
        gyro_pitch_radps = np.array(data['gyroRotationX(rad/s)']),
        gyro_yaw_radps = np.array(data['gyroRotationZ(rad/s)']),
        gyro_yaw_diff_radps = np.array(data['gyroRotationZ_diff(rad/s)']),
        gyro_pitch_corrected_radps = np.array(data['gyroRotationX_corrected(rad/s)']),
        calc_speed_ms = np.array(data['calc_speed_ms'])
    )

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

    empty = np.array([0.0 for x in range(time_res * timespan)])

    data = pd.DataFrame(list(zip(time_array, G_lat_array, G_long_array, empty, empty,
        c_fr_array, empty, c_rr_array, empty, dt_array, empty, empty, empty, empty, empty)),
        columns=COLUMNS_GLOBAL)

    return data

def get_unit_test_warp(
        warp_mag,
        warp_corner
    ):

    #  Default time resolution is set to 100hz
    warp_mag = -warp_mag
    timespan = 7 # s
    time_res = 1000  # hz

    G_lat_array = np.array([0.0 for x in range(time_res * timespan)])
    G_long_array = np.array([0.0 for x in range(time_res * timespan)])

    c_fr_array = np.array([0.0 for x in range(time_res * timespan)])
    c_fl_array = np.array([0.0 for x in range(time_res * timespan)])
    c_rr_array = np.array([0.0 for x in range(time_res * timespan)])
    c_rl_array = np.array([0.0 for x in range(time_res * timespan)])

    if warp_corner == 'FR':
        c_fr_array[1000:3000] = np.linspace(0, warp_mag, 2000)
        c_fr_array[3000:] = warp_mag
        c_fr_array = custom_smooth(c_fr_array, 1000)
        c_fr_array[4000:] = warp_mag
    elif warp_corner == 'FL':
        c_fl_array[1000:3000] = np.linspace(0, warp_mag, 2000)
        c_fl_array[3000:] = warp_mag
        c_fl_array = custom_smooth(c_fl_array, 1000)
        c_fl_array[4000:] = warp_mag
    elif warp_corner == 'RR':
        c_rr_array[1000:3000] = np.linspace(0, warp_mag, 2000)
        c_rr_array[3000:] = warp_mag
        c_rr_array = custom_smooth(c_rr_array, 1000)
        c_rr_array[4000:] = warp_mag
    elif warp_corner == 'RL':
        c_rl_array[1000:3000] = np.linspace(0, warp_mag, 2000)
        c_rl_array[3000:] = warp_mag
        c_rl_array = custom_smooth(c_rl_array, 1000)
        c_rl_array[4000:] = warp_mag
    else:
        print('Specify single corner in format "FR".')
        return None

    time_array = [x/time_res for x in range(len(G_lat_array))]
    dt_array = [1/time_res for all in range(len(G_lat_array))]

    control_array_Gz = np.array([0.0 for x in range(time_res * timespan)])
    control_array_roll_rate = np.array([0.0 for x in range(time_res * timespan)])
    control_array_pitch_rate = np.array([0.0 for x in range(time_res * timespan)])
    control_array_yaw_rate = np.array([0.0 for x in range(time_res * timespan)])
    control_array_pitch_rate_corrected = np.array([0.0 for x in range(time_res * timespan)])
    control_array_pitch_accel_corrected = np.array([0.0 for x in range(time_res * timespan)])

    data = pd.DataFrame(list(zip(time_array, G_lat_array, G_long_array, control_array_Gz,
        c_fr_array, c_fl_array, c_rr_array, c_rl_array,
        dt_array,
        control_array_roll_rate, control_array_pitch_rate, control_array_yaw_rate,
        control_array_pitch_rate_corrected, control_array_pitch_accel_corrected)),
        columns=COLUMNS_GLOBAL)

    return data

def get_init_empty():

    timespan = 5 # s
    time_res = 250  # hz

    G_lat_array = np.array([0.0 for x in range(time_res * timespan)])
    G_long_array = np.array([0.0 for x in range(time_res * timespan)])

    pitch_array = np.array([0.0 for x in range(time_res * timespan)])

    c_fr_array = np.array([0.0 for x in range(time_res * timespan)])
    c_fl_array = np.array([0.0 for x in range(time_res * timespan)])
    c_rr_array = np.array([0.0 for x in range(time_res * timespan)])
    c_rl_array = np.array([0.0 for x in range(time_res * timespan)])

    time_array = [x/time_res for x in range(len(G_lat_array))]
    dt_array = [1/time_res for all in range(len(G_lat_array))]

    control_array_Gz = np.array([0.0 for x in range(time_res * timespan)])
    control_array_roll_rate = np.array([0.0 for x in range(time_res * timespan)])
    control_array_pitch_rate = np.array([0.0 for x in range(time_res * timespan)])
    control_array_yaw_rate = np.array([0.0 for x in range(time_res * timespan)])
    control_array_pitch_rate_corrected = np.array([0.0 for x in range(time_res * timespan)])
    control_array_pitch_accel_corrected = np.array([0.0 for x in range(time_res * timespan)])
    empty = np.array([0.0 for x in range(time_res * timespan)])

    data = pd.DataFrame(list(zip(time_array, G_lat_array, G_long_array, control_array_Gz,
        pitch_array,
        c_fr_array, c_fl_array, c_rr_array, c_rl_array,
        dt_array,
        control_array_roll_rate, control_array_pitch_rate, control_array_yaw_rate,
        control_array_pitch_rate_corrected, control_array_pitch_accel_corrected, empty)),
        columns=COLUMNS_GLOBAL)

    return force_function_namedTuple(
        time = np.array(data['loggingTime(txt)']),
        G_lat = np.array(data['accelerometerAccelerationX(G)']),
        G_long = np.array(data['accelerometerAccelerationY(G)']),
        G_vert = np.array(data['accelerometerAccelerationZ(G)']),
        pitch_rad = np.array(data['motionPitch(rad)']),
        c_fr = np.array(data['c_fr']),
        c_fl = np.array(data['c_fl']),
        c_rr = np.array(data['c_rr']),
        c_rl = np.array(data['c_rl']),
        timestep = np.array(data['timestep']),
        gyro_roll_radps = np.array(data['gyroRotationY(rad/s)']),
        gyro_pitch_radps = np.array(data['gyroRotationX(rad/s)']),
        gyro_yaw_radps = np.array(data['gyroRotationZ(rad/s)']),
        gyro_yaw_diff_radps = np.array(data['gyroRotationZ_diff(rad/s)']),
        gyro_pitch_corrected_radps = np.array(data['gyroRotationX_corrected(rad/s)']),
        calc_speed_ms = np.array(data['calc_speed_ms'])
    )

def SNR_analysis(signal_path, control_path, filter_type, scenario):
    '''
    Process signal.
    Process control.
    Send to vis.
    Plot: 1. time-series with SNR, histograms with std-dev values, FFT plots with relative RMS
    Need: 1. start/end index in from-app function 2. plotting here
    '''

    signal = from_sensor_log_iOS_app_unbiased(path=signal_path, filter_type=filter_type,
            smoothing_window_size_ms=0, start_index=2400, end_index=4200)
    control = from_sensor_log_iOS_app_unbiased(path=control_path, filter_type=filter_type,
            smoothing_window_size_ms=0, start_index=4200, end_index=6400)

    vis.SNR_analysis(signal=signal, control=control, scenario=scenario)

    return

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
        columns=COLUMNS_GLOBAL)
    data = data.dropna(how='any')
    data = data.reset_index(drop=True)

    print(data_in['gyroRotationY(rad/s)'])

    return data