import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_response(force_function,
tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl,
damper_vel_fr, damper_vel_fl, damper_vel_rr, damper_vel_rl,
damper_force_fr, damper_force_fl, damper_force_rr, damper_force_rl,
lateral_load_dist_f, lateral_load_dist_r,
roll_angle_f, roll_angle_r, pitch_angle,
roll_angle_rate_f, roll_angle_rate_r, pitch_angle_rate):

    print('Graphing...')

    plt.style.use('seaborn-v0_8')
    fig, subplots = plt.subplots(3, 2, figsize=(14, 8))
    fig.suptitle('Race Telemetry on Battle_Bimmer_28_Dec_2022', fontsize=14)

    subplots[0,0].plot(force_function['loggingTime(txt)'], force_function['accelerometerAccelerationX(G)'], label='lateral accel (G)')
    subplots[0,0].plot(force_function['loggingTime(txt)'], -force_function['accelerometerAccelerationY(G)'], label='longitudinal accel (G)')
    subplots[0,0].plot(force_function['loggingTime(txt)'], force_function['c_fr']*-100, label='surface height (cm, fr)')
    subplots[0,0].plot(force_function['loggingTime(txt)'], force_function['c_rr']*-100, label='surface height (cm, rr)')
    subplots[0,0].set_ylabel('Input acceleration (G), and surface height (cm)')
    subplots[0,0].legend()
    subplots[0,0].grid(True)

    subplots[0,1].plot(force_function['loggingTime(txt)'], roll_angle_f, label='roll angle front (deg)')
    subplots[0,1].plot(force_function['loggingTime(txt)'], roll_angle_r, label='roll angle rear (deg)')
    subplots[0,1].plot(force_function['loggingTime(txt)'], pitch_angle, label='pitch angle (deg)')
    subplots[0,1].legend()
    subplots[0,1].grid(True)

    subplots[1,0].plot(force_function['loggingTime(txt)'], damper_vel_fr, label='damper speed (m/s, fr)')
    subplots[1,0].plot(force_function['loggingTime(txt)'], damper_vel_fl, label='damper speed (m/s, fl)')
    subplots[1,0].plot(force_function['loggingTime(txt)'], damper_vel_rr, label='damper speed (m/s, rr)')
    subplots[1,0].plot(force_function['loggingTime(txt)'], damper_vel_rl, label='damper speed (m/s, rl)')
    subplots[1,0].legend()
    subplots[1,0].grid(True)

    subplots[1,1].plot(force_function['loggingTime(txt)'], tire_load_fr, label='tire load (N, fr)')
    subplots[1,1].plot(force_function['loggingTime(txt)'], tire_load_fl, label='tire load (N, fl)')
    subplots[1,1].plot(force_function['loggingTime(txt)'], tire_load_rr, label='tire load (N, rr)')
    subplots[1,1].plot(force_function['loggingTime(txt)'], tire_load_rl, label='tire load (N, rl)')
    subplots[1,1].legend()
    subplots[1,1].grid(True)

    subplots[2,0].plot(force_function['loggingTime(txt)'], lateral_load_dist_f, label='lateral load dist (%, f)')
    subplots[2,0].plot(force_function['loggingTime(txt)'], lateral_load_dist_r, label='lateral load dist (%, r)')
    subplots[2,0].legend()
    subplots[2,0].grid(True)

    subplots[2,1].plot(force_function['loggingTime(txt)'], lateral_load_dist_f/(lateral_load_dist_r+lateral_load_dist_f), label='lateral load dist ratio (%, f)')
    subplots[2,1].legend()
    subplots[2,1].grid(True)

    fig.tight_layout()
    plt.show()

def check_correlation(force_function,
tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl,
damper_vel_fr, damper_vel_fl, damper_vel_rr, damper_vel_rl,
damper_force_fr, damper_force_fl, damper_force_rr, damper_force_rl,
lateral_load_dist_f, lateral_load_dist_r,
roll_angle_f, roll_angle_r, pitch_angle,
roll_angle_rate_f, roll_angle_rate_r, pitch_angle_rate):

    print('Graphing...')

    plt.style.use('seaborn-v0_8')
    fig, subplots = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Correlation of Race Telemetry on Battle_Bimmer_28_Dec_2022 (Left-Smoothing Window = 750ms)', fontsize=14)

    subplots[0].plot(force_function['loggingTime(txt)'], (180*force_function['gyroRotationY(rad/s)']/3.14)-.3, label='Recorded roll angle rate (deg/s)')
    subplots[0].plot(force_function['loggingTime(txt)'], roll_angle_rate_f, label='predicted roll angle rate (deg/s, f)')
    subplots[0].plot(force_function['loggingTime(txt)'], roll_angle_rate_r, label='predicted roll angle rate (deg/s, r)')
    subplots[0].set_xlabel('Time')
    subplots[0].set_ylabel('Roll Rate (deg/s)')
    subplots[0].legend()
    subplots[0].grid(True)

    roll_angle_rate_avg = (np.array(roll_angle_rate_f) + np.array(roll_angle_rate_r)) / 2

    slope, intercept, r_value, p_value, std_err = stats.linregress(roll_angle_rate_avg, (180*force_function['gyroRotationY(rad/s)']/3.14)-.3)
    r_squared = r_value ** 2

    subplots[1].scatter(roll_angle_rate_avg, (180*force_function['gyroRotationY(rad/s)']/3.14)-.3, label='(deg/s)', s=10)
    subplots[1].plot(np.linspace(-10, 10, 3), slope*np.linspace(-10, 10, 3)+intercept, color='orange', label=f'Linear fit, R-sq: {r_squared:.3f}, Slope: {slope:.3f}')
    subplots[1].plot([-10,10], [-10,10], color='green', label='unity')
    subplots[1].legend()
    subplots[1].set_xlabel('Predicted Roll Rate (deg/s)')
    subplots[1].set_ylabel('Recorded Roll Rate (deg/s)')
    subplots[1].grid(True)

    fig.tight_layout()
    plt.show()

def damper_response(force_function,
tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl,
damper_vel_fr, damper_vel_fl, damper_vel_rr, damper_vel_rl,
damper_force_fr, damper_force_fl, damper_force_rr, damper_force_rl,
lateral_load_dist_f, lateral_load_dist_r,
roll_angle_f, roll_angle_r, pitch_angle,
roll_angle_rate_f, roll_angle_rate_r, pitch_angle_rate):

    print('Graphing...')

    plt.style.use('seaborn-v0_8')
    fig, subplots = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle('Damper Response on Battle_Bimmer_28_Dec_2022', fontsize=14)

    subplots.scatter(damper_vel_fr, damper_force_fr, label='fr')
    subplots.scatter(damper_vel_fl, damper_force_fl, label='fl')
    subplots.scatter(damper_vel_rr, damper_force_rr, label='rr')
    subplots.scatter(damper_vel_rl, damper_force_rl, label='rl')
    subplots.legend()
    subplots.grid(True)

    fig.tight_layout()
    plt.show()

def ML_set(synth_data):

    print('Graphing...')

    plt.style.use('seaborn-v0_8')
    fig, subplots = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle('Synthesized ML Training Set', fontsize=14)

    for sample in synth_data[1:]:
        subplots[1].plot(sample[0][2], label=f'CM Height: {sample[1][0]:.3f} m')
        subplots[1].plot(sample[0][3])
        subplots[1].legend(fontsize = '8', loc = 'upper right')
        subplots[1].set_xlabel('Time (1/100s)')
        subplots[1].set_ylabel('Simulated Roll Angle, Degrees')
        subplots[1].set_title('Simulated Response to Inputs at Various CM Height Values')

    subplots[0].plot(sample[0][0], label=f'Lateral Input')
    subplots[0].plot(sample[0][1], label=f'Longitudinal Input')
    subplots[0].legend(fontsize = '8')
    subplots[0].set_xlabel('Time (1/100s)')
    subplots[0].set_ylabel('Synthetic Acceleration Inputs, G')
    subplots[0].set_title('Synthetic Acceleration Inputs, G')

    fig.tight_layout()
    plt.show()