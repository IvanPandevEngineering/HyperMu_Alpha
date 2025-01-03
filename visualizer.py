import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def fft_convert(series):
    'return frequencies, normalized magnitudes of a time series from the shaker. Currently fixed at 100hz'

    #  Remove mean from series to prevent DC component 
    series = series - np.mean(series)

    #  Perform FFT transform
    freqs = np.fft.fftfreq(len(series), 1 / 100)
    mags = np.abs(np.fft.fft(series))

    #  Normalize to preserve comparability to time-domain values
    #  Normalization by sqrt(len(N)) preserves amplitude, but not energy
    mags_norm = mags / np.sqrt(len(series))

    #  Apply fftshift to remove horizontal line when plotting
    return  np.fft.fftshift(freqs), np.fft.fftshift(mags_norm)

def get_fft_energy(series):
    'return normalized FFT energy of a time series from the shaker. Energy is a metric describing load variation.'

    mags = np.fft.fft(series)

    return np.sum(np.abs(mags)**2) / len(series)

def plot_basics(force_function, results, scenario):

    print('Graphing...')

    plt.style.use('seaborn-v0_8')
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    fig, subplots = plt.subplots(4, 2, figsize=(14, 9))
    fig.suptitle(f'General Performance Overview, {scenario}', fontsize=14)
    fig.text(0.005, 0.005, 'SAFETY DISCLAIMER: This software is intended strictly as a technical showcase for public viewing and commentary, NOT for public use, editing, or adoption. The simulation code within has not been fully validated for accuracy or real-world application. Do NOT apply any changes to real-world vehicles based on HyperMu simulation results. Modifying vehicle properties always carries a risk of deadly loss of vehicle control. Any attempt to use this software for real-world applications is highly discouraged and done at the user’s own risk. The author assumes no liability for any consequences arising from such misuse. All rights reserved, Copyright 2024 Ivan Pandev.', fontsize=8)

    subplots[0,0].plot(force_function['loggingTime(txt)'], force_function['accelerometerAccelerationX(G)'], label='lateral accel (G)')
    subplots[0,0].plot(force_function['loggingTime(txt)'], -force_function['accelerometerAccelerationY(G)'], label='longitudinal accel (G)')
    subplots[0,0].plot(force_function['loggingTime(txt)'], -force_function['accelerometerAccelerationZ(G)'], label='vertical accel (G)')
    subplots[0,0].plot(force_function['loggingTime(txt)'], force_function['c_fr']*-100, label='road surface height (cm, fr)')
    subplots[0,0].plot(force_function['loggingTime(txt)'], force_function['c_rr']*-100, label='road surface height (cm, rr)')
    subplots[0,0].set_ylabel('Function inputs (G, cm)')
    subplots[0,0].legend()
    subplots[0,0].grid(True)

    subplots[0,1].plot(force_function['loggingTime(txt)'], results['roll_angle_f'], label='roll angle front (deg)')
    subplots[0,1].plot(force_function['loggingTime(txt)'], results['roll_angle_r'], label='roll angle rear (deg)')
    subplots[0,1].plot(force_function['loggingTime(txt)'], results['pitch_angle'], label='pitch angle (deg)')
    subplots[0,1].set_ylabel('Chassis Attitude (deg)')
    subplots[0,1].legend()
    subplots[0,1].grid(True)

    subplots[1,0].plot(force_function['loggingTime(txt)'], results['spring_disp_fr'], label='spring displacement (m, fr)')
    subplots[1,0].plot(force_function['loggingTime(txt)'], results['spring_disp_rr'], label='spring displacement (m, rr)')
    subplots[1,0].plot(force_function['loggingTime(txt)'], results['damper_disp_rr'], label='damper displacement (m, rr)')
    subplots[1,0].set_ylabel('Suspesion Displacements (m)')
    subplots[1,0].legend()
    subplots[1,0].grid(True)

    subplots[1,1].plot(force_function['loggingTime(txt)'], results['bump_stop_F_fr'], label='bump stop force (N, fr)')
    subplots[1,1].plot(force_function['loggingTime(txt)'], results['bump_stop_F_fl'], label='bump stop force (N, fl)')
    subplots[1,1].plot(force_function['loggingTime(txt)'], results['bump_stop_F_rr'], label='bump stop force (N, rr)')
    subplots[1,1].plot(force_function['loggingTime(txt)'], results['bump_stop_F_rl'], label='bump stop force (N, rl)')
    subplots[1,1].set_ylabel('Deconstructed Spring Forces (N)')
    subplots[1,1].legend()
    subplots[1,1].grid(True)

    subplots[2,0].plot(force_function['loggingTime(txt)'], results['damper_vel_fr'], label='damper speed (m/s, fr)')
    subplots[2,0].plot(force_function['loggingTime(txt)'], results['damper_vel_fl'], label='damper speed (m/s, fl)')
    subplots[2,0].plot(force_function['loggingTime(txt)'], results['damper_vel_rr'], label='damper speed (m/s, rr)')
    subplots[2,0].plot(force_function['loggingTime(txt)'], results['damper_vel_rl'], label='damper speed (m/s, rl)')
    subplots[2,0].set_ylabel('Damper Speed (m/s)')
    subplots[2,0].legend()
    subplots[2,0].grid(True)

    subplots[2,1].plot(force_function['loggingTime(txt)'], results['tire_load_fr'], label='tire load (N, fr)')
    subplots[2,1].plot(force_function['loggingTime(txt)'], results['tire_load_fl'], label='tire load (N, fl)')
    subplots[2,1].plot(force_function['loggingTime(txt)'], results['tire_load_rr'], label='tire load (N, rr)')
    subplots[2,1].plot(force_function['loggingTime(txt)'], results['tire_load_rl'], label='tire load (N, rl)')
    #subplots[2,1].plot(force_function['loggingTime(txt)'], (results['tire_load_fr']+results['tire_load_fl']+results['tire_load_rr']+results['tire_load_rl'])/4, label='tire load (N, avg)')
    subplots[2,1].set_ylabel('Tire Load (N)')
    subplots[2,1].legend()
    subplots[2,1].grid(True)

    subplots[3,0].plot(force_function['loggingTime(txt)'], results['lateral_load_dist_f'], label='lateral load dist (%, f)')
    subplots[3,0].plot(force_function['loggingTime(txt)'], results['lateral_load_dist_r'], label='lateral load dist (%, r)')
    subplots[3,0].set_ylabel('Lateral Load Distribution (%)')
    subplots[3,0].legend()
    subplots[3,0].grid(True)

    subplots[3,1].plot(force_function['loggingTime(txt)'], results['lateral_load_dist_ratio'], label='lateral load dist ratio (%, f)')
    subplots[3,1].set_ylabel('Lat. Load Dist. Ratio (%)')
    subplots[3,1].legend()
    subplots[3,1].grid(True)

    fig.tight_layout()
    plt.show()

    return

def check_correlation_rollPitchRate(force_function, results, scenario):

    print('Graphing...')
    plt.style.use('seaborn-v0_8')
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    fig, subplots = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'Correlation on Roll/Pitch Rate, {scenario}', fontsize=14)
    fig.text(0.005, 0.005, 'SAFETY DISCLAIMER: This software is intended strictly as a technical showcase for public viewing and commentary, NOT for public use, editing, or adoption. The simulation code within has not been fully validated for accuracy or real-world application. Do NOT apply any changes to real-world vehicles based on HyperMu simulation results. Modifying vehicle properties always carries a risk of deadly loss of vehicle control. Any attempt to use this software for real-world applications is highly discouraged and done at the user’s own risk. The author assumes no liability for any consequences arising from such misuse. All rights reserved, Copyright 2024 Ivan Pandev.', fontsize=8)

    subplots[0][0].plot(force_function['loggingTime(txt)'], (180*force_function['gyroRotationY(rad/s)']/3.14), label='Recorded roll angle rate (deg/s)')
    subplots[0][0].plot(force_function['loggingTime(txt)'], -force_function['accelerometerAccelerationZ(G)'], label='vertical accel (G)')
    subplots[0][0].plot(force_function['loggingTime(txt)'], results['roll_angle_rate_f'], label='predicted roll angle rate (deg/s, f)')
    subplots[0][0].plot(force_function['loggingTime(txt)'], results['roll_angle_rate_r'], label='predicted roll angle rate (deg/s, r)')
    subplots[0][0].set_xlabel('Time')
    subplots[0][0].set_ylabel('Roll Rate (deg/s)')
    subplots[0][0].legend()
    subplots[0][0].grid(True)

    roll_angle_rate_avg = (np.array(results['roll_angle_rate_f']) + np.array(results['roll_angle_rate_r'])) / 2
    slope, intercept, r_value, p_value, std_err = stats.linregress(roll_angle_rate_avg, (180*force_function['gyroRotationY(rad/s)']/3.14))
    r_squared = r_value ** 2

    subplots[0][1].scatter(roll_angle_rate_avg, (180*force_function['gyroRotationY(rad/s)']/3.14), label='(deg/s)', s=10)
    subplots[0][1].plot(np.linspace(-10, 10, 3), slope*np.linspace(-10, 10, 3)+intercept, color='orange', label=f'Linear fit, R-sq: {r_squared:.3f}, Slope: {slope:.3f}')
    subplots[0][1].plot([-10,10], [-10,10], color='green', label='unity')
    subplots[0][1].legend()
    subplots[0][1].set_xlabel('Predicted Roll Rate (deg/s)')
    subplots[0][1].set_ylabel('Recorded Roll Rate (deg/s)')
    subplots[0][1].grid(True)

    subplots[1][0].plot(force_function['loggingTime(txt)'], (180*force_function['gyroRotationX_corrected(rad/s)']/3.14), label='Corrected pitch angle rate (deg/s)')
    #subplots[1][0].plot(force_function['loggingTime(txt)'], (180*force_function['gyroRotationZ(rad/s)']/3.14)+.2, label='Yaw angle rate (deg/s)')
    #subplots[1][0].plot(force_function['loggingTime(txt)'], (180*force_function['gyroRotationX(rad/s)']/3.14)+.2, label='Recorded raw pitch angle rate (deg/s)')
    subplots[1][0].plot(force_function['loggingTime(txt)'], -np.array(results['pitch_angle_rate']), label='predicted pitch angle rate (deg/s, f)')
    subplots[1][0].set_xlabel('Time')
    subplots[1][0].set_ylabel('Pitch Rate (deg/s)')
    subplots[1][0].legend()
    subplots[1][0].grid(True)

    slope_p, intercept_p, r_value_p, p_value, std_err = stats.linregress(-np.array(results['pitch_angle_rate']), (180*force_function['gyroRotationX_corrected(rad/s)']/3.14))
    r_squared_p = r_value_p ** 2

    subplots[1][1].scatter(-np.array(results['pitch_angle_rate']), (180*force_function['gyroRotationX_corrected(rad/s)']/3.14), label='(deg/s)', s=10)
    subplots[1][1].plot(np.linspace(-10, 10, 3), slope_p*np.linspace(-10, 10, 3)+intercept_p, color='orange', label=f'Linear fit, R-sq: {r_squared_p:.3f}, Slope: {slope_p:.3f}')
    subplots[1][1].plot([-10,10], [-10,10], color='green', label='unity')
    subplots[1][1].legend()
    subplots[1][1].set_xlabel('Predicted Pitch Rate (deg/s)')
    subplots[1][1].set_ylabel('Recorded Pitch Rate (deg/s)')
    subplots[1][1].grid(True)

    fig.tight_layout()
    plt.show()

def check_correlation_rollRateRearZ(force_function,
a_dd_rear_axle,
tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl,
damper_vel_fr, damper_vel_fl, damper_vel_rr, damper_vel_rl,
damper_force_fr, damper_force_fl, damper_force_rr, damper_force_rl,
lateral_load_dist_f, lateral_load_dist_r,
roll_angle_f, roll_angle_r, pitch_angle,
roll_angle_rate_f, roll_angle_rate_r, pitch_angle_rate):

    print('Graphing...')

    plt.style.use('seaborn-v0_8')
    fig, subplots = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Correlation of Race Telemetry on Battle_Bimmer_30_Sept_2023_w_Pass (Left-Smoothing Window = 750ms)', fontsize=14)
    fig.text(0.005, 0.005, 'SAFETY DISCLAIMER: This software is intended strictly as a technical showcase for public viewing and commentary, NOT for public use, editing, or adoption. The simulation code within has not been fully validated for accuracy or real-world application. Do NOT apply any changes to real-world vehicles based on HyperMu simulation results. Modifying vehicle properties always carries a risk of deadly loss of vehicle control. Any attempt to use this software for real-world applications is highly discouraged and done at the user’s own risk. The author assumes no liability for any consequences arising from such misuse. All rights reserved, Copyright 2024 Ivan Pandev.', fontsize=8)

    subplots[0][0].plot(force_function['loggingTime(txt)'], (180*force_function['gyroRotationY(rad/s)']/3.14)-.2, label='Recorded roll angle rate (deg/s)')
    subplots[0][0].plot(force_function['loggingTime(txt)'], roll_angle_rate_f, label='predicted roll angle rate (deg/s, f)')
    subplots[0][0].plot(force_function['loggingTime(txt)'], roll_angle_rate_r, label='predicted roll angle rate (deg/s, r)')
    subplots[0][0].set_xlabel('Time')
    subplots[0][0].set_ylabel('Roll Rate (deg/s)')
    subplots[0][0].legend()
    subplots[0][0].grid(True)

    roll_angle_rate_avg = (np.array(roll_angle_rate_f) + np.array(roll_angle_rate_r)) / 2
    slope, intercept, r_value, p_value, std_err = stats.linregress(roll_angle_rate_avg, (180*force_function['gyroRotationY(rad/s)']/3.14)-.2)
    r_squared = r_value ** 2

    subplots[0][1].scatter(roll_angle_rate_avg, (180*force_function['gyroRotationY(rad/s)']/3.14)-.2, label='(deg/s)', s=10)
    subplots[0][1].plot(np.linspace(-10, 10, 3), slope*np.linspace(-10, 10, 3)+intercept, color='orange', label=f'Linear fit, R-sq: {r_squared:.3f}, Slope: {slope:.3f}')
    subplots[0][1].plot([-10,10], [-10,10], color='green', label='unity')
    subplots[0][1].legend()
    subplots[0][1].set_xlabel('Predicted Roll Rate (deg/s)')
    subplots[0][1].set_ylabel('Recorded Roll Rate (deg/s)')
    subplots[0][1].grid(True)

    subplots[1][0].plot(force_function['loggingTime(txt)'], force_function['accelerometerAccelerationZ(G)'], label='Recorded Vertical Accel, Rear Axle')
    subplots[1][0].plot(force_function['loggingTime(txt)'], a_dd_rear_axle, label='Predicted Vertical Accel, Rear Axle')
    subplots[1][0].set_xlabel('Time')
    subplots[1][0].set_ylabel('Vertical Acceleration')
    subplots[1][0].legend()
    subplots[1][0].grid(True)

    slope_p, intercept_p, r_value_p, p_value, std_err = stats.linregress(a_dd_rear_axle, force_function['accelerometerAccelerationZ(G)'])
    r_squared_p = r_value_p ** 2

    subplots[1][1].scatter(a_dd_rear_axle, force_function['accelerometerAccelerationZ(G)'], label='(deg/s)', s=10)
    subplots[1][1].plot(np.linspace(-10, 10, 3), slope_p*np.linspace(-10, 10, 3)+intercept_p, color='orange', label=f'Linear fit, R-sq: {r_squared_p:.3f}, Slope: {slope_p:.3f}')
    subplots[1][1].plot([-10,10], [-10,10], color='green', label='unity')
    subplots[1][1].legend()
    subplots[1][1].set_xlabel('Predicted Vertical Accel, Rear Axle')
    subplots[1][1].set_ylabel('Recorded Vertical Accel, Rear Axle')
    subplots[1][1].grid(True)

    fig.tight_layout()
    plt.show()

def damper_response_detail(force_function,
a_dd_rear_axle,
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

def tire_response_detail_comparison(force_function, self, other, scenario):

    print('Graphing...')

    plt.style.use('ggplot')
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    fig, subplots = plt.subplots(2, 4, figsize=(14, 8))
    fig.suptitle(f'Tire Response Detail Comparison, {scenario}', fontsize=14)
    fig.text(0.005, 0.005, 'SAFETY DISCLAIMER: This software is intended strictly as a technical showcase for public viewing and commentary, NOT for public use, editing, or adoption. The simulation code within has not been fully validated for accuracy or real-world application. Do NOT apply any changes to real-world vehicles based on HyperMu simulation results. Modifying vehicle properties always carries a risk of deadly loss of vehicle control. Any attempt to use this software for real-world applications is highly discouraged and done at the user’s own risk. The author assumes no liability for any consequences arising from such misuse. All rights reserved, Copyright 2024 Ivan Pandev.', fontsize=8)

    subplots[0,0].hist(self['tire_load_fl'], bins=50, label='self tire load, fl')
    subplots[0,0].hist(other['tire_load_fl'], bins=50, alpha = 0.8, label='other tire load, fl')
    subplots[0,0].set_ylabel('Tire Load (N)')
    subplots[0,0].set_xlabel(f"self std_dev: {np.std(self['tire_load_fl']):.4}\n other std_dev: {np.std(other['tire_load_fl']):.4}")
    subplots[0,0].legend()
    subplots[0,0].grid(True)
    subplots[0,0].set_yscale('log')

    subplots[0,1].hist(self['tire_load_fr'], bins=50, label='self tire load, fr')
    subplots[0,1].hist(other['tire_load_fr'], bins=50, alpha = 0.7, label='other tire load, fr')
    subplots[0,1].set_ylabel('Tire Load (N)')
    subplots[0,1].set_xlabel(f"self std_dev: {np.std(self['tire_load_fr']):.4}\n other std_dev: {np.std(other['tire_load_fr']):.4}")
    subplots[0,1].legend()
    subplots[0,1].grid(True)
    subplots[0,1].set_yscale('log')

    subplots[1,0].hist(self['tire_load_rl'], bins=50, label='self tire load, fr')
    subplots[1,0].hist(other['tire_load_rl'], bins=50, alpha = 0.7, label='other tire load, fr')
    subplots[1,0].set_ylabel('Tire Load (N)')
    subplots[1,0].set_xlabel(f"self std_dev: {np.std(self['tire_load_rl']):.4}\n other std_dev: {np.std(other['tire_load_rl']):.4}")
    subplots[1,0].legend()
    subplots[1,0].grid(True)
    subplots[1,0].set_yscale('log')

    subplots[1,1].hist(self['tire_load_rr'], bins=50, label='self tire load, rr')
    subplots[1,1].hist(other['tire_load_rr'], bins=50, alpha = 0.7, label='other tire load, rr')
    subplots[1,1].set_ylabel('Tire Load (N)')
    subplots[1,1].set_xlabel(f"self std_dev: {np.std(self['tire_load_rr']):.4}\n other std_dev: {np.std(other['tire_load_rr']):.4}")
    subplots[1,1].legend()
    subplots[1,1].grid(True)
    subplots[1,1].set_yscale('log')

    subplots[0,2].plot(fft_convert(self['tire_load_fl'])[0], fft_convert(self['tire_load_fl'])[1], label='self_tire_load_fl')
    subplots[0,2].plot(fft_convert(other['tire_load_fl'])[0], fft_convert(other['tire_load_fl'])[1], alpha = 0.7, label='other_tire_load_fl')
    subplots[0,2].set_ylabel('Normalized Amplitude (N)')
    subplots[0,2].set_xlabel(f"self total energy: {get_fft_energy(self['tire_load_fl']):.3}\n other total energy: {get_fft_energy(other['tire_load_fl']):.3}")
    subplots[0,2].legend()
    subplots[0,2].grid(True)
    subplots[0,2].set_xscale('log')
    #subplots[0,2].set_yscale('log')

    subplots[0,3].plot(fft_convert(self['tire_load_fr'])[0], fft_convert(self['tire_load_fr'])[1], label='self_tire_load_fr')
    subplots[0,3].plot(fft_convert(other['tire_load_fr'])[0], fft_convert(other['tire_load_fr'])[1], alpha = 0.7, label='other_tire_load_fr')
    subplots[0,3].set_ylabel('Normalized Amplitude (N)')
    subplots[0,3].set_xlabel(f"self total energy: {get_fft_energy(self['tire_load_fr']):.3}\n other total energy: {get_fft_energy(other['tire_load_fr']):.3}")
    subplots[0,3].legend()
    subplots[0,3].grid(True)
    subplots[0,3].set_xscale('log')
    #subplots[0,3].set_yscale('log')

    subplots[1,2].plot(fft_convert(self['tire_load_rl'])[0], fft_convert(self['tire_load_rl'])[1], label='self_tire_load_rl')
    subplots[1,2].plot(fft_convert(other['tire_load_rl'])[0], fft_convert(other['tire_load_rl'])[1], alpha = 0.7, label='other_tire_load_rl')
    subplots[1,2].set_ylabel('Normalized Amplitude (N)')
    subplots[1,2].set_xlabel(f"self total energy: {get_fft_energy(self['tire_load_rl']):.3}\n other total energy: {get_fft_energy(other['tire_load_rl']):.3}")
    subplots[1,2].legend()
    subplots[1,2].grid(True)
    subplots[1,2].set_xscale('log')
    #subplots[1,2].set_yscale('log')

    subplots[1,3].plot(fft_convert(self['tire_load_rr'])[0], fft_convert(self['tire_load_rr'])[1], label='self_tire_load_rr')
    subplots[1,3].plot(fft_convert(other['tire_load_rr'])[0], fft_convert(other['tire_load_rr'])[1], alpha = 0.7, label='other_tire_load_rr')
    subplots[1,3].set_ylabel('Normalized Amplitude (N)')
    subplots[1,3].set_xlabel(f"self total energy: {get_fft_energy(self['tire_load_rr']):.3}\n other total energy: {get_fft_energy(other['tire_load_rr']):.3}")
    subplots[1,3].legend()
    subplots[1,3].grid(True)
    subplots[1,3].set_xscale('log')
    #subplots[1,3].set_yscale('log')

    fig.tight_layout()
    plt.show()

    return

def ML_set(synth_data):

    print('Graphing...')

    plt.style.use('seaborn-v0_8')
    fig, subplots = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Mixed Real, Synthetic ML Training Set', fontsize=14)

    for i, sample in enumerate(synth_data[1:]):
        if i == 0:
            subplots[1].plot(sample[0][1], label=f'Telemetry Response, CM Height: {sample[1][0]:.3f} m')
        else:
            subplots[1].plot(sample[0][1], label=f'Simulated Response, CM Height: {sample[1][0]:.3f} m')
        #subplots[1].plot(sample[0][3])
        subplots[1].legend(fontsize = '9', loc = 'upper right')
        subplots[1].set_xlabel('Time (s/100)')
        subplots[1].set_ylabel('Roll Angle Rate, (deg/s)')
        subplots[1].set_title('Response at Various CM Height Values')

    subplots[0].plot(sample[0][0], label=f'Lateral Input')
    #subplots[0].plot(sample[0][1], label=f'Longitudinal Input')
    subplots[0].legend(fontsize = '9')
    subplots[0].set_xlabel('Time (s/100)')
    subplots[0].set_ylabel('Acceleration Inputs, G')
    subplots[0].set_title('Acceleration Inputs, G')

    fig.tight_layout()
    plt.show()