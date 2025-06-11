'''
Copyright 2025 Ivan Pandev
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import formulas as f


DISCLAIMER = str("SAFETY DISCLAIMER: This software—and any video or written post showcasing its development or capabilities—is intended strictly as a technical demonstration for public viewing and commentary, NOT for public use, modification, or integration into real-world systems. The simulation code within has NOT been fully validated for accuracy or real-world application. Do NOT apply any changes to real-world vehicles based on HyperMu simulation results. Modifying vehicle properties always carries a risk of deadly loss of vehicle control. Any attempt to use this software for real-world applications is highly discouraged and done at the user’s own risk. The author assumes no liability for any consequences arising from such misuse. All rights reserved, Copyright 2025 Ivan Pandev.")

def plot_basics(force_function, results, scenario, desc):

    print('Graphing...\n')

    plt.style.use('seaborn-v0_8')
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    fig, subplots = plt.subplots(4, 2, figsize=(14, 9))
    fig.suptitle(f'General Performance Overview, {scenario}: {desc}', fontsize=14)
    fig.text(0.005, 0.005, f'{DISCLAIMER}', fontsize=8)

    subplots[0,0].plot(force_function.time, force_function.G_lat, label='lateral accel (G)')
    subplots[0,0].plot(force_function.time, force_function.G_long, label='longitudinal accel (G)')
    subplots[0,0].plot(force_function.time, -force_function.calc_speed_ms*(2.23694/100), label='estimated speed (mph/100)')
    #subplots[0,0].plot(force_function.time, -force_function['accelerometerAccelerationZ(G)'], label='vertical accel (G)')
    #subplots[0,0].plot(force_function.time, force_function['motionPitch(rad)']*56, label='pitch (deg)')
    subplots[0,0].plot(force_function.time, force_function.c_fr*-100, label='road surface height (cm, fr)')
    subplots[0,0].plot(force_function.time, force_function.c_fl*-100, label='road surface height (cm, fl)')
    subplots[0,0].plot(force_function.time, force_function.c_rr*-100, label='road surface height (cm, rr)')
    subplots[0,0].plot(force_function.time, force_function.c_rl*-100, label='road surface height (cm, rl)')
    subplots[0,0].set_ylabel('Function inputs (G, cm)')
    subplots[0,0].legend()
    subplots[0,0].grid(True)

    subplots[0,1].plot(force_function.time, results['roll_angle_f'], label='roll angle front (deg)')
    subplots[0,1].plot(force_function.time, results['roll_angle_r'], label='roll angle rear (deg)')
    subplots[0,1].plot(force_function.time, results['pitch_angle'], label='pitch angle (deg)')
    #subplots[0,1].plot(force_function.time, results['a_fr'], label='a_fr (m)')
    #subplots[0,1].plot(force_function.time, results['a_fl'], label='a_fl (m)')
    #subplots[0,1].plot(force_function.time, results['a_rr'], label='a_rr (m)')
    #subplots[0,1].plot(force_function.time, results['a_rl'], label='a_rl (m)')
    #subplots[0,1].plot(force_function.time, results['b_fr'], label='b_fr (m)')
    #subplots[0,1].plot(force_function.time, results['b_fl'], label='b_fl (m)')
    #subplots[0,1].plot(force_function.time, results['b_rr'], label='b_rr (m)')
    #subplots[0,1].plot(force_function.time, results['b_rl'], label='b_rl (m)')
    subplots[0,1].set_ylabel('Chassis Attitude (deg)')
    subplots[0,1].legend()
    subplots[0,1].grid(True)

    subplots[1,0].plot(force_function.time, results['spring_disp_fr'], label='spring displacement (m, fr)')
    subplots[1,0].plot(force_function.time, results['spring_disp_fl'], label='spring displacement (m, fl)')
    subplots[1,0].plot(force_function.time, results['spring_disp_rr'], label='spring displacement (m, rr)')
    subplots[1,0].plot(force_function.time, results['spring_disp_rl'], label='spring displacement (m, rl)')
    subplots[1,0].fill_between(force_function.time, 0, 0.07, where=np.array(results['tlsf_suspension_fr'])>0,
                               color='red', alpha=0.6)
    subplots[1,0].fill_between(force_function.time, 0, 0.07, where=np.array(results['tlsf_suspension_fl'])>0,
                               color='red', alpha=0.6)
    subplots[1,0].fill_between(force_function.time, 0, 0.07, where=np.array(results['tlsf_suspension_rr'])>0,
                               color='red', alpha=0.6)
    subplots[1,0].fill_between(force_function.time, 0, 0.07, where=np.array(results['tlsf_suspension_rl'])>0,
                               color='red', alpha=0.6)
    subplots[1,0].set_ylabel('Suspesion Displacements (m)')
    subplots[1,0].legend()
    subplots[1,0].grid(True)

    subplots[1,1].plot(force_function.time, results['bump_stop_F_fr'], label='bump stop force (N, fr)')
    subplots[1,1].plot(force_function.time, results['bump_stop_F_fl'], label='bump stop force (N, fl)')
    subplots[1,1].plot(force_function.time, results['bump_stop_F_rr'], label='bump stop force (N, rr)')
    subplots[1,1].plot(force_function.time, results['bump_stop_F_rl'], label='bump stop force (N, rl)')
    subplots[1,1].set_ylabel('Deconstructed Spring Forces (N)')
    subplots[1,1].legend()
    subplots[1,1].grid(True)

    subplots[2,0].plot(force_function.time, results['damper_vel_fr'], label='damper speed (m/s, fr)')
    subplots[2,0].plot(force_function.time, results['damper_vel_fl'], label='damper speed (m/s, fl)')
    subplots[2,0].plot(force_function.time, results['damper_vel_rr'], label='damper speed (m/s, rr)')
    subplots[2,0].plot(force_function.time, results['damper_vel_rl'], label='damper speed (m/s, rl)')
    subplots[2,0].set_ylabel('Damper Speed (m/s)')
    subplots[2,0].legend()
    subplots[2,0].grid(True)

    subplots[2,1].plot(force_function.time, results['tire_load_fr'], label='tire load (N, fr)')
    subplots[2,1].plot(force_function.time, results['tire_load_fl'], label='tire load (N, fl)')
    subplots[2,1].plot(force_function.time, results['tire_load_rr'], label='tire load (N, rr)')
    subplots[2,1].plot(force_function.time, results['tire_load_rl'], label='tire load (N, rl)')
    #subplots[2,1].plot(force_function.time, (results['tire_load_fr']+results['tire_load_fl']+results['tire_load_rr']+results['tire_load_rl'])/4, label='tire load (N, avg)')
    subplots[2,1].set_ylabel('Tire Load (N)')
    subplots[2,1].legend()
    subplots[2,1].grid(True)

    subplots[3,0].plot(force_function.time, results['lateral_load_dist_f'], label='lateral load dist (%, f)')
    subplots[3,0].plot(force_function.time, results['lateral_load_dist_r'], label='lateral load dist (%, r)')
    subplots[3,0].set_ylabel('Lateral Load Distribution (%)')
    subplots[3,0].legend()
    subplots[3,0].grid(True)

    subplots[3,1].plot(force_function.time, results['lateral_load_dist_ratio'], label='lateral load dist ratio (%, f)')
    subplots[3,1].set_ylabel('Lat. Load Dist. Ratio (%)')
    subplots[3,1].legend()
    subplots[3,1].grid(True)

    fig.tight_layout()
    plt.show()

    return

def check_correlation_rollPitchRate(force_function, results, scenario):

    print('Graphing...\n')
    plt.style.use('seaborn-v0_8')
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    fig, subplots = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Correlation on Roll/Pitch Rate, {scenario}', fontsize=14)
    fig.text(0.005, 0.005, f'{DISCLAIMER}', fontsize=8)

    subplots[0][0].plot(force_function.time, (180*force_function.gyro_roll_radps/3.14), label='Recorded roll angle rate (deg/s)')
    #subplots[0][0].plot(force_function.time, -force_function['accelerometerAccelerationZ(G)'], label='vertical accel (G)')
    subplots[0][0].plot(force_function.time, results['roll_angle_rate_f'], label='predicted roll angle rate (deg/s, f)')
    subplots[0][0].plot(force_function.time, results['roll_angle_rate_r'], label='predicted roll angle rate (deg/s, r)')
    subplots[0][0].set_xlabel('Time')
    subplots[0][0].set_ylabel('Roll Rate (deg/s)')
    subplots[0][0].legend()
    subplots[0][0].grid(True)

    roll_angle_rate_avg = (np.array(results['roll_angle_rate_f']) + np.array(results['roll_angle_rate_r'])) / 2
    slope, intercept, r_value, p_value, std_err = stats.linregress(roll_angle_rate_avg, (180*force_function.gyro_roll_radps/3.14))
    r_squared = r_value ** 2

    subplots[0][1].scatter(roll_angle_rate_avg, (180*force_function.gyro_roll_radps/3.14), label='(deg/s)', s=10)
    subplots[0][1].plot(np.linspace(-6, 6, 3), slope*np.linspace(-6, 6, 3)+intercept, color='orange', label=f'Linear fit, R-sq: {r_squared:.3f}, Slope: {slope:.3f}')
    subplots[0][1].plot([-6,6], [-6,6], color='green', label='unity')
    subplots[0][1].legend()
    subplots[0][1].set_xlabel('Predicted Roll Rate (deg/s)')
    subplots[0][1].set_ylabel('Recorded Roll Rate (deg/s)')
    subplots[0][1].grid(True)

    subplots[1][0].plot(force_function.time, (180*force_function.gyro_pitch_corrected_radps/3.14), label='Corrected pitch angle rate (deg/s)')
    subplots[1][0].plot(force_function.time, (180*force_function.gyro_pitch_radps/3.14), label='Recorded pitch rate (deg/s)')
    subplots[1][0].plot(force_function.time, (0.01*180*force_function.gyro_yaw_radps/3.14), label='Yaw angle rate (deg/s)')
    #subplots[1][0].plot(force_function.time, force_function.G_long, label='Long Accel (G)')
    subplots[1][0].plot(force_function.time, -np.array(results['pitch_angle_rate']), label='predicted pitch angle rate (deg/s)')
    subplots[1][0].set_xlabel('Time')
    subplots[1][0].set_ylabel('Pitch Rate (deg/s)')
    subplots[1][0].legend()
    subplots[1][0].grid(True)

    slope_p, intercept_p, r_value_p, p_value, std_err = stats.linregress(-np.array(results['pitch_angle_rate']), (180*force_function.gyro_pitch_corrected_radps/3.14))
    r_squared_p = r_value_p ** 2

    subplots[1][1].scatter(-np.array(results['pitch_angle_rate']), (180*force_function.gyro_pitch_corrected_radps/3.14), label='(deg/s)', s=10)
    subplots[1][1].plot(np.linspace(-6, 6, 3), slope_p*np.linspace(-6, 6, 3)+intercept_p, color='orange', label=f'Linear fit, R-sq: {r_squared_p:.3f}, Slope: {slope_p:.3f}')
    subplots[1][1].plot([-6,6], [-6,6], color='green', label='unity')
    subplots[1][1].legend()
    subplots[1][1].set_xlabel('Predicted Pitch Rate (deg/s)')
    subplots[1][1].set_ylabel('Recorded Pitch Rate (deg/s)')
    subplots[1][1].grid(True)

    subplots[0][2].plot(force_function.time[:-1000], f.get_RsqCorr_v_time(
        control = 180*force_function.gyro_roll_radps/3.14,
        results = np.array(results['roll_angle_rate_f']),
        window_s=4)
    )
    subplots[0][2].set_xlabel('Time (s)')
    subplots[0][2].set_ylabel('R-sq Correlation, Roll (%)')
    subplots[0][2].grid(True)

    subplots[1][2].plot(force_function.time[:-1000], f.get_RsqCorr_v_time(
        control = 180*force_function.gyro_pitch_corrected_radps/3.14,
        results = -np.array(results['pitch_angle_rate']),
        window_s=4)
    )
    subplots[1][2].legend()
    subplots[1][2].set_xlabel('Time (s)')
    subplots[1][2].set_ylabel('R-sq Correlation, Pitch (%)')
    subplots[1][2].grid(True)

    fig.tight_layout()
    plt.show()

def check_correlation_rollRateRearZ(force_function, results, scenario):

    print('Graphing...\n')
    plt.style.use('seaborn-v0_8')
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    fig, subplots = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'Correlation on Roll/Pitch Rate, {scenario}', fontsize=14)
    fig.text(0.005, 0.005, f'{DISCLAIMER}', fontsize=8)

    subplots[0][0].plot(force_function.time, (180*force_function.gyro_roll_radps/3.14), label='Recorded roll angle rate (deg/s)')
    subplots[0][0].plot(force_function.time, -force_function['accelerometerAccelerationZ(G)'], label='vertical accel (G)')
    subplots[0][0].plot(force_function.time, results['roll_angle_rate_f'], label='predicted roll angle rate (deg/s, f)')
    subplots[0][0].plot(force_function.time, results['roll_angle_rate_r'], label='predicted roll angle rate (deg/s, r)')
    subplots[0][0].set_xlabel('Time')
    subplots[0][0].set_ylabel('Roll Rate (deg/s)')
    subplots[0][0].legend()
    subplots[0][0].grid(True)

    roll_angle_rate_avg = (np.array(results['roll_angle_rate_f']) + np.array(results['roll_angle_rate_r'])) / 2
    slope, intercept, r_value, p_value, std_err = stats.linregress(roll_angle_rate_avg, (180*force_function.gyro_roll_radps/3.14))
    r_squared = r_value ** 2

    subplots[0][1].scatter(roll_angle_rate_avg, (180*force_function.gyro_roll_radps/3.14), label='(deg/s)', s=10)
    subplots[0][1].plot(np.linspace(-10, 10, 3), slope*np.linspace(-8, 8, 3)+intercept, color='orange', label=f'Linear fit, R-sq: {r_squared:.3f}, Slope: {slope:.3f}')
    subplots[0][1].plot([-10,10], [-10,10], color='green', label='unity')
    subplots[0][1].legend()
    subplots[0][1].set_xlabel('Predicted Roll Rate (deg/s)')
    subplots[0][1].set_ylabel('Recorded Roll Rate (deg/s)')
    subplots[0][1].grid(True)

    subplots[1][0].plot(force_function.time, (force_function['accelerometerAccelerationZ(G)']), label='Corrected pitch angle rate (deg/s)')
    #subplots[1][0].plot(force_function.time, (180*force_function['motionPitch(rad)']/3.14), label='pitch (deg)')
    #subplots[1][0].plot(force_function.time, (180*force_function['gyroRotationZ(rad/s)']/3.14)+.2, label='Yaw angle rate (deg/s)')
    #subplots[1][0].plot(force_function.time, (180*force_function.gyro_pitch_radps/3.14)+.2, label='Recorded raw pitch angle rate (deg/s)')
    subplots[1][0].plot(force_function.time, -np.array(results['a_dd_rear_axle']), label='predicted pitch angle rate (deg/s, f)')
    subplots[1][0].set_xlabel('Time')
    subplots[1][0].set_ylabel('Pitch Rate (deg/s)')
    subplots[1][0].legend()
    subplots[1][0].grid(True)

    slope_p, intercept_p, r_value_p, p_value, std_err = stats.linregress(-np.array(results['a_dd_rear_axle']), (force_function['accelerometerAccelerationZ(G)']))
    r_squared_p = r_value_p ** 2

    subplots[1][1].scatter(-np.array(results['a_dd_rear_axle']), (force_function['accelerometerAccelerationZ(G)']), label='(deg/s)', s=10)
    subplots[1][1].plot(np.linspace(-10, 10, 3), slope_p*np.linspace(-8, 8, 3)+intercept_p, color='orange', label=f'Linear fit, R-sq: {r_squared_p:.3f}, Slope: {slope_p:.3f}')
    subplots[1][1].plot([-10,10], [-10,10], color='green', label='unity')
    subplots[1][1].legend()
    subplots[1][1].set_xlabel('Predicted Pitch Rate (deg/s)')
    subplots[1][1].set_ylabel('Recorded Pitch Rate (deg/s)')
    subplots[1][1].grid(True)

    fig.tight_layout()
    plt.show()

def damper_response_detail(force_function, shaker_results, scenario):

    print('Graphing...\n')

    plt.style.use('seaborn-v0_8')
    fig, subplots = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle('Damper Response on Battle_Bimmer_28_Dec_2022', fontsize=14)

    subplots.plot(shaker_results['damper_vel_fr'], shaker_results['damper_force_fr'], label='fr')
    subplots.plot(shaker_results['damper_vel_fl'], shaker_results['damper_force_fl'], label='fl')
    subplots.plot(shaker_results['damper_vel_rr'], shaker_results['damper_force_rr'], label='rr')
    subplots.plot(shaker_results['damper_vel_rl'], shaker_results['damper_force_rl'], label='rl')
    subplots.legend()
    subplots.grid(True)

    fig.tight_layout()
    plt.show()

def check_correlation_one_wheel_warp(
        force_function,
        recorded_warp_data_dict,
        shaker_results_fr, shaker_results_fl, shaker_results_rr, shaker_results_rl,
        static_errors
    ):
    
    print('Graphing...\n')

    plt.style.use('seaborn-v0_8')
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    fig, subplots = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'One-Wheel Warp Load Distributions, Recorded v. Simulated', fontsize=14)
    fig.text(0.005, 0.005, f'{DISCLAIMER}', fontsize=8)

    #  Plotting FL Loads at different warp conditions.
    subplots[0,0].plot(force_function.time, shaker_results_fr['tire_load_fl'], label='Simulated, FR warp')
    subplots[0,0].plot(force_function.time, shaker_results_fl['tire_load_fl'], label='Simulated, FL warp')
    subplots[0,0].plot(force_function.time, shaker_results_rr['tire_load_fl'], label='Simulated, RR warp')
    subplots[0,0].plot(force_function.time, shaker_results_rl['tire_load_fl'], label='Simulated, RL warp')
    subplots[0,0].axhline(recorded_warp_data_dict['fr_offset_load_fl'], label='Recorded, FR Warp', color='black')
    subplots[0,0].axhline(recorded_warp_data_dict['fl_offset_load_fl'], label='Recorded, FL Warp', color='black', ls='--')
    subplots[0,0].axhline(recorded_warp_data_dict['rr_offset_load_fl'], label='Recorded, RR Warp', color='black', ls='-.')
    subplots[0,0].axhline(recorded_warp_data_dict['rl_offset_load_fl'], label='Recorded, RL Warp', color='black', ls=':')
    subplots[0,0].grid(True)
    subplots[0,0].legend()
    subplots[0,0].set_xlabel(f'Time (s)\n Weight Transfer Error: {static_errors["fl_error_delta"]:.3f} %')
    subplots[0,0].set_ylabel('Load on FL Tire (N)')

    #  Plotting FR Loads at different warp conditions.
    subplots[0,1].plot(force_function.time, shaker_results_fr['tire_load_fr'], label='Simulated, FR warp')
    subplots[0,1].plot(force_function.time, shaker_results_fl['tire_load_fr'], label='Simulated, FL warp')
    subplots[0,1].plot(force_function.time, shaker_results_rr['tire_load_fr'], label='Simulated, RR warp')
    subplots[0,1].plot(force_function.time, shaker_results_rl['tire_load_fr'], label='Simulated, RL warp')
    subplots[0,1].axhline(recorded_warp_data_dict['fr_offset_load_fr'], label='Recorded, FR Warp', color='black')
    subplots[0,1].axhline(recorded_warp_data_dict['fl_offset_load_fr'], label='Recorded, FL Warp', color='black', ls='--')
    subplots[0,1].axhline(recorded_warp_data_dict['rr_offset_load_fr'], label='Recorded, RR Warp', color='black', ls='-.')
    subplots[0,1].axhline(recorded_warp_data_dict['rl_offset_load_fr'], label='Recorded, RL Warp', color='black', ls=':')
    subplots[0,1].grid(True)
    subplots[0,1].legend()
    subplots[0,1].set_xlabel(f'Time (s)\n Weight Transfer Error: {static_errors["fr_error_delta"]:.3f} %')
    subplots[0,1].set_ylabel('Load on FR Tire (N)')

    #  Plotting RL Loads at different warp conditions.
    subplots[1,0].plot(force_function.time, shaker_results_fr['tire_load_rl'], label='Simulated, FR warp')
    subplots[1,0].plot(force_function.time, shaker_results_fl['tire_load_rl'], label='Simulated, FL warp')
    subplots[1,0].plot(force_function.time, shaker_results_rr['tire_load_rl'], label='Simulated, RR warp')
    subplots[1,0].plot(force_function.time, shaker_results_rl['tire_load_rl'], label='Simulated, RL warp')
    subplots[1,0].axhline(recorded_warp_data_dict['fr_offset_load_rl'], label='Recorded, FR Warp', color='black')
    subplots[1,0].axhline(recorded_warp_data_dict['fl_offset_load_rl'], label='Recorded, FL Warp', color='black', ls='--')
    subplots[1,0].axhline(recorded_warp_data_dict['rr_offset_load_rl'], label='Recorded, RR Warp', color='black', ls='-.')
    subplots[1,0].axhline(recorded_warp_data_dict['rl_offset_load_rl'], label='Recorded, RL Warp', color='black', ls=':')
    subplots[1,0].grid(True)
    subplots[1,0].legend()
    subplots[1,0].set_xlabel(f'Time (s)\n Weight Transfer Error: {static_errors["rl_error_delta"]:.3f} %')
    subplots[1,0].set_ylabel('Load on RL Tire (N)')

    #  Plotting RR Loads at different warp conditions.
    subplots[1,1].plot(force_function.time, shaker_results_fr['tire_load_rr'], label='Simulated, FR warp')
    subplots[1,1].plot(force_function.time, shaker_results_fl['tire_load_rr'], label='Simulated, FL warp')
    subplots[1,1].plot(force_function.time, shaker_results_rr['tire_load_rr'], label='Simulated, RR warp')
    subplots[1,1].plot(force_function.time, shaker_results_rl['tire_load_rr'], label='Simulated, RL warp')
    subplots[1,1].axhline(recorded_warp_data_dict['fr_offset_load_rr'], label='Recorded, FR Warp', color='black')
    subplots[1,1].axhline(recorded_warp_data_dict['fl_offset_load_rr'], label='Recorded, FL Warp', color='black', ls='--')
    subplots[1,1].axhline(recorded_warp_data_dict['rr_offset_load_rr'], label='Recorded, RR Warp', color='black', ls='-.')
    subplots[1,1].axhline(recorded_warp_data_dict['rl_offset_load_rr'], label='Recorded, RL Warp', color='black', ls=':')
    subplots[1,1].grid(True)
    subplots[1,1].legend()
    subplots[1,1].set_xlabel(f'Time (s)\n Weight Transfer Error: {static_errors["rr_error_delta"]:.3f} %')
    subplots[1,1].set_ylabel('Load on RR Tire (N)')

    fig.tight_layout()
    plt.show()

def tire_response_detail_comparison(force_function, self, other, scenario, desc):

    print('Graphing...\n')

    plt.style.use('ggplot')
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    fig, subplots = plt.subplots(2, 4, figsize=(14, 8))
    fig.suptitle(f'Tire Response Detail Comparison, {scenario}: {desc}', fontsize=14)
    fig.text(0.005, 0.005, f'{DISCLAIMER}', fontsize=8)

    subplots[0,0].hist(self['tire_load_fl'], bins=50, label='Baseline (fl)')
    subplots[0,0].hist(other['tire_load_fl'], bins=50, alpha = 0.8, label='Candidate (fl)')
    subplots[0,0].axvline(x=np.mean(self['tire_load_fl']), color='orange', linestyle='dashed', label='Baseline Mean Load (fl)')
    subplots[0,0].axvline(x=np.mean(other['tire_load_fl']), color='blue', linestyle='dashed', label='Candidate Mean Load (fl)')
    subplots[0,0].set_ylabel('Count')
    subplots[0,0].set_xlabel(f"""Tire Load (N)
Baseline: x\u0304:{np.mean(self['tire_load_fl']):.0f}, \u03C3:{np.std(self['tire_load_fl']):.0f}
Candidate: x\u0304:{np.mean(other['tire_load_fl']):.0f}, \u03C3:{np.std(other['tire_load_fl']):.0f}""")
    subplots[0,0].legend()
    subplots[0,0].grid(True)
    subplots[0,0].set_yscale('log')

    subplots[0,1].hist(self['tire_load_fr'], bins=50, label='Baseline (fr)')
    subplots[0,1].hist(other['tire_load_fr'], bins=50, alpha = 0.7, label='Candidate (fr)')
    subplots[0,1].axvline(x=np.mean(self['tire_load_fr']), color='orange', linestyle='dashed', label='Baseline Mean Load (fr)')
    subplots[0,1].axvline(x=np.mean(other['tire_load_fr']), color='blue', linestyle='dashed', label='Candidate Mean Load (fr)')
    subplots[0,1].set_ylabel('Count')
    subplots[0,1].set_xlabel(f"""Tire Load (N)
Baseline: x\u0304:{np.mean(self['tire_load_fr']):.0f}, \u03C3:{np.std(self['tire_load_fr']):.0f}
Candidate: x\u0304:{np.mean(other['tire_load_fr']):.0f}, \u03C3:{np.std(other['tire_load_fr']):.0f}""")
    subplots[0,1].legend()
    subplots[0,1].grid(True)
    subplots[0,1].set_yscale('log')

    subplots[1,0].hist(self['tire_load_rl'], bins=50, label='Baseline (rl)')
    subplots[1,0].hist(other['tire_load_rl'], bins=50, alpha = 0.7, label='Candidate (rl)')
    subplots[1,0].axvline(x=np.mean(self['tire_load_rl']), color='orange', linestyle='dashed', label='Baseline Mean Load (rl)')
    subplots[1,0].axvline(x=np.mean(other['tire_load_rl']), color='blue', linestyle='dashed', label='Candidate Mean Load (rl)')
    subplots[1,0].set_ylabel('Count')
    subplots[1,0].set_xlabel(f"""Tire Load (N)
Baseline: x\u0304:{np.mean(self['tire_load_rl']):.0f}, \u03C3:{np.std(self['tire_load_rl']):.0f}
Candidate: x\u0304:{np.mean(other['tire_load_rl']):.0f}, \u03C3:{np.std(other['tire_load_rl']):.0f}""")
    subplots[1,0].legend()
    subplots[1,0].grid(True)
    subplots[1,0].set_yscale('log')

    subplots[1,1].hist(self['tire_load_rr'], bins=50, label='Baseline (rr)')
    subplots[1,1].hist(other['tire_load_rr'], bins=50, alpha = 0.7, label='Candidate (rr)')
    subplots[1,1].axvline(x=np.mean(self['tire_load_rr']), color='orange', linestyle='dashed', label='Baseline Mean Load (rr))')
    subplots[1,1].axvline(x=np.mean(other['tire_load_rr']), color='blue', linestyle='dashed', label='Candidate Mean Load (rr)')
    subplots[1,1].set_ylabel('Count')
    subplots[1,1].set_xlabel(f"""Tire Load (N)
Baseline: x\u0304:{np.mean(self['tire_load_rr']):.0f}, \u03C3:{np.std(self['tire_load_rr']):.0f}
Candidate: x\u0304:{np.mean(other['tire_load_rr']):.0f}, \u03C3:{np.std(other['tire_load_rr']):.0f}""")
    subplots[1,1].legend()
    subplots[1,1].grid(True)
    subplots[1,1].set_yscale('log')

    subplots[0,2].plot(f.fft_convert(self['tire_load_fl'])[0], f.fft_convert(self['tire_load_fl'])[1], label='Baseline (fl)')
    subplots[0,2].plot(f.fft_convert(other['tire_load_fl'])[0], f.fft_convert(other['tire_load_fl'])[1], alpha = 0.7, label='Candidate (fl)')
    subplots[0,2].set_ylabel('Normalized Tire Load Amplitude (N)')
    subplots[0,2].set_xlabel(f"Frequency (hz)\n Baseline RMS (N): {f.get_RMS(self['tire_load_fl']):.3}\n Candidate RMS (N): {f.get_RMS(other['tire_load_fl']):.3}")
    subplots[0,2].legend()
    subplots[0,2].grid(True)
    subplots[0,2].set_xscale('log')
    subplots[0,2].set_xlim(left = None, right = 100)

    subplots[0,3].plot(f.fft_convert(self['tire_load_fr'])[0], f.fft_convert(self['tire_load_fr'])[1], label='Baseline (fr)')
    subplots[0,3].plot(f.fft_convert(other['tire_load_fr'])[0], f.fft_convert(other['tire_load_fr'])[1], alpha = 0.7, label='Candidate (fr)')
    subplots[0,3].set_ylabel('Normalized Tire Load Amplitude (N)')
    subplots[0,3].set_xlabel(f"Frequency (hz)\n Baseline RMS (N): {f.get_RMS(self['tire_load_fr']):.3}\n Candidate RMS (N): {f.get_RMS(other['tire_load_fr']):.3}")
    subplots[0,3].legend()
    subplots[0,3].grid(True)
    subplots[0,3].set_xscale('log')
    subplots[0,3].set_xlim(left = None, right = 100)

    subplots[1,2].plot(f.fft_convert(self['tire_load_rl'])[0], f.fft_convert(self['tire_load_rl'])[1], label='Baseline (rl)')
    subplots[1,2].plot(f.fft_convert(other['tire_load_rl'])[0], f.fft_convert(other['tire_load_rl'])[1], alpha = 0.7, label='Candidate (rl)')
    subplots[1,2].set_ylabel('Normalized Tire Load Amplitude (N)')
    subplots[1,2].set_xlabel(f"Frequency (hz)\n Baseline RMS (N): {f.get_RMS(self['tire_load_rl']):.3}\n Candidate RMS (N): {f.get_RMS(other['tire_load_rl']):.3}")
    subplots[1,2].legend()
    subplots[1,2].grid(True)
    subplots[1,2].set_xscale('log')
    subplots[1,2].set_xlim(left = None, right = 100)

    subplots[1,3].plot(f.fft_convert(self['tire_load_rr'])[0], f.fft_convert(self['tire_load_rr'])[1], label='Baseline (rr)')
    subplots[1,3].plot(f.fft_convert(other['tire_load_rr'])[0], f.fft_convert(other['tire_load_rr'])[1], alpha = 0.7, label='Candidate (rr)')
    subplots[1,3].set_ylabel('Normalized Amplitude (N)')
    subplots[1,3].set_xlabel(f"Frequency (hz)\n Baseline RMS (N): {f.get_RMS(self['tire_load_rr']):.3}\n Candidate RMS (N): {f.get_RMS(other['tire_load_rr']):.3}")
    subplots[1,3].legend()
    subplots[1,3].grid(True)
    subplots[1,3].set_xscale('log')
    subplots[1,3].set_xlim(left = None, right = 100)

    fig.tight_layout()
    plt.show()

    return

def load_transfer_detail_comparison(force_function, self, other, scenario, desc):

    print('Graphing...\n')

    plt.style.use('ggplot')
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    fig, subplots = plt.subplots(4, 2, figsize=(12, 10))
    fig.suptitle(f'Tire Response Detail Comparison, {scenario}: {desc}', fontsize=14)
    fig.text(0.005, 0.005, f'{DISCLAIMER}', fontsize=8)

    subplots[0,0].hist(self['lateral_load_dist_f'], bins=50, label='Baseline')
    subplots[0,0].hist(other['lateral_load_dist_f'], bins=50, alpha = 0.8, label='Candidate')
    subplots[0,0].axvline(x=np.mean(self['lateral_load_dist_f']), color='orange', linestyle='dashed', label='Baseline Mean')
    subplots[0,0].axvline(x=np.mean(other['lateral_load_dist_f']), color='blue', linestyle='dashed', label='Candidate Mean')
    subplots[0,0].set_ylabel('Count')
    subplots[0,0].set_xlabel(f"""Lateral Load Distribution (Outer %, Front)
Baseline: x\u0304:{np.mean(self['lateral_load_dist_f']):.2f}, \u03C3:{np.std(self['lateral_load_dist_f']):.2f}
Candidate: x\u0304:{np.mean(other['lateral_load_dist_f']):.2f}, \u03C3:{np.std(other['lateral_load_dist_f']):.2f}""")
    subplots[0,0].legend()
    subplots[0,0].grid(True)
    subplots[0,0].set_yscale('log')

    subplots[1,0].hist(self['lateral_load_dist_r'], bins=50, label='Baseline')
    subplots[1,0].hist(other['lateral_load_dist_r'], bins=50, alpha = 0.7, label='Candidate')
    subplots[1,0].axvline(x=np.mean(self['lateral_load_dist_r']), color='orange', linestyle='dashed', label='Baseline Mean')
    subplots[1,0].axvline(x=np.mean(other['lateral_load_dist_r']), color='blue', linestyle='dashed', label='Candidate Mean')
    subplots[1,0].set_ylabel('Count')
    subplots[1,0].set_xlabel(f"""Lateral Load Distribution (Outer %, Rear)
Baseline: x\u0304:{np.mean(self['lateral_load_dist_r']):.2f}, \u03C3:{np.std(self['lateral_load_dist_r']):.2f}
Candidate: x\u0304:{np.mean(other['lateral_load_dist_r']):.2f}, \u03C3:{np.std(other['lateral_load_dist_r']):.2f}""")
    subplots[1,0].legend()
    subplots[1,0].grid(True)
    subplots[1,0].set_yscale('log')

    subplots[2,0].hist(self['lateral_load_dist_ratio'], bins=50, label='Baseline')
    subplots[2,0].hist(other['lateral_load_dist_ratio'], bins=50, alpha = 0.7, label='Candidate')
    subplots[2,0].axvline(x=np.mean(self['lateral_load_dist_ratio']), color='orange', linestyle='dashed', label='Baseline Mean')
    subplots[2,0].axvline(x=np.mean(other['lateral_load_dist_ratio']), color='blue', linestyle='dashed', label='Candidate Mean')
    subplots[2,0].set_ylabel('Count')
    subplots[2,0].set_xlabel(f"""Lateral Load Distribution Ratio (% Front)
Baseline: x\u0304:{np.mean(self['lateral_load_dist_ratio']):.2f}, \u03C3:{np.std(self['lateral_load_dist_ratio']):.2f}
Candidate: x\u0304:{np.mean(other['lateral_load_dist_ratio']):.2f}, \u03C3:{np.std(other['lateral_load_dist_ratio']):.2f}""")
    subplots[2,0].legend()
    subplots[2,0].grid(True)
    subplots[2,0].set_yscale('log')

    subplots[3,0].hist(self['long_load_dist'], bins=50, label='Baseline')
    subplots[3,0].hist(other['long_load_dist'], bins=50, alpha = 0.7, label='Candidate')
    subplots[3,0].axvline(x=np.mean(self['long_load_dist']), color='orange', linestyle='dashed', label='Baseline Mean')
    subplots[3,0].axvline(x=np.mean(other['long_load_dist']), color='blue', linestyle='dashed', label='Candidate Mean')
    subplots[3,0].set_ylabel('Count')
    subplots[3,0].set_xlabel(f"""Longitudinal Load Distribution (% Front)
Baseline: x\u0304:{np.mean(self['long_load_dist']):.2f}, \u03C3:{np.std(self['long_load_dist']):.2f}
Candidate: x\u0304:{np.mean(other['long_load_dist']):.2f}, \u03C3:{np.std(other['long_load_dist']):.2f}""")
    subplots[3,0].legend()
    subplots[3,0].grid(True)
    subplots[3,0].set_yscale('log')

    subplots[0,1].plot(f.fft_convert(self['lateral_load_dist_f'])[0], f.fft_convert(self['lateral_load_dist_f'])[1], label='Baseline')
    subplots[0,1].plot(f.fft_convert(other['lateral_load_dist_f'])[0], f.fft_convert(other['lateral_load_dist_f'])[1], alpha = 0.7, label='Candidate')
    subplots[0,1].set_ylabel('Norm. Lat. Load % Amp (Front)')
    subplots[0,1].set_xlabel(f"Frequency (hz)\n Baseline RMS (N): {f.get_RMS(self['lateral_load_dist_f']):.3}\n Candidate RMS (N): {f.get_RMS(other['lateral_load_dist_f']):.3}")
    subplots[0,1].legend()
    subplots[0,1].grid(True)
    subplots[0,1].set_xscale('log')
    subplots[0,1].set_xlim(left = None, right = 100)

    subplots[1,1].plot(f.fft_convert(self['lateral_load_dist_r'])[0], f.fft_convert(self['lateral_load_dist_r'])[1], label='Baseline')
    subplots[1,1].plot(f.fft_convert(other['lateral_load_dist_r'])[0], f.fft_convert(other['lateral_load_dist_r'])[1], alpha = 0.7, label='Candidate')
    subplots[1,1].set_ylabel('Norm. Lat. Load % Amp (Rear)')
    subplots[1,1].set_xlabel(f"Frequency (hz)\n Baseline RMS (N): {f.get_RMS(self['lateral_load_dist_r']):.3}\n Candidate RMS (N): {f.get_RMS(other['lateral_load_dist_r']):.3}")
    subplots[1,1].legend()
    subplots[1,1].grid(True)
    subplots[1,1].set_xscale('log')
    subplots[1,1].set_xlim(left = None, right = 100)

    subplots[2,1].plot(f.fft_convert(self['lateral_load_dist_ratio'])[0], f.fft_convert(self['lateral_load_dist_ratio'])[1], label='Baseline')
    subplots[2,1].plot(f.fft_convert(other['lateral_load_dist_ratio'])[0], f.fft_convert(other['lateral_load_dist_ratio'])[1], alpha = 0.7, label='Candidate')
    subplots[2,1].set_ylabel('Norm. Lat. Load Ratio % Amp')
    subplots[2,1].set_xlabel(f"Frequency (hz)\n Baseline RMS (N): {f.get_RMS(self['lateral_load_dist_ratio']):.3}\n Candidate RMS (N): {f.get_RMS(other['lateral_load_dist_ratio']):.3}")
    subplots[2,1].legend()
    subplots[2,1].grid(True)
    subplots[2,1].set_xscale('log')
    subplots[2,1].set_xlim(left = None, right = 100)

    subplots[3,1].plot(f.fft_convert(self['long_load_dist'])[0], f.fft_convert(self['long_load_dist'])[1], label='Baseline')
    subplots[3,1].plot(f.fft_convert(other['long_load_dist'])[0], f.fft_convert(other['long_load_dist'])[1], alpha = 0.7, label='Candidate')
    subplots[3,1].set_ylabel('Norm. Long Load Ratio % Amp')
    subplots[3,1].set_xlabel(f"Frequency (hz)\n Baseline RMS (N): {f.get_RMS(self['long_load_dist']):.3}\n Candidate RMS (N): {f.get_RMS(other['long_load_dist']):.3}")
    subplots[3,1].legend()
    subplots[3,1].grid(True)
    subplots[3,1].set_xscale('log')
    subplots[3,1].set_xlim(left = None, right = 100)

    fig.tight_layout()
    plt.show()

    return

def SNR_analysis(signal, control, scenario):

    print('Graphing...\n')

    plt.style.use('ggplot')
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    fig, subplots = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f'Signal-to-Noise Ratio Analysis, {scenario}', fontsize=14)
    fig.text(0.005, 0.005, f'{DISCLAIMER}', fontsize=8)

    subplots[0,0].plot(signal['gyroRotationY(rad/s)'], label='Signal (rad/s)')
    subplots[0,0].plot(control['gyroRotationY(rad/s)'], label='Control (rad/s)')
    subplots[0,0].plot(control['accelerometerAccelerationX(G)'], label='Control Lateral (G)')

    subplots[0,1].hist(signal['gyroRotationY(rad/s)'], bins=50, label='Baseline')
    subplots[0,1].hist(control['gyroRotationY(rad/s)'], bins=50, alpha = 0.8, label='Candidate')
    subplots[0,1].set_ylabel('Count')
    subplots[0,1].set_xlabel(f"Lateral Load Distribution (Outer %, Front)\n Baseline Std Dev: {np.std(signal['gyroRotationY(rad/s)']):.4}\n Candidate Std Dev: {np.std(control['gyroRotationY(rad/s)']):.4}")
    subplots[0,1].legend()
    subplots[0,1].grid(True)
    subplots[0,1].set_yscale('log')

    subplots[0,2].plot(f.fft_convert(signal['gyroRotationY(rad/s)'])[0], f.fft_convert(signal['gyroRotationY(rad/s)'])[1], label='Baseline')
    subplots[0,2].plot(f.fft_convert(control['gyroRotationY(rad/s)'])[0], f.fft_convert(control['gyroRotationY(rad/s)'])[1], alpha = 0.7, label='Candidate')
    subplots[0,2].set_ylabel('Norm. Lat. Load % Amp (Front)')
    subplots[0,2].set_xlabel(f"Frequency (hz)\n Baseline RMS (N): {f.get_RMS(signal['gyroRotationY(rad/s)']):.3}\n Candidate RMS (N): {f.get_RMS(control['gyroRotationY(rad/s)']):.3}")
    subplots[0,2].legend()
    subplots[0,2].grid(True)
    subplots[0,2].set_xscale('log')
    subplots[0,2].set_xlim(left = None, right = 100)
    
    subplots[1,0].plot(signal['gyroRotationX(rad/s)'], label='Signal (rad/s)')
    subplots[1,0].plot(control['gyroRotationX(rad/s)'], label='Control (rad/s)')
    subplots[1,0].plot(control['accelerometerAccelerationY(G)'], label='Control Lateral (G)')

    subplots[1,1].hist(signal['gyroRotationX(rad/s)'], bins=50, label='Baseline')
    subplots[1,1].hist(control['gyroRotationX(rad/s)'], bins=50, alpha = 0.8, label='Candidate')
    subplots[1,1].set_ylabel('Count')
    subplots[1,1].set_xlabel(f"Lateral Load Distribution (Outer %, Front)\n Baseline Std Dev: {np.std(signal['gyroRotationX(rad/s)']):.4}\n Candidate Std Dev: {np.std(control['gyroRotationX(rad/s)']):.4}")
    subplots[1,1].legend()
    subplots[1,1].grid(True)
    subplots[1,1].set_yscale('log')

    subplots[1,2].plot(f.fft_convert(signal['gyroRotationX(rad/s)'])[0], f.fft_convert(signal['gyroRotationX(rad/s)'])[1], label='Baseline')
    subplots[1,2].plot(f.fft_convert(control['gyroRotationX(rad/s)'])[0], f.fft_convert(control['gyroRotationX(rad/s)'])[1], alpha = 0.7, label='Candidate')
    subplots[1,2].set_ylabel('Norm. Lat. Load % Amp (Front)')
    subplots[1,2].set_xlabel(f"Frequency (hz)\n Baseline RMS (N): {f.get_RMS(signal['gyroRotationX(rad/s)']):.3}\n Candidate RMS (N): {f.get_RMS(control['gyroRotationX(rad/s)']):.3}")
    subplots[1,2].legend()
    subplots[1,2].grid(True)
    subplots[1,2].set_xscale('log')
    subplots[1,2].set_xlim(left = None, right = 100)

    fig.tight_layout()
    plt.show()

    return

def ML_set(synth_data):

    print('Graphing...\n')

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