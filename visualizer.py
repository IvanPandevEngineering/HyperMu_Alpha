import matplotlib.pyplot as plt

def plot_response(force_function,
tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl,
damper_vel_fr, damper_vel_fl, damper_vel_rr, damper_vel_rl,
lateral_load_dist_f, lateral_load_dist_r,
roll_angle_f, roll_angle_r, pitch_angle):

    print('Graphing...')

    plt.style.use('seaborn-v0_8')
    fig, subplots = plt.subplots(3, 2, figsize=(14, 8))
    fig.suptitle('Race Telemetry on Battle_Bimmer_28_Dec_2022', fontsize=14)

    subplots[0,0].plot(force_function['loggingTime(txt)'], force_function['accelerometerAccelerationX(G)'], label='lateral accel (G)')
    subplots[0,0].plot(force_function['loggingTime(txt)'], -force_function['accelerometerAccelerationY(G)'], label='longitudinal accel (G)')
    subplots[0,0].plot(force_function['loggingTime(txt)'], force_function['c_fr']*-100, label='surface height (cm, fr)')
    subplots[0,0].plot(force_function['loggingTime(txt)'], force_function['c_rr']*-100, label='surface height (cm, rr)')
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
lateral_load_dist_f, lateral_load_dist_r,
roll_angle_f, roll_angle_r, pitch_angle):

    print('Graphing...')

    plt.style.use('seaborn-v0_8')
    fig, subplots = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Race Telemetry on Battle_Bimmer_28_Dec_2022', fontsize=14)

    subplots[0].plot(force_function['loggingTime(txt)'], (180*force_function['motionRoll(rad)']/3.14)+.6, label='Control from Sensor Data (deg)')
    #subplots[0].plot(force_function['loggingTime(txt)'], force_function['accelerometerAccelerationX(G)'], label='lateral accel (G)')
    subplots[0].plot(force_function['loggingTime(txt)'], roll_angle_f, label='roll angle (deg, f)')
    subplots[0].plot(force_function['loggingTime(txt)'], roll_angle_r, label='roll angle (deg, r)')
    subplots[0].legend()
    subplots[0].grid(True)

    subplots[1].plot(force_function['loggingTime(txt)'], 180*force_function['motionPitch(rad)']/3.14, label='Control from Sensor Data (deg)')
    subplots[1].plot(force_function['loggingTime(txt)'], force_function['accelerometerAccelerationY(G)'], label='long accel (G)')
    subplots[1].plot(force_function['loggingTime(txt)'], pitch_angle, label='pitch angle (deg)')
    subplots[1].legend()
    subplots[1].grid(True)

    fig.tight_layout()
    plt.show()