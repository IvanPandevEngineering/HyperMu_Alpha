import vehicle as v
import condition_data as cdata
import matplotlib.pyplot as plt

BB = v.vehicle('Battle_Bimmer_28_Dec_2022.yml')

BB.summary()

#time, Gs, tire_load, damper_vel, body_deflection = BB.G_replay_1Dtest(telemetry_path = r'C:\\Users\\Ivan Pandev\Documents\\vsCodeTest\\sample_data\\12_March_2023\\SensorLogFiles_my_iOS_device_230314_07-21-07\\2023-03-12_14_34_29_my_iOS_device.csv')

#time, tire_load, damper_vel, body_deflection = BB.G_replay(telemetry_path = r'C:\\Users\\Ivan Pandev\Documents\\vsCodeTest\\sample_data\\12_March_2023\\SensorLogFiles_my_iOS_device_230314_07-21-07\\2023-03-12_14_34_29_my_iOS_device.csv')
force_function, \
tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl, \
damper_vel_fr, damper_vel_fl, damper_vel_rr, damper_vel_rl, \
lateral_load_dist_f, lateral_load_dist_r, \
roll_angle_f, roll_angle_r, pitch_angle = BB.Shaker()

# time_array, c_array, tire_load = BB.Shaker()

print('Graphing...')

fig, subplots = plt.subplots(2, 3, figsize=(12, 8))

subplots[0,0].plot(force_function['loggingTime(txt)'], force_function['accelerometerAccelerationX(G)'], label='lateral accel (G)')
subplots[0,0].plot(force_function['loggingTime(txt)'], force_function['accelerometerAccelerationY(G)'], label='longitudinal accel (G)')
subplots[0,0].legend()
subplots[0,0].grid(True)

subplots[0,1].plot(force_function['loggingTime(txt)'], roll_angle_f, label='roll angle front (deg)')
subplots[0,1].plot(force_function['loggingTime(txt)'], roll_angle_r, label='roll angle rear (deg)')
subplots[0,1].plot(force_function['loggingTime(txt)'], pitch_angle, label='pitch angle (deg)')
subplots[0,1].legend()
subplots[0,1].grid(True)

subplots[0,2].plot(force_function['loggingTime(txt)'], damper_vel_fr, label='damper speed (m/s, fr)')
subplots[0,2].plot(force_function['loggingTime(txt)'], damper_vel_fl, label='damper speed (m/s, fl)')
subplots[0,2].plot(force_function['loggingTime(txt)'], damper_vel_rr, label='damper speed (m/s, rr)')
subplots[0,2].plot(force_function['loggingTime(txt)'], damper_vel_rl, label='damper speed (m/s, rl)')
subplots[0,2].legend()
subplots[0,2].grid(True)

subplots[1,0].plot(force_function['loggingTime(txt)'], tire_load_fr, label='tire load (N, fr)')
subplots[1,0].plot(force_function['loggingTime(txt)'], tire_load_fl, label='tire load (N, fl)')
subplots[1,0].plot(force_function['loggingTime(txt)'], tire_load_rr, label='tire load (N, rr)')
subplots[1,0].plot(force_function['loggingTime(txt)'], tire_load_rl, label='tire load (N, rl)')
subplots[1,0].legend()
subplots[1,0].grid(True)

subplots[1,1].plot(force_function['loggingTime(txt)'], lateral_load_dist_f, label='lateral load dist (%, f)')
subplots[1,1].plot(force_function['loggingTime(txt)'], lateral_load_dist_r, label='lateral load dist (%, r)')
subplots[1,1].legend()
subplots[1,1].grid(True)

fig.tight_layout()
plt.show()

#a = cdata.from_sensor_log_iOS_app(path = r'C:\\Users\\Ivan Pandev\Documents\\vsCodeTest\\sample_data\\12_March_2023\\SensorLogFiles_my_iOS_device_230314_07-21-07\\2023-03-12_14_34_29_my_iOS_device.csv')
#print(a)

#print('Graphing...')
#plt.plot(a['loggingTime(txt)'], a['accelerometerAccelerationX(G)'], a['motionRoll(rad)']*180/3.14+0.6)
#plt.show()