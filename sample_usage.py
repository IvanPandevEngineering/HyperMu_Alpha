import vehicle as v
import condition_data as cdata
import matplotlib.pyplot as plt

BB = v.vehicle('Battle_Bimmer_28_Dec_2022.yml')

BB.summary()

#time, Gs, tire_load, damper_vel, body_deflection = BB.G_replay_1Dtest(telemetry_path = r'C:\\Users\\Ivan Pandev\Documents\\vsCodeTest\\sample_data\\12_March_2023\\SensorLogFiles_my_iOS_device_230314_07-21-07\\2023-03-12_14_34_29_my_iOS_device.csv')
time, tire_load, damper_vel, body_deflection = BB.G_replay(telemetry_path = r'C:\\Users\\Ivan Pandev\Documents\\vsCodeTest\\sample_data\\12_March_2023\\SensorLogFiles_my_iOS_device_230314_07-21-07\\2023-03-12_14_34_29_my_iOS_device.csv')

# time_array, c_array, tire_load = BB.Shaker()

print('Graphing...')
#plt.plot(time, Gs, label='lateral acceleration (Gs)')
plt.plot(time, body_deflection, label='body deflection (cm)')
plt.legend()
plt.show()

#a = cdata.from_sensor_log_iOS_app(path = r'C:\\Users\\Ivan Pandev\Documents\\vsCodeTest\\sample_data\\12_March_2023\\SensorLogFiles_my_iOS_device_230314_07-21-07\\2023-03-12_14_34_29_my_iOS_device.csv')
#print(a)

#print('Graphing...')
#plt.plot(a['loggingTime(txt)'], a['accelerometerAccelerationX(G)'], a['motionRoll(rad)']*180/3.14+0.6)
#plt.show()