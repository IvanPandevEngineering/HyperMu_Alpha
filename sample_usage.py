import vehicle as v
import condition_data as cdata
import matplotlib.pyplot as plt

BB = v.vehicle(r'sample_vehicles\\Battle_Bimmer_13_Mar_2023.yml')

BB.summary()

BB.plot_shaker(
#    replay_src = r'sample_telemetry\\12_March_2023\\2023-03-12_14_34_29_my_iOS_device.csv'
)

#BB.correlation_check(
#    replay_src = r'sample_telemetry\\12_March_2023\\2023-03-12_14_34_29_my_iOS_device.csv'
#)


#BB.correlation_check(
#    replay_src = r'sample_telemetry\\16_June_2023\\2023-06-16_18_50_04_my_iOS_device.csv'
#)

#BB.damper_response(
#    #replay_src = r'sample_telemetry\\12_March_2023\\2023-03-12_14_34_29_my_iOS_device.csv'
#)

#cdata.from_sensor_log_iOS_app_dev(
#    r'sample_telemetry\\12_March_2023\\2023-03-12_14_34_29_my_iOS_device.csv'
#)

#BB.synth_data_for_ML(
#    replay_src = r'sample_telemetry\\12_March_2023\\2023-03-12_14_34_29_my_iOS_device.csv'
#)