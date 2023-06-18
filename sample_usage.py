import vehicle as v
import condition_data as cdata
import matplotlib.pyplot as plt

BB = v.vehicle('Battle_Bimmer_28_Dec_2022.yml')

#BB.summary()

#BB.plot_shaker(
#    replay_src = r'sample_data\\12_March_2023\\SensorLogFiles_my_iOS_device_230314_07-21-07\\2023-03-12_14_34_29_my_iOS_device.csv'
#)

BB.correlation_check(
    replay_src = r'sample_data\\12_March_2023\\SensorLogFiles_my_iOS_device_230314_07-21-07\\2023-03-12_14_34_29_my_iOS_device.csv'
)


#BB.correlation_check(
#    replay_src = r'sample_data\\16_June_2023\\2023-06-16_18_50_04_my_iOS_device.csv'
#)