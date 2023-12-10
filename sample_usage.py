import vehicle as v

BB = v.vehicle(r'sample_vehicles\\Battle_Bimmer_30_Sept_2023_w_pass.yml')

BB.summary()

#BB.plot_shaker(
#    replay_src = r'sample_telemetry\\15_Oct_2023\\2023-10-15_13_34_03_my_iOS_device.csv',
#    smoothing_window_size_ms = 10
#)

BB.correlation_check(
    replay_src = r'sample_telemetry\\15_Oct_2023\\2023-10-15_13_34_03_my_iOS_device.csv',
    smoothing_window_size_ms = 750
)