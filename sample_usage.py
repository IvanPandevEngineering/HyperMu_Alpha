from vehicle import HyperMuVehicle

BB = HyperMuVehicle(r'sample_vehicles\\Battle_Bimmer_30_Sept_2023_w_pass.yml')

# BB.summary()

# BB.plot_shaker_basics(replay_src = 'demo')

# BB.plot_shaker_basics(replay_src = 'roll_frequency_sweep')

# BB.plot_shaker_basics(replay_src = 'curbs', return_dict = False)

# BB.plot_shaker_basics(replay_src = r'sample_telemetry\\15_Oct_2023\\2023-10-15_13_34_03_my_iOS_device.csv', smoothing_window_size_ms = 10)

# BB.correlation_rollPitchRate(replay_src = r'sample_telemetry\\15_Oct_2023\\2023-10-15_13_34_03_my_iOS_device.csv',
#                              smoothing_window_size_ms = 750)

# POST DEMO

# BB.correlation_rollPitchRate(replay_src = r'sample_telemetry\\1_Dec_2024\\2024-12-01_15_33_54_my_iOS_device.csv', smoothing_window_size_ms = 750)

BB2 = HyperMuVehicle(r'sample_vehicles\\Battle_Bimmer_30_Sept_2023_w_pass_DamperDev.yml')

# BB3 = HyperMuVehicle(r'sample_vehicles\\Battle_Bimmer_05_Jan_2025.yml')

#  Find frequencies from last video, see improvements
BB.compare_tire_response_detail(other_vehicle=BB2,
           replay_src = r'roll_frequency_sweep',
           smoothing_window_size_ms = 10)

#  Some regressions here
BB.compare_tire_response_detail(other_vehicle=BB2,
           replay_src = r'curbs',
           smoothing_window_size_ms = 10)

#  See results are not that significant for typical autoX G-profiles...
BB.compare_tire_response_detail(other_vehicle=BB2,
           replay_src = r'sample_telemetry\\15_Oct_2023\\2023-10-15_13_34_03_my_iOS_device.csv',
           smoothing_window_size_ms = 10)

#  Final Comparison
BB.compare_load_transfer_detail(other_vehicle=BB2,
           replay_src = r'sample_telemetry\\15_Oct_2023\\2023-10-15_13_34_03_my_iOS_device.csv',
           smoothing_window_size_ms = 10)