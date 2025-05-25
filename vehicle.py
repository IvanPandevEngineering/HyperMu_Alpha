'''
Copyright 2025 Ivan Pandev
'''

import numpy as np
import pickle
from tqdm import tqdm
import yaml

import chassis_model as model
import condition_data as cd
import formulas as f
import RK4_iterator as RK4
import visualizer as vis
import virtual_params as vparam


def user_warning():

    print('\n_____ SAFETY WARNING: _____\n')
    print(f'{vis.DISCLAIMER}\n')
    print('Do you agree, understand, and wish to continue? [Y]es/[N]o?\n')

    user_response = input()

    assert user_response == str('Y'), 'Not agreed.'

    pass

def make_metric(value, unit: str):
    if unit == 'in':
        return value/39.3701  # Meters
    if unit == 'lb':
        return value*0.453592  # Kilograms
    if unit == 'lbf':
        return value*4.448219  # Newtons
    if unit == 'lbf/in':
        return value*4.448219/0.0254  # N/m
    if unit == 'lbs-in^2':
        return value*0.453592/(39.3701**2)  # kg-m^2
    if unit == 'mph':
        return value/2.23694  # m/s
    else:
        return value

def unpack_yml(path: str):

    with open(path, 'r') as stream:
        dict = yaml.safe_load(stream)

    for key in dict['parameters']:
        dict['parameters'][key] = make_metric(dict['parameters'][key][0], dict['parameters'][key][1])

    return(dict)

def error_percent(rec, sim, baseline):
    rec = rec - baseline
    sim = sim - baseline
    return 100 * (sim - rec) / rec

def get_force_function(**kwargs):

    if kwargs['replay_src'] == 'demo':
        force_function = cd.get_unit_test_Slalom_w_Curbs()
        scenario = 'Unit Test: Demo'
    elif kwargs['replay_src'] == 'curbs':
        force_function = cd.get_unit_test_Curbs()
        scenario = 'Unit Test: Curb Riding'
    elif kwargs['replay_src'] == 'roll_frequency_sweep':
        force_function = cd.get_unit_test_Roll_Harmonic_Sweep()
        scenario = 'Unit Test: Lat Accel Freq Sweep'
    elif kwargs['replay_src'] == 'init':
        force_function = cd.get_init_empty()
        scenario = 'Init Empty'
    elif kwargs['replay_src'] == 'one_wheel_warp':
        force_function = cd.get_unit_test_warp(
            warp_mag = kwargs['warp_mag'],
            warp_corner = kwargs['warp_corner']
        )
        scenario = 'Unit Test: One-Wheel Warp Offset'
    else:
        if kwargs['sensor'] == 'iOS_app':
            force_function = cd.from_sensor_log_iOS_app_unbiased(
                kwargs['replay_src'],
                kwargs['filter_type'],
                kwargs['smoothing_window_size_ms'],
                kwargs['start_index'],
                kwargs['end_index'],
            )
            scenario = 'G-Replay from Telemetry'
        elif kwargs['sensor'] == 'RaceBox':
            force_function = cd.from_RaceBox(
                kwargs['replay_src'],
                kwargs['filter_type'],
                kwargs['smoothing_window_size_ms'],
                kwargs['start_index'],
                kwargs['end_index'],
            )
            scenario = 'G-Replay from Telemetry'

    return force_function, scenario

def get_inputs_dt(i, row, force_function):

    inputs_dt = RK4.time_dependent_inputs(
        G_lat = row['accelerometerAccelerationX(G)'],
        G_lat_next = force_function['accelerometerAccelerationX(G)'][i+1],
        G_lat_half_next = (row['accelerometerAccelerationX(G)'] + force_function['accelerometerAccelerationX(G)'][i+1])/2, 
        G_long = row['accelerometerAccelerationY(G)'],
        G_long_next = force_function['accelerometerAccelerationY(G)'][i+1],
        G_long_half_next = (row['accelerometerAccelerationY(G)'] + force_function['accelerometerAccelerationY(G)'][i+1])/2,
        G_vert = row['accelerometerAccelerationZ(G)'],
        G_vert_next = force_function['accelerometerAccelerationZ(G)'][i+1],
        G_vert_half_next = (row['accelerometerAccelerationZ(G)'] + force_function['accelerometerAccelerationZ(G)'][i+1])/2,
        c_fr = row['c_fr'],
        c_fl = row['c_fl'],
        c_rr = row['c_rr'],
        c_rl = row['c_rl'],
        c_d_fr = (row['c_fr']-force_function['c_fr'][i+1]) / row['timestep'],
        c_d_fl = (row['c_fl']-force_function['c_fl'][i+1]) / row['timestep'],
        c_d_rr = (row['c_rr']-force_function['c_rr'][i+1]) / row['timestep'],
        c_d_rl = (row['c_rl']-force_function['c_rl'][i+1]) / row['timestep'],
        c_fr_next = force_function['c_fr'][i+1],
        c_fl_next = force_function['c_fl'][i+1],
        c_rr_next = force_function['c_rr'][i+1],
        c_rl_next = force_function['c_rl'][i+1],
        c_d_fr_next = (force_function['c_fr'][i+1]-force_function['c_fr'][i+2]) / force_function['timestep'][i+1],
        c_d_fl_next = (force_function['c_fl'][i+1]-force_function['c_fl'][i+2]) / force_function['timestep'][i+1],
        c_d_rr_next = (force_function['c_rr'][i+1]-force_function['c_rr'][i+2]) / force_function['timestep'][i+1],
        c_d_rl_next = (force_function['c_rl'][i+1]-force_function['c_rl'][i+2]) / force_function['timestep'][i+1],
        speed_ms = row['calc_speed_ms'],
        speed_ms_half_next = (force_function['calc_speed_ms'][i+1] + force_function['calc_speed_ms'][i])/2,
        speed_ms_next = force_function['calc_speed_ms'][i+1]
    )

    return inputs_dt

class HyperMuVehicle:

    def __init__(self, vehicle_yml_path: str):

        user_warning()

        vpd = unpack_yml(vehicle_yml_path)['parameters']  # Vehicle parameter dictionary

        self.test_lat_g = vpd['test_lat_g']
        self.ref_df_speed = vpd['reference_downforce_speed']
        self.ref_df_f = vpd['reference_downforce_front']
        self.ref_df_r = vpd['reference_downforce_rear']
        self.CLpA_f = f.get_CLpA(self.ref_df_speed, self.ref_df_f)
        self.CLpA_r = f.get_CLpA(self.ref_df_speed, self.ref_df_r)

        self.K_ch = vpd['torsional_spring_rate']
        self.wheel_base = vpd['wheel_base']
        self.tw_f = vpd['track_width_front']
        self.tw_r = vpd['track_width_rear']

        self.tire_diam_f = vpd['tire_diameter_front']
        self.tire_diam_r = vpd['tire_diameter_rear']

        self.WS_motion_ratio_f = vpd['WS_motion_ratio_f']
        self.WS_motion_ratio_r = vpd['WS_motion_ratio_r']
        self.WD_motion_ratio_f = vpd['WD_motion_ratio_f']
        self.WD_motion_ratio_r = vpd['WD_motion_ratio_r']

        #  K's to be used in ChassisDyne chassis model solver. All spring rates converted to at-wheel rates.
        self.K_s_f = vpd['spring_rate_f'] / self.WS_motion_ratio_f**2
        self.K_s_r = vpd['spring_rate_r'] / self.WS_motion_ratio_r**2
        self.K_bs_f = vpd['bump_stop_spring_rate_f'] / self.WS_motion_ratio_f**2 # front bump stop is at strut
        self.K_bs_r = vpd['bump_stop_spring_rate_r'] / self.WD_motion_ratio_r**2 # rear bump stop is at damper
        self.K_arb_f = vpd['arb_rate_f']
        self.K_arb_r = vpd['arb_rate_r']

        #  C's to be used in ChassisDyne chassis model solver. All damper rates converted to at-wheel rates.
        self.C_lsc_f = vpd['slow_compression_damper_rate_f'] / self.WD_motion_ratio_f**2
        self.C_lsc_r = vpd['slow_compression_damper_rate_r'] / self.WD_motion_ratio_r**2
        self.C_lsr_f = vpd['slow_rebound_damper_rate_f'] / self.WD_motion_ratio_f**2
        self.C_lsr_r = vpd['slow_rebound_damper_rate_r'] / self.WD_motion_ratio_r**2
        self.C_hsc_f = vpd['fast_compression_damper_rate_f'] / self.WD_motion_ratio_f**2
        self.C_hsc_r = vpd['fast_compression_damper_rate_r'] / self.WD_motion_ratio_r**2
        self.C_hsr_f = vpd['fast_rebound_damper_rate_f'] / self.WD_motion_ratio_f**2
        self.C_hsr_r = vpd['fast_rebound_damper_rate_r'] / self.WD_motion_ratio_r**2
        self.knee_c_f = vpd['knee_speed_compression_f'] / self.WD_motion_ratio_f
        self.knee_c_r = vpd['knee_speed_compression_r'] / self.WD_motion_ratio_r
        self.knee_r_f = vpd['knee_speed_rebound_f'] / self.WD_motion_ratio_f
        self.knee_r_r = vpd['knee_speed_rebound_r'] / self.WD_motion_ratio_r

        self.tw_v, self.K_s_f_v, self.K_s_r_v, self.K_arb_f_v, self.K_arb_r_v,\
        self.C_lsc_f_v, self.C_lsc_r_v,\
        self.C_lsr_f_v, self.C_lsr_r_v,\
        self.C_hsc_f_v, self.C_hsc_r_v,\
        self.C_hsr_f_v, self.C_hsr_r_v  = \
            \
        vparam.virtual_params(
            self.tw_f, self.tw_r,
            self.K_s_f, self.K_s_r,
            self.K_arb_f, self.K_arb_r,
            self.C_lsc_f, self.C_lsc_r,
            self.C_lsr_f, self.C_lsr_r,
            self.C_hsc_f, self.C_hsc_r,
            self.C_hsr_f, self.C_hsr_r
        )

        self.K_t_f = vpd['tire_rate_f']
        self.K_t_r = vpd['tire_rate_r']
        self.C_t_f = self.K_t_f / 100
        self.C_t_r = self.K_t_r / 100

        self.cm_height = vpd['center_of_mass_height']
        self.rc_height_f = vpd['roll_center_height_front']
        self.rc_height_r = vpd['roll_center_height_rear']
        self.pitch_center_height_accel = vpd['pitch_center_height_accel']
        self.pitch_center_height_braking = vpd['pitch_center_height_braking']
        self.pc_height_ic2cp = vpd['pitch_center_height_ic2cp']  # Still used for parallel axis theorem calcs, consider moving.
        self.anti_dive = vpd['anti-dive']
        self.anti_squat = vpd['anti-squat']

        #  Unpack unsprung masses
        self.usm_fr = vpd['corner_unsprung_mass_fr']
        self.usm_fl = vpd['corner_unsprung_mass_fl']
        self.usm_f = (self.usm_fl + self.usm_fr)/2
        self.usm_rr = vpd['corner_unsprung_mass_rr']
        self.usm_rl = vpd['corner_unsprung_mass_rl']
        self.usm_r = (self.usm_rl + self.usm_rr)/2

        #  Unpack sprung masses
        self.sm_fr = vpd['corner_mass_fr'] - self.usm_fr
        self.sm_fl = vpd['corner_mass_fl'] - self.usm_fl
        self.sm_rr = vpd['corner_mass_rr'] - self.usm_rr
        self.sm_rl = vpd['corner_mass_rl'] - self.usm_rl
        self.sm = self.sm_fr + self.sm_fl + self.sm_rr + self.sm_rl
        self.sm_f = (self.sm_fr + self.sm_fl) / self.sm
        self.sm_r = (self.sm_rr + self.sm_rl) / self.sm

        #  Unpack total masses
        self.m = vpd['corner_mass_fr'] + vpd['corner_mass_fl'] + vpd['corner_mass_rr'] + vpd['corner_mass_rl']
        self.m_f = (vpd['corner_mass_fr'] + vpd['corner_mass_fl']) / self.m
        self.m_r = (vpd['corner_mass_rr'] + vpd['corner_mass_rl']) / self.m
        self.resting_load_fr_N = vpd['corner_mass_fr'] * 9.80665
        self.resting_load_fl_N = vpd['corner_mass_fl'] * 9.80665
        self.resting_load_rr_N = vpd['corner_mass_rr'] * 9.80665
        self.resting_load_rl_N = vpd['corner_mass_rl'] * 9.80665

        self.wheel_base_f = self.wheel_base * (1 - self.m_f)
        self.wheel_base_r = self.wheel_base * (self.m_f)
        self.max_compression_f = vpd['max_compression_front'] * self.WS_motion_ratio_f
        self.max_compression_r = vpd['max_compression_rear'] * self.WD_motion_ratio_r
        self.max_droop_f = vpd['max_droop_front']  # No W/S, W/D convertions, because droop values taken at wheel originally.
        self.max_droop_r = vpd['max_droop_rear']  # No W/S, W/D convertions, because droop values taken at wheel originally.
        self.compression_to_bumpstop_front = vpd['compression_to_bumpstop_front'] * self.WS_motion_ratio_f
        self.compression_to_bumpstop_rear = vpd['compression_to_bumpstop_rear'] * self.WD_motion_ratio_r # rear bump stop on rear damper, not spring

        self.nominal_engine_brake_G = vpd['nominal_engine_brake_G']
        self.differential_ratio = vpd['differential_ratio']

        self.init_a_fr = f.get_pre_init_a(self.sm_fr, self.usm_fr, self.K_s_f, self.K_t_f)  # initial a_fr
        self.init_a_fl = f.get_pre_init_a(self.sm_fl, self.usm_fl, self.K_s_f, self.K_t_f)  # initial a_fl
        self.init_a_rr = f.get_pre_init_a(self.sm_rr, self.usm_rr, self.K_s_r, self.K_t_r)  # initial a_rr
        self.init_a_rl = f.get_pre_init_a(self.sm_rl, self.usm_rl, self.K_s_r, self.K_t_r)  # initial a_rl
        self.init_b_fr = f.get_pre_init_b(self.sm_fr, self.usm_fr, self.K_t_f)  # initial b_fr
        self.init_b_fl = f.get_pre_init_b(self.sm_fl, self.usm_fl, self.K_t_f)  # initial b_fl
        self.init_b_rr = f.get_pre_init_b(self.sm_rr, self.usm_rr, self.K_t_r)  # initial b_rr
        self.init_b_rl = f.get_pre_init_b(self.sm_rl, self.usm_rl, self.K_t_r)  # initial b_rl

        self.I_roll_at_cg = vpd['moment_of_inertia_about_cg_roll']
        self.I_roll = f.parallel_axis_theorem(self.I_roll_at_cg, self.sm, self.cm_height - (self.rc_height_r + self.sm_f * (self.rc_height_f - self.rc_height_r)))
        self.I_pitch_at_cg = vpd['moment_of_inertia_about_cg_pitch']
        #TODO: Review
        self.I_pitch = f.parallel_axis_theorem(self.I_pitch_at_cg, self.sm, f.get_I_pitch_offset(self.cm_height, self.pc_height_ic2cp, self.sm_f, self.K_s_f_v, self.K_s_r_v, self.K_t_f, self.K_t_r))

        self.set_init_state(replay_src='init')

        #self.LatLT_properties = f.roll_LatLD_per_g((self.usm_fr+self.usm_fl), (self.usm_rr + self.usm_rl), (self.sm_fr + self.sm_fl), (self.sm_rr + self.sm_rl), self.tw_v, self.tw_f, self.tw_r, self.tire_diam_f, self.tire_diam_r, self.rc_height_f, self.rc_height_r, self.cm_height, self.m, self.m_f, self.K_s_f_v, self.K_s_r_v, self.K_arb_f_v, self.K_arb_r_v, self.K_t_f, self.K_t_r, self.df_f, self.df_r)
        self.roll_tip_G = f.get_roll_tip_G(self.tw_f, self.tw_r, self.m_f, self.cm_height, self.m, self.ref_df_f, self.ref_df_r)
        self.aero_response = f.aero_platform_response(self.ref_df_f, self.ref_df_r, self.m_f, self.wheel_base, self.K_s_f_v, self.K_s_r_v, self.K_t_f, self.K_t_r)
        self.LongLD_per_g = f.pitch_LongLD_per_g(self.cm_height, self.wheel_base, self.m, self.ref_df_f, self.ref_df_r)
        self.pitch_gradient_accel = f.pitch_gradient(self.m, self.wheel_base, self.cm_height, self.pitch_center_height_accel, self.K_s_f_v, self.K_s_r_v, self.K_t_f, self.K_t_r)
        self.pitch_gradient_decel = f.pitch_gradient(self.m, self.wheel_base, self.cm_height, self.pitch_center_height_braking, self.K_s_f_v, self.K_s_r_v, self.K_t_f, self.K_t_r)
        self.pitch_frequency = f.pitch_frquency(self.I_pitch, self.sm_f, self.wheel_base, self.K_s_f_v, self.K_s_r_v, self.K_t_f, self.K_t_r)
        self.pitch_damping_slow = f.pitch_damping(self.I_pitch, self.sm_f, self.wheel_base, self.K_s_f_v, self.K_s_r_v, self.K_t_f, self.K_t_r, self.C_lsc_f_v, self.C_lsc_r_v, self.C_lsr_f_v, self.C_lsr_r_v)
        self.pitch_damping_fast = f.pitch_damping(self.I_pitch, self.sm_f, self.wheel_base, self.K_s_f_v, self.K_s_r_v, self.K_t_f, self.K_t_r, self.C_hsc_f_v, self.C_hsc_r_v, self.C_hsr_f_v, self.C_hsr_r_v)

    def summary(self):
        print('\n')
        print('_______ ROLL RESPONSE _______')
        print(f'Roll Gradient: {f.roll_gradient(self.tw_v, self.rc_height_f, self.rc_height_r, self.cm_height, self.m, self.m_f, self.K_s_f_v, self.K_s_r_v, self.K_arb_f_v, self.K_arb_r_v, self.K_t_f, self.K_t_r):.3f} deg/G')
        print(f'Roll Frequency: {f.roll_frequency(self.K_s_f_v, self.K_s_r_v, self.K_arb_f_v, self.K_arb_r_v, self.K_t_f, self.K_t_r, self.tw_v, self.I_roll):.3f} hz')
        print(f'Roll Damping Ratio (Slow): {f.roll_damping(self.K_s_f_v, self.K_s_r_v, self.K_arb_f_v, self.K_arb_r_v, self.K_t_f, self.K_t_r, self.tw_v, self.I_roll, self.C_lsc_f_v, self.C_lsc_r_v, self.C_lsr_f_v, self.C_lsr_r_v):.3f}')
        print(f'Roll Damping Ratio (Fast): {f.roll_damping(self.K_s_f_v, self.K_s_r_v, self.K_arb_f_v, self.K_arb_r_v, self.K_t_f, self.K_t_r, self.tw_v, self.I_roll, self.C_hsc_f_v, self.C_hsc_r_v, self.C_hsr_f_v, self.C_hsr_r_v):.3f}')
        print(f'Lateral Load Distribution Front: +{self.LatLT_properties[0]:.3%} Outside/G')
        print(f'Lateral Load Distribution Rear: +{self.LatLT_properties[1]:.3%} Outside/G')
        print(f'Lateral Load Distribution Ratio: +{self.LatLT_properties[2]:.1%} Front/G')
        print(f'Tip-Over G: {self.roll_tip_G:.3f} G')
        print('\n')

        print('_______ PITCH RESPONSE _______')
        print(f'Pitch Gradient, Accel: {self.pitch_gradient_accel:.3f} deg/G')
        print(f'Pitch Gradient, Decel: {self.pitch_gradient_decel:.3f} deg/G')
        print(f'Pitch Frequency*: {self.pitch_frequency:.3f} hz')
        print(f'Pitch Damping Ratio* (Slow): {self.pitch_damping_slow:.3f}')
        print(f'Pitch Damping Ratio* (Fast): {self.pitch_damping_fast:.3f}')
        print(f'Longitudinal Load Distribution: +/-{self.LongLD_per_g:.3%} Front/G')
        print('* - taken from contact-patch-to-IC line, see documenation.')
        print('\n')

        print('_______ AERO PLATFORM RESPONSE _______')
        print(f'Pitch Angle Change: {self.aero_response[0]:.3f} degrees, dive')
        print(f'Front Ride Height Change: {self.aero_response[1]*1000:.3f} mm')
        print(f'Rear Ride Height Change: {self.aero_response[2]*1000:.3f} mm')
        print(f'Stability Margin: {self.aero_response[3]*1000:.3f} mm, stability')
        print('\n')

        print('_______ FRONT RIGHT CORNER _______')
        print(f'Wheel Rate: {self.K_s_f/1000:.3f} N/mm')
        print(f'Natural Frequency: {np.sqrt(self.K_s_f/self.sm_fr)/(2*np.pi):.3f} hz')
        print(f'Damping Ratio, Slow Compression: {f.zeta(self.C_lsc_f, self.K_s_f, self.sm_fr):.3f} -/-')
        print(f'Damping Ratio, Slow Rebound: {f.zeta(self.C_lsr_f, self.K_s_f, self.sm_fr):.3f} -/-')
        print(f'Damping Ratio, Fast Compression: {f.zeta(self.C_hsc_f, self.K_s_f, self.sm_fr):.3f} -/-')
        print(f'Damping Ratio, Fast Rebound: {f.zeta(self.C_hsr_f, self.K_s_f, self.sm_fr):.3f} -/-')
        print('\n')

        print('_______ FRONT LEFT CORNER _______')
        print(f'Wheel Rate: {self.K_s_f/1000:.3f} N/mm')
        print(f'Natural Frequency: {np.sqrt(self.K_s_f/self.sm_fl)/(2*np.pi):.3f} hz')
        print(f'Damping Ratio, Slow Compression: {f.zeta(self.C_lsc_f, self.K_s_f, self.sm_fl):.3f} -/-')
        print(f'Damping Ratio, Slow Rebound: {f.zeta(self.C_lsr_f, self.K_s_f, self.sm_fl):.3f} -/-')
        print(f'Damping Ratio, Fast Compression: {f.zeta(self.C_hsc_f, self.K_s_f, self.sm_fl):.3f} -/-')
        print(f'Damping Ratio, Fast Rebound: {f.zeta(self.C_hsr_f, self.K_s_f, self.sm_fl):.3f} -/-')
        print('\n')

        print('_______ REAR RIGHT CORNER _______')
        print(f'Wheel Rate: {self.K_s_r/1000:.3f} N/mm')
        print(f'Natural Frequency: {np.sqrt(self.K_s_r/self.sm_rr)/(2*np.pi):.3f} hz')
        print(f'Damping Ratio, Slow Compression: {f.zeta(self.C_lsc_r, self.K_s_r, self.sm_rr):.3f} -/-')
        print(f'Damping Ratio, Slow Rebound: {f.zeta(self.C_lsr_r, self.K_s_r, self.sm_rr):.3f} -/-')
        print(f'Damping Ratio, Fast Compression: {f.zeta(self.C_hsc_r, self.K_s_r, self.sm_rr):.3f} -/-')
        print(f'Damping Ratio, Fast Rebound: {f.zeta(self.C_hsr_r, self.K_s_r, self.sm_rr):.3f} -/-')
        print('\n')

        print('_______ REAR LEFT CORNER _______')
        print(f'Wheel Rate: {self.K_s_r/1000:.3f} N/mm')
        print(f'Natural Frequency: {np.sqrt(self.K_s_r/self.sm_rl)/(2*np.pi):.3f} hz')
        print(f'Damping Ratio, Slow Compression: {f.zeta(self.C_lsc_r, self.K_s_r, self.sm_rl):.3f} -/-')
        print(f'Damping Ratio, Slow Rebound: {f.zeta(self.C_lsr_r, self.K_s_r, self.sm_rl):.3f} -/-')
        print(f'Damping Ratio, Fast Compression: {f.zeta(self.C_hsc_r, self.K_s_r, self.sm_rl):.3f} -/-')
        print(f'Damping Ratio, Fast Rebound: {f.zeta(self.C_hsr_r, self.K_s_r, self.sm_rl):.3f} -/-')
        print('\n')

        print('_______ MASS DISTRIBUTION _______')
        print(f'Front Axle Mass Distribution: {self.m_f:.3%}')
        print(f'Left Mass Distribution: {(self.sm_fl+self.usm_fl+self.sm_rl+self.usm_rl)/self.m:.3%}')
        print(f'Cross-Wise Mass Distribution (FL/RR): {(self.sm_fl+self.usm_fl+self.sm_rr+self.usm_rr)/self.m:.3%}')
        print('\n')

    def set_init_state(self, **kwargs):

        print('Solving vehicle model initial state.')

        force_function, shaker_results, scenario = self.Shaker(**kwargs)
        self.init_a_fr = shaker_results['a_fr'][-1]
        self.init_a_fl = shaker_results['a_fl'][-1]
        self.init_a_rr = shaker_results['a_rr'][-1]
        self.init_a_rl = shaker_results['a_rl'][-1]
        self.init_b_fr = shaker_results['b_fr'][-1]
        self.init_b_fl = shaker_results['b_fl'][-1]
        self.init_b_rr = shaker_results['b_rr'][-1]
        self.init_b_rl = shaker_results['b_rl'][-1]

        print('Vehicle initial state resolved.')

        return True

    def Shaker(self, **kwargs):

        #  Create force function from chosen telemetry conversion function, selection of function TBD
        force_function, scenario = get_force_function(**kwargs)

        #  Initiate the positional state of the chassis
        state = model.chassis_state(
            self.init_a_fr, self.init_a_fl, self.init_a_rr, self.init_a_rl,
            self.init_b_fr, self.init_b_fl, self.init_b_rr, self.init_b_rl,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
        )

        graphing_dict={}
        for var in model.state_for_plotting._fields:
            graphing_dict[f'{var}']=[]

        for i, row in tqdm(force_function.iterrows(), desc="Starting RK4 solver...", ncols=100):

            state, graphing_vars = RK4.RK4_step(
                dt = force_function['timestep'][i+1],
                self = self,
                state = state,
                inputs_dt = get_inputs_dt(i, row, force_function)
            )

            for var in model.state_for_plotting._fields:
                graphing_dict[f'{var}'].append(getattr(graphing_vars, var))

            if i == len(force_function)-3:
                break

        force_function = force_function[2:]
        assert len(force_function) == len(graphing_dict['tire_load_fr']), 'Length mismatch.'

        print('Shaker solver complete.\n')

        return force_function, graphing_dict, scenario
    
    def plot_shaker_basics(self, **kwargs):

        force_function, shaker_results, scenario = self.Shaker(**kwargs)
        vis.plot_basics(force_function, shaker_results, scenario)
    
    def correlation_rollPitchRate(self, **kwargs):

        force_function, shaker_results, scenario = self.Shaker(**kwargs)
        vis.check_correlation_rollPitchRate(force_function, shaker_results, scenario)
    
    def correlation_rollRateRearZ(self, **kwargs):

        shaker_results = self.Shaker(**kwargs)
        vis.check_correlation_rollRateRearZ(*shaker_results)

    def static_correlation(self, **kwargs):

        warp_data_dict = unpack_yml(kwargs['control_data_file_path'])['parameters']

        force_function_fr, shaker_results_fr, scenario_fr = self.Shaker(
            **kwargs, warp_mag = warp_data_dict['fr_offset_magnitude'], warp_corner = 'FR')
        force_function_fl, shaker_results_fl, scenario_fl = self.Shaker(
            **kwargs, warp_mag = warp_data_dict['fl_offset_magnitude'], warp_corner = 'FL')
        force_function_rr, shaker_results_rr, scenario_rr = self.Shaker(
            **kwargs, warp_mag = warp_data_dict['rr_offset_magnitude'], warp_corner = 'RR')
        force_function_rl, shaker_results_rl, scenario_rl = self.Shaker(
            **kwargs, warp_mag = warp_data_dict['rl_offset_magnitude'], warp_corner = 'RL')
        
        static_errors_dict = self.get_static_errors_dict(
            recorded_warp_data_dict = warp_data_dict,
            shaker_results_fr = shaker_results_fr,
            shaker_results_fl = shaker_results_fl,
            shaker_results_rr = shaker_results_rr,
            shaker_results_rl = shaker_results_rl
        )

        vis.check_correlation_one_wheel_warp(
            force_function = force_function_fr,
            recorded_warp_data_dict = warp_data_dict,
            shaker_results_fr = shaker_results_fr,
            shaker_results_fl = shaker_results_fl,
            shaker_results_rr = shaker_results_rr,
            shaker_results_rl = shaker_results_rl,
            static_errors = static_errors_dict
        )

    def damper_response_detail(self, **kwargs):

        force_function, shaker_results, scenario = self.Shaker(**kwargs)
        vis.damper_response_detail(force_function, shaker_results, scenario)
    
    def compare_tire_response_detail(self, other_vehicle, **kwargs):
        
        force_function, shaker_results_self, scenario = self.Shaker(**kwargs)
        force_function, shaker_results_other, scenario = other_vehicle.Shaker(**kwargs)
        vis.tire_response_detail_comparison(force_function, shaker_results_self, shaker_results_other, scenario)
    
    def compare_load_transfer_detail(self, other_vehicle, **kwargs):
        
        force_function, shaker_results_self, scenario = self.Shaker(**kwargs)
        force_function, shaker_results_other, scenario = other_vehicle.Shaker(**kwargs)
        vis.load_transfer_detail_comparison(force_function, shaker_results_self, shaker_results_other, scenario)

    def get_static_errors_dict(
        self,
        recorded_warp_data_dict,
        shaker_results_fr,
        shaker_results_fl,
        shaker_results_rr,
        shaker_results_rl
    ):
        
        errors_dict={}

        fr_error_magnitude_fr_warp = error_percent(
            rec = recorded_warp_data_dict['fr_offset_load_fr'], sim = shaker_results_fr['tire_load_fr'][-1], baseline = 0)
        fr_error_magnitude_fl_warp = error_percent(
            rec = recorded_warp_data_dict['fl_offset_load_fr'], sim = shaker_results_fl['tire_load_fr'][-1], baseline = 0)
        fr_error_magnitude_rr_warp = error_percent(
            rec = recorded_warp_data_dict['rr_offset_load_fr'], sim = shaker_results_rr['tire_load_fr'][-1], baseline = 0)
        fr_error_magnitude_rl_warp = error_percent(
            rec = recorded_warp_data_dict['rl_offset_load_fr'], sim = shaker_results_rl['tire_load_fr'][-1], baseline = 0)
        errors_dict['fr_error_magnitude'] = np.average([
            fr_error_magnitude_fr_warp,
            fr_error_magnitude_fl_warp,
            fr_error_magnitude_rr_warp,
            fr_error_magnitude_rl_warp
        ])
        
        fl_error_magnitude_fr_warp = error_percent(
            rec = recorded_warp_data_dict['fr_offset_load_fl'], sim = shaker_results_fr['tire_load_fl'][-1], baseline = 0)
        fl_error_magnitude_fl_warp = error_percent(
            rec = recorded_warp_data_dict['fl_offset_load_fl'], sim = shaker_results_fl['tire_load_fl'][-1], baseline = 0)
        fl_error_magnitude_rr_warp = error_percent(
            rec = recorded_warp_data_dict['rr_offset_load_fl'], sim = shaker_results_rr['tire_load_fl'][-1], baseline = 0)
        fl_error_magnitude_rl_warp = error_percent(
            rec = recorded_warp_data_dict['rl_offset_load_fl'], sim = shaker_results_rl['tire_load_fl'][-1], baseline = 0)
        errors_dict['fl_error_magnitude'] = np.average([
            fl_error_magnitude_fr_warp,
            fl_error_magnitude_fl_warp,
            fl_error_magnitude_rr_warp,
            fl_error_magnitude_rl_warp
        ])

        rr_error_magnitude_fr_warp = error_percent(
            rec = recorded_warp_data_dict['fr_offset_load_rr'], sim = shaker_results_fr['tire_load_rr'][-1], baseline = 0)
        rr_error_magnitude_fl_warp = error_percent(
            rec = recorded_warp_data_dict['fl_offset_load_rr'], sim = shaker_results_fl['tire_load_rr'][-1], baseline = 0)
        rr_error_magnitude_rr_warp = error_percent(
            rec = recorded_warp_data_dict['rr_offset_load_rr'], sim = shaker_results_rr['tire_load_rr'][-1], baseline = 0)
        rr_error_magnitude_rl_warp = error_percent(
            rec = recorded_warp_data_dict['rl_offset_load_rr'], sim = shaker_results_rl['tire_load_rr'][-1], baseline = 0)
        errors_dict['rr_error_magnitude'] = np.average([
            rr_error_magnitude_fr_warp,
            rr_error_magnitude_fl_warp,
            rr_error_magnitude_rr_warp,
            rr_error_magnitude_rl_warp
        ])
        
        rl_error_magnitude_fr_warp = error_percent(
            rec = recorded_warp_data_dict['fr_offset_load_rl'], sim = shaker_results_fr['tire_load_rl'][-1], baseline = 0)
        rl_error_magnitude_fl_warp = error_percent(
            rec = recorded_warp_data_dict['fl_offset_load_rl'], sim = shaker_results_fl['tire_load_rl'][-1], baseline = 0)
        rl_error_magnitude_rr_warp = error_percent(
            rec = recorded_warp_data_dict['rr_offset_load_rl'], sim = shaker_results_rr['tire_load_rl'][-1], baseline = 0)
        rl_error_magnitude_rl_warp = error_percent(
            rec = recorded_warp_data_dict['rl_offset_load_rl'], sim = shaker_results_rl['tire_load_rl'][-1], baseline = 0)
        errors_dict['rl_error_magnitude'] = np.average([
            rl_error_magnitude_fr_warp,
            rl_error_magnitude_fl_warp,
            rl_error_magnitude_rr_warp,
            rl_error_magnitude_rl_warp
        ])

        fr_error_delta_fr_warp = error_percent(
            rec = recorded_warp_data_dict['fr_offset_load_fr'], sim = shaker_results_fr['tire_load_fr'][-1], baseline = self.resting_load_fr_N)
        fr_error_delta_fl_warp = error_percent(
            rec = recorded_warp_data_dict['fl_offset_load_fr'], sim = shaker_results_fl['tire_load_fr'][-1], baseline = self.resting_load_fr_N)
        fr_error_delta_rr_warp = error_percent(
            rec = recorded_warp_data_dict['rr_offset_load_fr'], sim = shaker_results_rr['tire_load_fr'][-1], baseline = self.resting_load_fr_N)
        fr_error_delta_rl_warp = error_percent(
            rec = recorded_warp_data_dict['rl_offset_load_fr'], sim = shaker_results_rl['tire_load_fr'][-1], baseline = self.resting_load_fr_N)
        errors_dict['fr_error_delta'] = np.average([
            fr_error_delta_fr_warp,
            fr_error_delta_fl_warp,
            fr_error_delta_rr_warp,
            fr_error_delta_rl_warp
        ])
        
        fl_error_delta_fr_warp = error_percent(
            rec = recorded_warp_data_dict['fr_offset_load_fl'], sim = shaker_results_fr['tire_load_fl'][-1], baseline = self.resting_load_fl_N)
        fl_error_delta_fl_warp = error_percent(
            rec = recorded_warp_data_dict['fl_offset_load_fl'], sim = shaker_results_fl['tire_load_fl'][-1], baseline = self.resting_load_fl_N)
        fl_error_delta_rr_warp = error_percent(
            rec = recorded_warp_data_dict['rr_offset_load_fl'], sim = shaker_results_rr['tire_load_fl'][-1], baseline = self.resting_load_fl_N)
        fl_error_delta_rl_warp = error_percent(
            rec = recorded_warp_data_dict['rl_offset_load_fl'], sim = shaker_results_rl['tire_load_fl'][-1], baseline = self.resting_load_fl_N)
        errors_dict['fl_error_delta'] = np.average([
            fl_error_delta_fr_warp,
            fl_error_delta_fl_warp,
            fl_error_delta_rr_warp,
            fl_error_delta_rl_warp
        ])

        rr_error_delta_fr_warp = error_percent(
            rec = recorded_warp_data_dict['fr_offset_load_rr'], sim = shaker_results_fr['tire_load_rr'][-1], baseline = self.resting_load_rr_N)
        rr_error_delta_fl_warp = error_percent(
            rec = recorded_warp_data_dict['fl_offset_load_rr'], sim = shaker_results_fl['tire_load_rr'][-1], baseline = self.resting_load_rr_N)
        rr_error_delta_rr_warp = error_percent(
            rec = recorded_warp_data_dict['rr_offset_load_rr'], sim = shaker_results_rr['tire_load_rr'][-1], baseline = self.resting_load_rr_N)
        rr_error_delta_rl_warp = error_percent(
            rec = recorded_warp_data_dict['rl_offset_load_rr'], sim = shaker_results_rl['tire_load_rr'][-1], baseline = self.resting_load_rr_N)
        errors_dict['rr_error_delta'] = np.average([
            rr_error_delta_fr_warp,
            rr_error_delta_fl_warp,
            rr_error_delta_rr_warp,
            rr_error_delta_rl_warp
        ])
        
        rl_error_delta_fr_warp = error_percent(
            rec = recorded_warp_data_dict['fr_offset_load_rl'], sim = shaker_results_fr['tire_load_rl'][-1], baseline = self.resting_load_rl_N)
        rl_error_delta_fl_warp = error_percent(
            rec = recorded_warp_data_dict['fl_offset_load_rl'], sim = shaker_results_fl['tire_load_rl'][-1], baseline = self.resting_load_rl_N)
        rl_error_delta_rr_warp = error_percent(
            rec = recorded_warp_data_dict['rr_offset_load_rl'], sim = shaker_results_rr['tire_load_rl'][-1], baseline = self.resting_load_rl_N)
        rl_error_delta_rl_warp = error_percent(
            rec = recorded_warp_data_dict['rl_offset_load_rl'], sim = shaker_results_rl['tire_load_rl'][-1], baseline = self.resting_load_rl_N)
        errors_dict['rl_error_delta'] = np.average([
            rl_error_delta_fr_warp,
            rl_error_delta_fl_warp,
            rl_error_delta_rr_warp,
            rl_error_delta_rl_warp
        ])

        return errors_dict

# Depricated functions for development and debugging only.

    def synth_data_for_ML(self, **kwargs):
        
        '''
        Designed for a ML project which seeks to estimate CM height based on lat/long acceleration and roll/pitch angles.
        Training data for the NN contains both telemetry and simulated results.
        '''

        # Create return array
        synth_data = [('Inputs','Outputs')]
        window_size = 100

        # Create initial entry into return array, only telemetry data, no simulated data
        force_function, \
        tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl, \
        damper_vel_fr, damper_vel_fl, damper_vel_rr, damper_vel_rl, \
        damper_force_fr, damper_force_fl, damper_force_rr, damper_force_rl, \
        lateral_load_dist_f, lateral_load_dist_r, \
        roll_angle_f, roll_angle_r, pitch_angle,\
        roll_angle_rate_f, roll_angle_rate_r, pitch_angle_rate = self.Shaker(**kwargs)

        for i in range(len(force_function)-window_size):
            if i % window_size == 0:
                synth_data.append(([
                        np.array(force_function['accelerometerAccelerationX(G)'][i:i+window_size]),
                        np.array(force_function['gyroRotationY(rad/s)'][i:i+window_size]*180/np.pi)],
                        [self.cm_height]
                    ))

        # Now vary vehicle property, add simulated response to training set
        for height in np.linspace(0.28, 0.67, 20):
            self.cm_height = height
            print(f'Now solving with new parameter: {self.cm_height} cm_height')

            force_function, \
            tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl, \
            damper_vel_fr, damper_vel_fl, damper_vel_rr, damper_vel_rl, \
            damper_force_fr, damper_force_fl, damper_force_rr, damper_force_rl, \
            lateral_load_dist_f, lateral_load_dist_r, \
            roll_angle_f, roll_angle_r, pitch_angle,\
            roll_angle_rate_f, roll_angle_rate_r, pitch_angle_rate = self.Shaker(**kwargs)

            for i in range(len(force_function)-window_size):
                if i % window_size == 0:
                    synth_data.append(([
                        np.array(force_function['accelerometerAccelerationX(G)'][i:i+window_size]),
                        (roll_angle_rate_f+roll_angle_rate_r)[i:i+window_size]/2],
                        [self.cm_height]
                    ))

        with open('ML_training_data_CM.pkl', 'wb') as file:
            pickle.dump(synth_data, file)

        return synth_data

    def G_replay_1Dtest(self, telemetry_path: str):

        force_function = cd.from_sensor_log_iOS_app_unbiased(telemetry_path)

        #  Initialize variables
        a, a_d, b, b_d, c, c_d = [0] * 6
        tire_load, damper_vel, body_deflection = [],[],[]

        print('Starting RK4 solver for G-replay...')

        for i, row in force_function.iterrows():

            dt = force_function['timestep'][i+1]

            F_sm = row['accelerometerAccelerationX(G)'] * f.get_roll_moment_1g(self.rc_height_r, self.m_f, self.rc_height_f, self.cm_height, self.m) / self.tw_v
            F_usm = row['accelerometerAccelerationX(G)'] * f.LatLT_usm_geometric_1g_axle(self.m, self.tire_diam_f, self.tw_v)

            F_sm_next = force_function['accelerometerAccelerationX(G)'][i+1] * f.get_roll_moment_1g(self.rc_height_r, self.m_f, self.rc_height_f, self.cm_height, self.m) / self.tw_v
            F_usm_next = force_function['accelerometerAccelerationX(G)'][i+1] * f.LatLT_usm_geometric_1g_axle(self.m, self.tire_diam_f, self.tw_v)
            
            F_sm_half_next = (F_sm + F_sm_next) / 2
            F_usm_half_next = (F_usm + F_usm_next) / 2

            a, a_d, b, b_d = RK4.RK4_iterator_1Dtest(
                self, dt,
                a, a_d, b, b_d, c, c_d,
                F_sm, F_usm, F_sm_half_next, F_usm_half_next, F_sm_next, F_usm_next
            )

            tire_load.append(b * self.K_t_f + b_d * self.C_t_f)
            damper_vel.append(a_d)
            body_deflection.append(a*10)

            if i+2 == len(force_function):
                break

        return force_function['loggingTime(txt)'][1:], force_function['accelerometerAccelerationX(G)'][1:], tire_load, damper_vel, body_deflection
    
    def Shaker_1Dtest(self):

        time_array, c_array, dt = cd.get_bump_function()

        a, a_d, b, b_d, c, c_d = [0] * 6
        tire_load, damper_vel, body_deflection = [],[],[]

        print('Starting RK4 solver for Shaker...')

        for i, item in enumerate(c_array):

            c = item
            c_d = item + c_array[i+1] / dt

            a, a_d, b, b_d = RK4.RK4_iterator_1Dtest(
                self, dt,
                a, a_d, b, b_d, c, c_d,
                F_sm = 0, F_usm = 0, F_sm_half_next = 0, F_usm_half_next = 0, F_sm_next = 0, F_usm_next = 0
            )

            tire_load.append((b * self.K_t_f + b_d * self.C_t_f) / 400000)
            damper_vel.append(a_d)
            body_deflection.append(a * 50000)

            if i+2 == len(c_array):
                break

        return time_array[1:], c_array[1:], tire_load
