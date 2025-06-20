'''
Copyright 2025 Ivan Pandev
'''

from numba import jit
import numpy as np
import pickle
from tqdm import tqdm
import yaml
import yappi

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

def make_metric(value:float, unit:str) -> float:
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
    
    for key in dict['kinematic_tables']:
        for item in dict['kinematic_tables'][key]:
            dict['kinematic_tables'][key][item] = make_metric(
                np.array(dict['kinematic_tables'][key][item][0]),
                dict['kinematic_tables'][key][item][1]
            )

    return(dict)

def error_percent(rec, sim, baseline):
    rec = rec - baseline
    sim = sim - baseline
    return 100 * (sim - rec) / rec

def trim_force_function(tuple_instance):
    modified_data = {field: np_array[:-2] for field, np_array in tuple_instance._asdict().items()}
    return cd.force_function_namedTuple(**modified_data)

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

#@jit(nopython=True, cache=True)
# JIT increases this function time for telemetry runs, speeds up init runs moderately.
def get_inputs_dt(i, force_function):

    inputs_dt = RK4.time_dependent_inputs(
        dt = force_function.timestep[i+1],

        G_lat = force_function.G_lat[i],
        G_lat_next = force_function.G_lat[i+1],
        G_lat_half_next = (force_function.G_lat[i] + force_function.G_lat[i+1])/2,

        G_long = force_function.G_long[i],
        G_long_next = force_function.G_long[i+1],
        G_long_half_next = (force_function.G_long[i] + force_function.G_long[i+1])/2,

        G_vert = force_function.G_vert[i],
        G_vert_next = force_function.G_vert[i+1],
        G_vert_half_next = (force_function.G_vert[i] + force_function.G_vert[i+1])/2,

        c_fr = force_function.c_fr[i],
        c_fl = force_function.c_fl[i],
        c_rr = force_function.c_rr[i],
        c_rl = force_function.c_rl[i],
        c_d_fr = (force_function.c_fr[i]-force_function.c_fr[i+1]) / force_function.timestep[i],
        c_d_fl = (force_function.c_fl[i]-force_function.c_fl[i+1]) / force_function.timestep[i],
        c_d_rr = (force_function.c_rr[i]-force_function.c_rr[i+1]) / force_function.timestep[i],
        c_d_rl = (force_function.c_rl[i]-force_function.c_rl[i+1]) / force_function.timestep[i],

        c_fr_next = force_function.c_fr[i+1],
        c_fl_next = force_function.c_fl[i+1],
        c_rr_next = force_function.c_rr[i+1],
        c_rl_next = force_function.c_rl[i+1],
        c_d_fr_next = (force_function.c_fr[i+1]-force_function.c_fr[i+2]) / force_function.timestep[i+1],
        c_d_fl_next = (force_function.c_fl[i+1]-force_function.c_fl[i+2]) / force_function.timestep[i+1],
        c_d_rr_next = (force_function.c_rr[i+1]-force_function.c_rr[i+2]) / force_function.timestep[i+1],
        c_d_rl_next = (force_function.c_rl[i+1]-force_function.c_rl[i+2]) / force_function.timestep[i+1],
        
        speed_ms = force_function.calc_speed_ms[i],
        speed_ms_half_next = (force_function.calc_speed_ms[i+1] + force_function.calc_speed_ms[i])/2,
        speed_ms_next = force_function.calc_speed_ms[i+1]
    )

    return inputs_dt

class HyperMuVehicle:

    def __init__(self, vehicle_yml_path: str):

        user_warning()

        vpd = unpack_yml(vehicle_yml_path)['parameters']  # Vehicle parameter dictionary
        vktd = unpack_yml(vehicle_yml_path)['kinematic_tables']  # Vehicle kinematic values dictionary

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

        #  Motion ratio lookup table properties
        self.measured_WS_indecies_f = vktd['front_WS_motion_ratio']['index']
        self.measured_WS_motion_ratios_f = vktd['front_WS_motion_ratio']['ratio']
        self.measured_WS_indecies_r = vktd['rear_WS_motion_ratio']['index']
        self.measured_WS_motion_ratios_r = vktd['rear_WS_motion_ratio']['ratio']
        self.measured_WD_indecies_f = vktd['front_WD_motion_ratio']['index']
        self.measured_WD_motion_ratios_f = vktd['front_WD_motion_ratio']['ratio']
        self.measured_WD_indecies_r = vktd['rear_WD_motion_ratio']['index']
        self.measured_WD_motion_ratios_r = vktd['rear_WD_motion_ratio']['ratio']

        self.measured_damper_speeds_f = vktd['front_damper_curve']['speed']
        self.measured_damper_forces_f = vktd['front_damper_curve']['force']
        self.measured_damper_speeds_r = vktd['rear_damper_curve']['speed']
        self.measured_damper_forces_r = vktd['rear_damper_curve']['force']

        #  Pre-initialized active motion ratios
        self.WS_motion_ratio_preinit_fr = vktd['front_WS_motion_ratio']['ratio'][0]
        self.WS_motion_ratio_preinit_fl = vktd['front_WS_motion_ratio']['ratio'][0]
        self.WS_motion_ratio_preinit_rr = vktd['rear_WS_motion_ratio']['ratio'][0]
        self.WS_motion_ratio_preinit_rl = vktd['rear_WS_motion_ratio']['ratio'][0]
        self.WD_motion_ratio_preinit_fr = vktd['front_WD_motion_ratio']['ratio'][0]
        self.WD_motion_ratio_preinit_fl = vktd['front_WD_motion_ratio']['ratio'][0]
        self.WD_motion_ratio_preinit_rr = vktd['rear_WD_motion_ratio']['ratio'][0]
        self.WD_motion_ratio_preinit_rl = vktd['rear_WD_motion_ratio']['ratio'][0]

        #  Active K's to be used in ChassisDyne chassis model solver. All spring rates converted to at-wheel rates.
        self.ride_spring_rate_f = vpd['spring_rate_f']
        self.ride_spring_rate_r = vpd['spring_rate_r']
        self.bump_stop_rate_f = vpd['bump_stop_spring_rate_f']
        self.bump_stop_rate_r = vpd['bump_stop_spring_rate_r']
        self.K_arb_f = vpd['arb_rate_f']
        self.K_arb_r = vpd['arb_rate_r']

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
        self.resting_load_fr_N = vpd['corner_mass_fr'] * f.G
        self.resting_load_fl_N = vpd['corner_mass_fl'] * f.G
        self.resting_load_rr_N = vpd['corner_mass_rr'] * f.G
        self.resting_load_rl_N = vpd['corner_mass_rl'] * f.G

        self.wheel_base_f = self.wheel_base * (1 - self.m_f)
        self.wheel_base_r = self.wheel_base * (self.m_f)
        self.max_compression_f = vpd['max_compression_front']  # Taken at wheel displacement
        self.max_compression_r = vpd['max_compression_rear']  # Taken at wheel displacement
        self.max_droop_f = vpd['max_droop_front']  # No W/S, W/D convertions, because droop values taken at wheel originally.
        self.max_droop_r = vpd['max_droop_rear']  # No W/S, W/D convertions, because droop values taken at wheel originally.
        self.compression_to_bumpstop_front = vpd['compression_to_bumpstop_front']
        self.compression_to_bumpstop_rear = vpd['compression_to_bumpstop_rear']

        self.nominal_engine_brake_G = vpd['nominal_engine_brake_G']
        self.differential_ratio = vpd['differential_ratio']

        # Very rough estimates pre-initializing in the class instance.
        self.init_a_fr = f.get_pre_init_a(self.sm_fr, self.usm_fr, self.ride_spring_rate_f, self.K_t_f)  # initial a_fr
        self.init_a_fl = f.get_pre_init_a(self.sm_fl, self.usm_fl, self.ride_spring_rate_f, self.K_t_f)  # initial a_fl
        self.init_a_rr = f.get_pre_init_a(self.sm_rr, self.usm_rr, self.ride_spring_rate_r, self.K_t_r)  # initial a_rr
        self.init_a_rl = f.get_pre_init_a(self.sm_rl, self.usm_rl, self.ride_spring_rate_r, self.K_t_r)  # initial a_rl
        self.init_b_fr = f.get_pre_init_b(self.sm_fr, self.usm_fr, self.K_t_f)  # initial b_fr
        self.init_b_fl = f.get_pre_init_b(self.sm_fl, self.usm_fl, self.K_t_f)  # initial b_fl
        self.init_b_rr = f.get_pre_init_b(self.sm_rr, self.usm_rr, self.K_t_r)  # initial b_rr
        self.init_b_rl = f.get_pre_init_b(self.sm_rl, self.usm_rl, self.K_t_r)  # initial b_rl

        self.I_roll_at_cg = vpd['moment_of_inertia_about_cg_roll']
        self.I_roll = f.parallel_axis_theorem(self.I_roll_at_cg, self.sm, self.cm_height - (self.rc_height_r + self.sm_f * (self.rc_height_f - self.rc_height_r)))
        self.I_pitch_at_cg = vpd['moment_of_inertia_about_cg_pitch']
        #TODO: URGENT needs serious fix. Please do this.
        self.I_pitch = f.parallel_axis_theorem(self.I_pitch_at_cg, self.sm, f.get_I_pitch_offset(self.cm_height, self.pc_height_ic2cp, self.sm_f, self.ride_spring_rate_f, self.ride_spring_rate_r, self.K_t_f, self.K_t_r))

        self.set_init_state(replay_src='init')

    def set_init_state(self, **kwargs):

        print('Solving vehicle model initial state...')

        force_function, shaker_results, scenario = self.Shaker(**kwargs)
        self.init_a_fr = shaker_results['a_fr'][-1]
        self.init_a_fl = shaker_results['a_fl'][-1]
        self.init_a_rr = shaker_results['a_rr'][-1]
        self.init_a_rl = shaker_results['a_rl'][-1]
        self.init_b_fr = shaker_results['b_fr'][-1]
        self.init_b_fl = shaker_results['b_fl'][-1]
        self.init_b_rr = shaker_results['b_rr'][-1]
        self.init_b_rl = shaker_results['b_rl'][-1]

        print('Vehicle initial state resolved. Inital displacements stored.')

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
        
        #yappi.start()
        for i in tqdm(range(len(force_function.time)-2), desc="Starting RK4 solver"):

            state, graphing_vars = RK4.RK4_step(
                self = self,
                #  tyres = tyres
                state = state,
                inputs_dt = get_inputs_dt(i, force_function)
            )

            for var in model.state_for_plotting._fields:
                graphing_dict[f'{var}'].append(getattr(graphing_vars, var))
        
        #yappi.stop()
        assert (len(force_function.time)-2) == len(graphing_dict['tire_load_fr']), 'Length mismatch.'

        print('Shaker solver complete.')
        #yappi.get_func_stats().print_all()

        return trim_force_function(force_function), graphing_dict, scenario
    
    def plot_shaker_basics(self, desc, **kwargs):

        force_function, shaker_results, scenario = self.Shaker(**kwargs)
        vis.plot_basics(force_function, shaker_results, scenario, desc)
    
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
    
    def compare_tire_response_detail(self, other_vehicle, desc, **kwargs):
        
        force_function, shaker_results_self, scenario = self.Shaker(**kwargs)
        force_function, shaker_results_other, scenario = other_vehicle.Shaker(**kwargs)
        vis.tire_response_detail_comparison(force_function, shaker_results_self, shaker_results_other, scenario, desc)
    
    def compare_load_transfer_detail(self, other_vehicle, desc, **kwargs):
        
        force_function, shaker_results_self, scenario = self.Shaker(**kwargs)
        force_function, shaker_results_other, scenario = other_vehicle.Shaker(**kwargs)
        vis.load_transfer_detail_comparison(force_function, shaker_results_self, shaker_results_other, scenario, desc)

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
