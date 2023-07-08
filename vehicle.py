import yaml
import numpy as np
import formulas as f
import condition_data as cd
from virtual_params import virtual_params
from RK4_iterator import RK4_step, RK4_iterator_1Dtest
import visualizer as vis
from chassis_model import chassis_state

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
    else:
        return value

def unpack_yml(path: str):

    with open(path, 'r') as stream:
        dict = yaml.safe_load(stream)

    for key in dict['parameters']:
        dict['parameters'][key] = make_metric(dict['parameters'][key][0], dict['parameters'][key][1])

    return(dict)

class vehicle:

    def __init__(self, vehicle_yml_path: str):

        vpd = unpack_yml(vehicle_yml_path)['parameters']  # Vehicle parameter dictionary

        self.test_lat_g = vpd['test_lat_g']
        self.df_f = vpd['test_downforce_front']
        self.df_r = vpd['test_downforce_rear']

        self.K_ch = vpd['torsional_spring_rate']
        self.wheel_base = vpd['wheel_base']
        self.tw_f = vpd['track_width_front']
        self.tw_r = vpd['track_width_rear']

        self.tire_diam_f = vpd['tire_diameter_front']
        self.tire_diam_r = vpd['tire_diameter_rear']

        self.K_s_f = vpd['spring_rate_f'] / (vpd['WS_motion_ratio_f']**2)
        self.K_s_r = vpd['spring_rate_r'] / (vpd['WS_motion_ratio_r']**2)
        self.K_arb_f = vpd['arb_rate_f']
        self.K_arb_r = vpd['arb_rate_r']

        self.C_lsc_f = vpd['slow_compression_damper_rate_f'] / (vpd['WD_motion_ratio_f']**2)
        self.C_lsc_r = vpd['slow_compression_damper_rate_r'] / (vpd['WD_motion_ratio_r']**2)
        self.C_lsr_f = vpd['slow_rebound_damper_rate_f'] / (vpd['WD_motion_ratio_f']**2)
        self.C_lsr_r = vpd['slow_rebound_damper_rate_r'] / (vpd['WD_motion_ratio_r']**2)
        self.C_hsc_f = vpd['fast_compression_damper_rate_f'] / (vpd['WD_motion_ratio_f']**2)
        self.C_hsc_r = vpd['fast_compression_damper_rate_r'] / (vpd['WD_motion_ratio_r']**2)
        self.C_hsr_f = vpd['fast_rebound_damper_rate_f'] / (vpd['WD_motion_ratio_f']**2)
        self.C_hsr_r = vpd['fast_rebound_damper_rate_r'] / (vpd['WD_motion_ratio_r']**2)
        self.knee_c_f = vpd['knee_speed_compression_f'] / (vpd['WD_motion_ratio_f']**2)
        self.knee_c_r = vpd['knee_speed_compression_r'] / (vpd['WD_motion_ratio_r']**2)
        self.knee_r_f = vpd['knee_speed_rebound_f'] / (vpd['WD_motion_ratio_f']**2)
        self.knee_r_r = vpd['knee_speed_rebound_r'] / (vpd['WD_motion_ratio_r']**2)
        self.H_C_s = vpd['hysteresis_factor']

        self.tw_v, self.K_s_f_v, self.K_s_r_v, self.K_arb_f_v, self.K_arb_r_v,\
        self.C_lsc_f_v, self.C_lsc_r_v,\
        self.C_lsr_f_v, self.C_lsr_r_v,\
        self.C_hsc_f_v, self.C_hsc_r_v,\
        self.C_hsr_f_v, self.C_hsr_r_v  = \
            \
        virtual_params(
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
        self.C_t_f = vpd['tire_rate_f'] / 1000
        self.C_t_r = vpd['tire_rate_r'] / 1000

        self.cm_height = vpd['center_of_mass_height']
        self.rc_height_f = vpd['roll_center_height_front']
        self.rc_height_r = vpd['roll_center_height_rear']
        self.pc_height_accel = vpd['pitch_center_height_accel']
        self.pc_height_decel = vpd['pitch_center_height_decel']
        self.pc_height_ic2cp = vpd['pitch_center_height_ic2cp']
        self.anti_dive = vpd['anti-dive']
        self.anti_squat = vpd['anti-squat']

        self.usm_fr = vpd['corner_unsprung_mass_fr']
        self.usm_fl = vpd['corner_unsprung_mass_fl']
        self.usm_f = (self.usm_fl + self.usm_fr)/2
        self.usm_rr = vpd['corner_unsprung_mass_rr']
        self.usm_rl = vpd['corner_unsprung_mass_rl']
        self.usm_r = (self.usm_rl + self.usm_rr)/2

        self.sm_fr = vpd['corner_mass_fr'] - vpd['corner_unsprung_mass_fr']
        self.sm_fl = vpd['corner_mass_fl'] - vpd['corner_unsprung_mass_fl']
        self.sm_rr = vpd['corner_mass_rr'] - vpd['corner_unsprung_mass_rr']
        self.sm_rl = vpd['corner_mass_rl'] - vpd['corner_unsprung_mass_rl']

        self.m = vpd['corner_mass_fr'] + vpd['corner_mass_fl'] + vpd['corner_mass_rr'] + vpd['corner_mass_rl']
        self.m_f = (vpd['corner_mass_fr'] + vpd['corner_mass_fl']) / self.m
        self.m_r = (vpd['corner_mass_rr'] + vpd['corner_mass_rl']) / self.m
        self.sm = self.sm_fr + self.sm_fl + self.sm_rr + self.sm_rl
        self.sm_f = (self.sm_fr + self.sm_fl) / self.sm
        self.sm_r = (self.sm_rr + self.sm_rl) / self.sm

        self.wheel_base_f = self.wheel_base * (1 - self.m_f)
        self.wheel_base_r = self.wheel_base * (self.m_f)

        self.I_roll_at_cg = vpd['moment_of_inertia_about_cg_roll']
        self.I_roll = f.parallel_axis_theorem(self.I_roll_at_cg, self.sm, self.cm_height - (self.rc_height_r + self.sm_f * (self.rc_height_f - self.rc_height_r)))
        self.I_pitch_at_cg = vpd['moment_of_inertia_about_cg_pitch']
        self.I_pitch = f.parallel_axis_theorem(self.I_pitch_at_cg, self.sm, f.get_I_pitch_offset(self.cm_height, self.pc_height_ic2cp, self.sm_f, self.K_s_f_v, self.K_s_r_v, self.K_t_f, self.K_t_r))
 
        self.LatLT_properties = f.roll_LatLD_per_g((self.usm_fr+self.usm_fl), (self.usm_rr + self.usm_rl), (self.sm_fr + self.sm_fl), (self.sm_rr + self.sm_rl), self.tw_v, self.tw_f, self.tw_r, self.tire_diam_f, self.tire_diam_r, self.rc_height_f, self.rc_height_r, self.cm_height, self.m, self.m_f, self.K_s_f_v, self.K_s_r_v, self.K_arb_f_v, self.K_arb_r_v, self.K_t_f, self.K_t_r, self.df_f, self.df_r)
        self.roll_tip_G = f.get_roll_tip_G(self.tw_f, self.tw_r, self.m_f, self.cm_height, self.m, self.df_f, self.df_r)
        self.aero_response = f.aero_platform_response(self.df_f, self.df_r, self.m_f, self.wheel_base, self.K_s_f_v, self.K_s_r_v, self.K_t_f, self.K_t_r)
        self.LongLD_per_g = f.pitch_LongLD_per_g(self.cm_height, self.wheel_base, self.m, self.df_f, self.df_r)
        self.pitch_gradient_accel = f.pitch_gradient(self.m, self.wheel_base, self.cm_height, self.pc_height_accel, self.K_s_f_v, self.K_s_r_v, self.K_t_f, self.K_t_r)
        self.pitch_gradient_decel = f.pitch_gradient(self.m, self.wheel_base, self.cm_height, self.pc_height_decel, self.K_s_f_v, self.K_s_r_v, self.K_t_f, self.K_t_r)
        self.pitch_frequency = f.pitch_frquency(self.I_pitch, self.sm_f, self.wheel_base, self.K_s_f_v, self.K_s_r_v, self.K_t_f, self.K_t_r)
        self.pitch_damping_slow = f.pitch_damping(self.I_pitch, self.sm_f, self.wheel_base, self.K_s_f_v, self.K_s_r_v, self.K_t_f, self.K_t_r, self.C_lsc_f_v, self.C_lsc_r_v, self.C_lsr_f_v, self.C_lsr_r_v)
        self.pitch_damping_fast = f.pitch_damping(self.I_pitch, self.sm_f, self.wheel_base, self.K_s_f_v, self.K_s_r_v, self.K_t_f, self.K_t_r, self.C_hsc_f_v, self.C_hsc_r_v, self.C_hsr_f_v, self.C_hsr_r_v)

    def summary(self):
        print('\n')
        print('_______ ROLL RESPONSE _______')
        print(f'Roll Gradient: {round(f.roll_gradient(self.tw_v, self.rc_height_f, self.rc_height_r, self.cm_height, self.m, self.m_f, self.K_s_f_v, self.K_s_r_v, self.K_arb_f_v, self.K_arb_r_v, self.K_t_f, self.K_t_r), 3)} deg/G')
        print(f'Roll Frequency: {round(f.roll_frequency(self.K_s_f_v, self.K_s_r_v, self.K_arb_f_v, self.K_arb_r_v, self.K_t_f, self.K_t_r, self.tw_v, self.I_roll), 3)} hz')
        print(f'Roll Damping Ratio (Slow): {round(f.roll_damping(self.K_s_f_v, self.K_s_r_v, self.K_arb_f_v, self.K_arb_r_v, self.K_t_f, self.K_t_r, self.tw_v, self.I_roll, self.C_lsc_f_v, self.C_lsc_r_v, self.C_lsr_f_v, self.C_lsr_r_v), 3)}')
        print(f'Roll Damping Ratio (Fast): {round(f.roll_damping(self.K_s_f_v, self.K_s_r_v, self.K_arb_f_v, self.K_arb_r_v, self.K_t_f, self.K_t_r, self.tw_v, self.I_roll, self.C_hsc_f_v, self.C_hsc_r_v, self.C_hsr_f_v, self.C_hsr_r_v), 3)}')
        print(f'Lateral Load Distribution Front: +{round(100 * self.LatLT_properties[0], 3)} % Outside/G')
        print(f'Lateral Load Distribution Rear: +{round(100 * self.LatLT_properties[1], 3)} % Outside/G')
        print(f'Lateral Load Distribution Ratio: +{round(self.LatLT_properties[2], 3)} % Front/G')
        print(f'Tip-Over G: {round(self.roll_tip_G, 3)} G')
        print('\n')

        print('_______ PITCH RESPONSE _______')
        print(f'Pitch Gradient, Accel: {round(self.pitch_gradient_accel, 3)} deg/G')
        print(f'Pitch Gradient, Decel: {round(self.pitch_gradient_decel, 3)} deg/G')
        print(f'Pitch Frequency*: {round(self.pitch_frequency, 3)} hz')
        print(f'Pitch Damping Ratio* (Slow): {round(self.pitch_damping_slow, 3)}')
        print(f'Pitch Damping Ratio* (Fast): {round(self.pitch_damping_fast, 3)}')
        print(f'Longitudinal Load Distribution: +/-{round(100*self.LongLD_per_g, 3)} % Front/G')
        print('* - taken from contact-patch-to-IC line, see documenation.')
        print('\n')

        print('_______ AERO PLATFORM RESPONSE _______')
        print(f'Pitch Angle Change: {round(self.aero_response[0], 3)} degrees, dive')
        print(f'Front Ride Height Change: {round(self.aero_response[1]*1000, 3)} mm')
        print(f'Rear Ride Height Change: {round(self.aero_response[2]*1000, 3)} mm')
        print(f'Stability Margin: {round(self.aero_response[3]*1000, 3)} mm, stability')
        print('\n')

        print('_______ FRONT RIGHT CORNER _______')
        print(f'Wheel Rate: {round(self.K_s_f/1000, 3)} N/mm')
        print(f'Natural Frequency: {round(np.sqrt(self.K_s_f/self.sm_fr)/(2*np.pi), 3)} hz')
        print(f'Damping Ratio, Slow Compression: {round(f.zeta(self.C_lsc_f, self.K_s_f, self.sm_fr), 3)} -/-')
        print(f'Damping Ratio, Slow Rebound: {round(f.zeta(self.C_lsr_f, self.K_s_f, self.sm_fr), 3)} -/-')
        print(f'Damping Ratio, Fast Compression: {round(f.zeta(self.C_hsc_f, self.K_s_f, self.sm_fr), 3)} -/-')
        print(f'Damping Ratio, Fast Rebound: {round(f.zeta(self.C_hsr_f, self.K_s_f, self.sm_fr), 3)} -/-')
        print('\n')

        print('_______ FRONT LEFT CORNER _______')
        print(f'Wheel Rate: {round(self.K_s_f/1000, 3)} N/mm')
        print(f'Natural Frequency: {round(np.sqrt(self.K_s_f/self.sm_fl)/(2*np.pi), 3)} hz')
        print(f'Damping Ratio, Slow Compression: {round(f.zeta(self.C_lsc_f, self.K_s_f, self.sm_fl), 3)} -/-')
        print(f'Damping Ratio, Slow Rebound: {round(f.zeta(self.C_lsr_f, self.K_s_f, self.sm_fl), 3)} -/-')
        print(f'Damping Ratio, Fast Compression: {round(f.zeta(self.C_hsc_f, self.K_s_f, self.sm_fl), 3)} -/-')
        print(f'Damping Ratio, Fast Rebound: {round(f.zeta(self.C_hsr_f, self.K_s_f, self.sm_fl), 3)} -/-')
        print('\n')

        print('_______ REAR RIGHT CORNER _______')
        print(f'Wheel Rate: {round(self.K_s_r/1000, 3)} N/mm')
        print(f'Natural Frequency: {round(np.sqrt(self.K_s_r/self.sm_rr)/(2*np.pi), 3)} hz')
        print(f'Damping Ratio, Slow Compression: {round(f.zeta(self.C_lsc_r, self.K_s_r, self.sm_rr), 3)} -/-')
        print(f'Damping Ratio, Slow Rebound: {round(f.zeta(self.C_lsr_r, self.K_s_r, self.sm_rr), 3)} -/-')
        print(f'Damping Ratio, Fast Compression: {round(f.zeta(self.C_hsc_r, self.K_s_r, self.sm_rr), 3)} -/-')
        print(f'Damping Ratio, Fast Rebound: {round(f.zeta(self.C_hsr_r, self.K_s_r, self.sm_rr), 3)} -/-')
        print('\n')

        print('_______ REAR LEFT CORNER _______')
        print(f'Wheel Rate: {round(self.K_s_r/1000, 3)} N/mm')
        print(f'Natural Frequency: {round(np.sqrt(self.K_s_r/self.sm_rl)/(2*np.pi), 3)} hz')
        print(f'Damping Ratio, Slow Compression: {round(f.zeta(self.C_lsc_r, self.K_s_r, self.sm_rl), 3)} -/-')
        print(f'Damping Ratio, Slow Rebound: {round(f.zeta(self.C_lsr_r, self.K_s_r, self.sm_rl), 3)} -/-')
        print(f'Damping Ratio, Fast Compression: {round(f.zeta(self.C_hsc_r, self.K_s_r, self.sm_rl), 3)} -/-')
        print(f'Damping Ratio, Fast Rebound: {round(f.zeta(self.C_hsr_r, self.K_s_r, self.sm_rl), 3)} -/-')
        print('\n')

        print('_______ MASS DISTRIBUTION _______')
        print(f'Front Axle Mass Distribution: {round(100*self.m_f, 3)} %')
        print(f'Left Mass Distribution: {round(100*(self.sm_fl+self.usm_fl+self.sm_rl+self.usm_rl)/self.m, 3)} %')
        print(f'Cross-Wise Mass Distribution (FL/RR): {round(100*(self.sm_fl+self.usm_fl+self.sm_rr+self.usm_rr)/self.m, 3)} %')
        print('\n')

    def Shaker(self, **kwargs):

        #  Create force function from chosen telemetry conversion function, selection of function TBD
        if kwargs:
            force_function = cd.from_sensor_log_iOS_app(
                kwargs['replay_src']
            )
        else:
            force_function = cd.get_demo_G_function()

        state = chassis_state(
            0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0
        )
        
        roll_angle_f, roll_angle_r, pitch_angle, \
        tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl, \
        lateral_load_dist_f, lateral_load_dist_r, \
        damper_vel_fr, damper_vel_fl, damper_vel_rr, damper_vel_rl, \
        damper_force_fr, damper_force_fl, damper_force_rr, damper_force_rl = \
        [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

        print('Starting RK4 solver for G-replay...')

        for i, row in force_function.iterrows():

            dt = force_function['timestep'][i+1]

            G_lat = row['accelerometerAccelerationX(G)']
            G_lat_next = force_function['accelerometerAccelerationX(G)'][i+1]
            G_lat_half_next = (G_lat + G_lat_next) /2
            G_long = row['accelerometerAccelerationY(G)']
            G_long_next = force_function['accelerometerAccelerationY(G)'][i+1]
            G_long_half_next = (G_long + G_long_next) /2

            state = RK4_step(  #TODO: Must append updated road surface information as well, possibly construct new chassis state here.
                dt, self, state,
                G_lat, G_long, G_lat_half_next, G_long_half_next, G_lat_next, G_long_next  # lateral and longitudinal acceleration in G
            )

            tire_load_fr.append(f.get_tire_load(self, b = b_fr, b_d = b_d_fr, c = c_fr, c_d = c_d_fr, m_dist = self.m_f)) 
            tire_load_fl.append(f.get_tire_load(self, b = b_fl, b_d = b_d_fl, c = c_fl, c_d = c_d_fl, m_dist = self.m_f))
            tire_load_rr.append(f.get_tire_load(self, b = b_rr, b_d = b_d_rr, c = c_rr, c_d = c_d_rr, m_dist = self.m_r))
            tire_load_rl.append(f.get_tire_load(self, b = b_rl, b_d = b_d_rl, c = c_rl, c_d = c_d_rl, m_dist = self.m_r))

            #damper vel is actually wheel vel, for now.
            damper_vel_fr.append(a_d_fr - b_d_fr)
            damper_vel_fl.append(a_d_fl - b_d_fl)
            damper_vel_rr.append(a_d_rr - b_d_rr)
            damper_vel_rl.append(a_d_rl - b_d_rl)

            damper_force_fr.append(f.get_damper_force(
                C_lsc = self.C_lsc_f, C_hsc = self.C_hsc_f, C_lsr = self.C_lsr_f, C_hsr = self.C_hsr_f, a_d = a_d_fr, a_dd = a_dd_fr, b_d = b_d_fr, b_dd = b_dd_fr, knee_c = self.knee_c_f, knee_r = self.knee_r_f, H_C_s = self.H_C_s
            ))
            damper_force_fl.append(f.get_damper_force(
                C_lsc = self.C_lsc_f, C_hsc = self.C_hsc_f, C_lsr = self.C_lsr_f, C_hsr = self.C_hsr_f, a_d = a_d_fl, a_dd = a_dd_fl, b_d = b_d_fl, b_dd = b_dd_fl, knee_c = self.knee_c_f, knee_r = self.knee_r_f, H_C_s = self.H_C_s
            ))
            damper_force_rr.append(f.get_damper_force(
                C_lsc = self.C_lsc_r, C_hsc = self.C_hsc_r, C_lsr = self.C_lsr_r, C_hsr = self.C_hsr_r, a_d = a_d_rr, a_dd = a_dd_rr, b_d = b_d_rr, b_dd = b_dd_rr, knee_c = self.knee_c_r, knee_r = self.knee_r_r, H_C_s = self.H_C_s
            ))
            damper_force_rl.append(f.get_damper_force(
                C_lsc = self.C_lsc_r, C_hsc = self.C_hsc_r, C_lsr = self.C_lsr_r, C_hsr = self.C_hsr_r, a_d = a_d_rl, a_dd = a_dd_rl, b_d = b_d_rl, b_dd = b_dd_rl, knee_c = self.knee_c_r, knee_r = self.knee_r_r, H_C_s = self.H_C_s
            ))

            roll_angle_f.append((a_fr - a_fl)*180/3.14)
            roll_angle_r.append((a_rr - a_rl)*180/3.14)
            pitch_angle.append((a_fr+a_fl)*180/(2*3.14) - (a_rr+a_rl)*180/(2*3.14))
            lateral_load_dist_f.append(tire_load_fr[i] / (tire_load_fr[i] + tire_load_fl[i]))
            lateral_load_dist_r.append(tire_load_rr[i] / (tire_load_rr[i] + tire_load_rl[i]))

            if i == len(force_function)-2:
                break

        print('Solver complete.')

        return(
            force_function[1:],
            tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl,
            damper_vel_fr, damper_vel_fl, damper_vel_rr, damper_vel_rl,
            damper_force_fr, damper_force_fl, damper_force_rr, damper_force_rl,
            np.array(lateral_load_dist_f), np.array(lateral_load_dist_r),
            roll_angle_f, roll_angle_r, pitch_angle
        )

    def plot_shaker(self, **kwargs):

        force_function, \
        tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl, \
        damper_vel_fr, damper_vel_fl, damper_vel_rr, damper_vel_rl, \
        damper_force_fr, damper_force_fl, damper_force_rr, damper_force_rl, \
        lateral_load_dist_f, lateral_load_dist_r, \
        roll_angle_f, roll_angle_r, pitch_angle = self.Shaker(**kwargs)

        vis.plot_response(
            force_function,
            tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl,
            damper_vel_fr, damper_vel_fl, damper_vel_rr, damper_vel_rl,
            damper_force_fr, damper_force_fl, damper_force_rr, damper_force_rl,
            lateral_load_dist_f, lateral_load_dist_r,
            roll_angle_f, roll_angle_r, pitch_angle
        )
    
    def correlation_check(self, **kwargs):
        
        force_function, \
        tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl, \
        damper_vel_fr, damper_vel_fl, damper_vel_rr, damper_vel_rl, \
        damper_force_fr, damper_force_fl, damper_force_rr, damper_force_rl, \
        lateral_load_dist_f, lateral_load_dist_r, \
        roll_angle_f, roll_angle_r, pitch_angle = self.Shaker(**kwargs)

        vis.check_correlation(
            force_function,
            tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl,
            damper_vel_fr, damper_vel_fl, damper_vel_rr, damper_vel_rl,
            damper_force_fr, damper_force_fl, damper_force_rr, damper_force_rl,
            lateral_load_dist_f, lateral_load_dist_r,
            roll_angle_f, roll_angle_r, pitch_angle
        )

    def damper_response(self, **kwargs):

        force_function, \
        tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl, \
        damper_vel_fr, damper_vel_fl, damper_vel_rr, damper_vel_rl, \
        damper_force_fr, damper_force_fl, damper_force_rr, damper_force_rl, \
        lateral_load_dist_f, lateral_load_dist_r, \
        roll_angle_f, roll_angle_r, pitch_angle = self.Shaker(**kwargs)

        vis.damper_response(
            force_function,
            tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl,
            damper_vel_fr, damper_vel_fl, damper_vel_rr, damper_vel_rl,
            damper_force_fr, damper_force_fl, damper_force_rr, damper_force_rl,
            lateral_load_dist_f, lateral_load_dist_r,
            roll_angle_f, roll_angle_r, pitch_angle
        )

# Depricated functions for development and debugging only.

    def G_replay_1Dtest(self, telemetry_path: str):

        force_function = cd.from_sensor_log_iOS_app(telemetry_path)

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
            
            #C_s = get_damper_rate()

            a, a_d, b, b_d = RK4_iterator_1Dtest(
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

            a, a_d, b, b_d = RK4_iterator_1Dtest(
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