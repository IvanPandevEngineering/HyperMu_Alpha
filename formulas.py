'''
Copyright 2025 Ivan Pandev
'''

import numpy as np
from numba import jit
from scipy import stats

BINS_FOR_INTEG = 30
FREQ_DATA = 200  # hz
G = 9.80665  # m/(s**2)
K_TRAVEL_LIMIT = 1e8  # N/m, spring rate associated with component crashes like suspension bottoming
PERIOD_DATA = 1/FREQ_DATA  # s
PERIOD_DATA_MS = 1000/FREQ_DATA  # ms


'''
SECTION 1. Begin helper functions for analytical properties below:
'''

def K_eq_series(k1, k2):
    return k1*k2 / (k1 + k2)

def parallel_axis_theorem(I_cm, sm, h):  # kg*m^2, Moment of inertia about some axis h meters from the cm

    return I_cm + sm * (h**2)

def get_I_pitch_offset(cm_height, pc_height, sm_f, K_s_f_v, K_s_r_v, K_t_f, K_t_r):  # m, used as an input to find I_pitch from I_pitch_at_cm

    vertical_offset = cm_height - pc_height

    K_end_f = get_K_end(K_s_f_v, K_t_f)

    K_end_r = get_K_end(K_s_r_v, K_t_r)

    K_end_bias_f = K_end_f / (K_end_f + K_end_r)

    horizontal_offset = sm_f - K_end_bias_f

    return np.sqrt(vertical_offset**2 + horizontal_offset**2)

def get_K_end(K_s_v, K_t):
    return K_eq_series(K_s_v, K_t) * 2

'''
SECTION 2. Begin functions for supporting time-dependent solving below.
'''

def get_active_cm_height():
    return

def get_active_tire_diam():
    return

def LatLT_sm_elastic_1g_axle(sm, rc_height, cm_height, tw):  # N, transferred to outside OR lifted from inside tire
    
    return G * sm * (cm_height - rc_height) / tw

def LatLT_sm_geometric_1g_axle(sm, rc_height, tw):  # N, transferred to outside OR lifted from inside tire
    
    return G * sm * rc_height / tw

def LatLT_usm_geometric_1g_axle(usm, tire_diameter, tw, b_r, b_l):  # N, transferred to outside OR lifted from inside tire
    
    effective_tire_radius = tire_diameter/2 - (b_r + b_l)/2

    return G * usm * effective_tire_radius / tw

def LongLT_sm_elastic_1g_v3(LongG, sm, cm_height, wheel_base, nominal_engine_brake_G,
                            pc_height_braking, pc_height_accel):  # N, transferred to outside OR lifted from ONE end tire
    if LongG > nominal_engine_brake_G:  # Braking Condition
        return G * sm/4 * (cm_height - pc_height_braking) / (wheel_base/2)
    else:  # Accel Condition
        return G * sm/4 * (cm_height - pc_height_accel) / (wheel_base/2)

def LongLT_sm_geometric_1g_v3(LongG, sm, wheel_base, nominal_engine_brake_G,
                              pc_height_braking, pc_height_accel, drive_tire_diam):  # N, transferred to outside OR lifted from ONE end tire
    if LongG > nominal_engine_brake_G:  # Braking Condition
        return G * sm/4 * (pc_height_braking) / (wheel_base/2)
    else:  # Accel Condition
        return G * sm/4 * (pc_height_accel - drive_tire_diam/2) / (wheel_base/2)

#TODO: Finalize and draw out the below equations. Not done yet.
def LongLT_usm_geometric_1g(usm_f, usm_r, tire_diameter_f, tire_diameter_r, wb_end, b_fr, b_fl, b_rr, b_rl):  # N, transferred to outside OR lifted from One end tire
    
    # Prototype formula dynamically affecting tire radii. Should be applied everywhere.
    effective_tire_radius_f = tire_diameter_f/2 - (b_fr + b_fl)/2
    effective_tire_radius_r = tire_diameter_r/2 - (b_rr + b_rl)/2

    return G * ((usm_f * effective_tire_radius_f + usm_r * effective_tire_radius_r)/2) / wb_end

def get_long_LT_diff_trq(G_Long, m, drive_wheel_diam, wheel_base, nominal_engine_brake_G):
    '''Effects of engine torque, including engine braking, reacting through the differential mounts longitudinally, per ONE wheel.'''
    G_Long = min(G_Long, nominal_engine_brake_G)  # Accel Gs are recorded negative. Engine braking is slightly positive.
    trq_half_shafts = G_Long*m*G*drive_wheel_diam/2
    return trq_half_shafts/(wheel_base*2)

def get_lat_LT_diff_trq(G_Long, m, drive_wheel_diam, tw, nominal_engine_brake_G, diff_ratio):
    '''Effects of engine torque, including engine braking, reacting through the engine mounts laterally, per ONE wheel.'''
    G_Long = min(G_Long, nominal_engine_brake_G)  # Accel Gs are recorded negative. Engine braking is slightly positive.
    trq_half_shafts = G_Long*m*G*drive_wheel_diam/2
    trq_driveshaft = trq_half_shafts/diff_ratio
    return trq_driveshaft/(tw*2)

def get_inst_I_roll_properties(I_roll, tw):
    return I_roll, tw/2

def get_inst_I_pitch_properties(I_pitch, wheel_base, sm_f):
    return I_pitch, wheel_base*(1-sm_f), wheel_base*sm_f

@jit(nopython=True, cache=True)
def get_chassis_flex_LT(K_ch, a_fr, a_fl, a_rr, a_rl, tw_f, tw_r):
    'Force added or subtracted to a single wheel bue to chassis twist. Return front and rear axle values.'
    roll_angle_f_deg = 180 * np.arctan((a_fr-a_fl)/tw_f) / np.pi
    roll_angle_r_deg = 180 * np.arctan((a_rr-a_rl)/tw_r) / np.pi

    return K_ch * (roll_angle_f_deg - roll_angle_r_deg) / (tw_f / 2), K_ch * (roll_angle_f_deg - roll_angle_r_deg) / (tw_r / 2)

def get_ARB_F(K_arb, a_r, b_r, a_l, b_l):
    return K_arb * ((a_r - b_r) - (a_l - b_l))

def get_tire_spring_F(K_t, b, c):
    return max(K_t * (b - c), 0)

def get_tire_damper_F(C_t, b_d, c_d):
    return C_t * (b_d - c_d)

def get_tire_load(tire_spring_F, tire_damper_F):
    return max(tire_spring_F + tire_damper_F, 0)

def get_damper_vel(a_d, b_d, active_motion_ratio):
    'Inputs are given at the wheel. Returns damper velocity at the damper.'
    return (a_d - b_d) / active_motion_ratio

@jit(nopython=True, cache=True)
def get_ideal_damper_force_wheel(damper_vel, speed_indecies, forces, active_motion_ratio):
    '''
    Returns damper force experienced at the wheel, without hysteresis or
    gas catridge spring effects.
    No need for explicit knee speed definition; forces are interpolated w.r.t.
    instantaneous damper speed.
    '''

    return np.interp(
        x = damper_vel,
        xp = speed_indecies,
        fp = forces
    ) / active_motion_ratio

@jit(nopython=True, cache=True)
def get_roll_angle_deg_per_axle(a_r, a_l, tw):
    return 180 * np.arctan((a_r-a_l)/tw) / np.pi

@jit(nopython=True, cache=True)
def get_pitch_angle_deg(a_fr, a_fl, a_rr, a_rl, wb):
    disp_f = (a_fr + a_fl)/2
    disp_r = (a_rr + a_rl)/2
    return 180 * np.arctan((disp_f - disp_r)/wb) / np.pi

@jit(nopython=True, cache=True)
def get_roll_angle_rate_deg_per_axle(a_r_d, a_l_d, tw):
    return 180 * np.arctan((a_r_d-a_l_d)/tw) / np.pi

@jit(nopython=True, cache=True)
def get_pitch_angle_rate_deg(a_fr_d, a_fl_d, a_rr_d, a_rl_d, wb):
    vel_f = (a_fr_d + a_fl_d)/2
    vel_r = (a_rr_d + a_rl_d)/2
    return 180 * np.arctan((vel_f - vel_r)/wb) / np.pi

def get_lateral_load_dist_axle(tire_load_r, tire_load_l):
    if (tire_load_r + tire_load_l) > 0:
        return max(
            tire_load_r / (tire_load_r + tire_load_l),
            tire_load_l / (tire_load_r + tire_load_l)
    )
    #  Handle div0 cases when both tires have zero load
    else:
        return -1

def get_lateral_load_dist_ratio(lateral_load_dist_f, lateral_load_dist_r):
    if (lateral_load_dist_f + lateral_load_dist_r) > 0:
        return lateral_load_dist_f / (lateral_load_dist_f + lateral_load_dist_r)
    #  Handle div0 cases when both axles have zero load transfer
    else:
        return -1

def get_long_load_dist(tire_load_fr, tire_load_fl, tire_load_rr, tire_load_rl):
    if tire_load_fr+tire_load_fl+tire_load_rr+tire_load_rl > 0:
        return 100 * (tire_load_fr+tire_load_fl) / (tire_load_fr+tire_load_fl+tire_load_rr+tire_load_rl)
    #  Handle div0 cases when both axles have zero load transfer
    else:
        return -1

def get_pre_init_b(sm, usm, K_t):
    'Returns at-rest tire-to-ground deflection, taken from the unloaded, free-spring position.'
    return (sm + usm) * G / K_t

def get_pre_init_a(sm, usm, K_s, K_t):
    'Returns at-rest chassis-to-ground deflection, taken from the unloaded, free-spring position.'
    return sm * G / K_s + get_pre_init_b(sm, usm, K_t)

def get_linear_spring_F_wheel(disp, rate, active_motion_ratio):
    'Returns ride spring or bump stop engagement force, at the wheel.'
    return disp * rate / active_motion_ratio

@jit(nopython=True, cache=True)
def get_hysteresis_saturation_component(a_d, b_d, weight):
    return 1 / np.cosh(weight * (a_d - b_d))

@jit(nopython=True, cache=True)
def get_hysteresis_coef(Hy, a_d, b_d):
    return Hy * get_hysteresis_saturation_component(a_d, b_d, 6)

@jit(nopython=True, cache=True)
def get_hysteresis_force(Hy, a_d, b_d, a_dd, b_dd):
    return Hy * (a_dd - b_dd) * get_hysteresis_saturation_component(a_d, b_d, 6)

def get_df_end(speed_ms, CLpA):
    return CLpA/2 * (speed_ms**2)

def get_CLpA(ref_speed, ref_df):
    '''DF = CL * V**2 * A * p/2
    2DF/V**2 = CLpA'''
    return 2*ref_df/(ref_speed**2)

def get_compression_limit_stop_force(disp, limit):
    '''
    Returns an order(s)-of-magnitude-greater spring force when suspension travels
    beyond damper travel limit in compression or spring compression limit (block height).
    '''
    return K_TRAVEL_LIMIT * max((disp - limit), 0)

def get_extension_limit_stop_force(disp, limit):
    '''
    Returns an order(s)-of-magnitude-greater spring force when suspension travels
    beyond damper travel limit in extension.
    '''
    return K_TRAVEL_LIMIT * min((disp - limit), 0)

@jit(nopython=True, cache=True)
def interp_active_motion_ratio(a, b, indecies, motion_ratios):
    'Interpolates motion ratio data points w.r.t. wheel travel to find instantaneous motion ratio.'
    return np.interp(
        x = a - b,
        xp = indecies,
        fp = motion_ratios
    )

@jit(nopython=True, cache=True)
def integ_component_disp(a, b, indecies, motion_ratios):
    '''
    Integrates motion ratio data points w.r.t. wheel travel to find component travel.
    Can be used for either springs or dampers.
    Since motion ratios are expressed in wheel travel to component travel,
    fp must equal the inverse of the motion ratio data.
    '''
    x = np.linspace(0, a-b, num=BINS_FOR_INTEG)  # range over which to integrate, vector defined by a-b
    y = np.interp(x = x, xp=indecies, fp=1/motion_ratios)  # interpolates motion ratio data at each point in vector x
    return np.trapezoid(y, x)   # simple numpy trapezoidal integration

'''
SECTION 3. General helper functions.
'''

def fft_convert(series):
    'return frequencies, normalized magnitudes of a time series from the shaker. Currently fixed at 1000hz'

    #  Remove mean from series to prevent DC component 
    series = series - np.mean(series)

    #  Perform FFT transform
    freqs = np.fft.fftfreq(len(series), PERIOD_DATA)
    mags = np.abs(np.fft.fft(series))

    #  Normalize to preserve comparability to time-domain values
    #  Normalization by sqrt(len(N)) preserves amplitude, but not energy
    mags_norm = mags / np.sqrt(len(series))

    #  Apply fftshift to remove horizontal line when plotting
    return  np.fft.fftshift(freqs), np.fft.fftshift(mags_norm)

def get_RMS(series):
    'return RMS of a time series from the shaker. Energy is a metric describing load variation.'

    mags = np.fft.fft(series)

    #  Normalized signal energy is useful for "work" calculations given other factors like displacement, etc...
    signal_energy = np.sum(np.abs(mags)**2) / len(series)

    return np.sqrt(signal_energy)

def get_RsqCorr_v_time(control, results, window_s):
    print('Calculating R-sq correlation w.r.t. time...')

    window = window_s * FREQ_DATA

    corr_series=[]
    for i in range(len(control[:-window])):
        corr_series.append(stats.linregress(control[i:i+window], results[i:i+window])[2])
    
    # print(corr_series)
    return corr_series