'''
Copyright 2025 Ivan Pandev
'''

import numpy as np
from numba import jit
from scipy import stats

FREQ_DATA = 500  # hz
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

def get_aero_weight_ratio(m, df_f, df_r):  # %, Weight under aero effects as a proportion of static weight

    return (m * 9.80665 + df_f + df_r) / (m * 9.80665)

def zeta(c, K_s, sm):  # Corner damping ratio
    return c / (2 * np.sqrt(K_s * sm))

def get_wheelbase_arms(sm_f, wheel_base):  # m, front and rear pitch moment arms

    return (1-sm_f) * wheel_base, sm_f * wheel_base  # front, rear

def get_I_pitch_offset(cm_height, pc_height, sm_f, K_s_f_v, K_s_r_v, K_t_f, K_t_r):  # m, used as an input to find I_pitch from I_pitch_at_cm

    vertical_offset = cm_height - pc_height

    K_end_f = get_K_end(K_s_f_v, K_t_f)

    K_end_r = get_K_end(K_s_r_v, K_t_r)

    K_end_bias_f = K_end_f / (K_end_f + K_end_r)

    horizontal_offset = sm_f - K_end_bias_f

    return np.sqrt(vertical_offset**2 + horizontal_offset**2)

def get_disp_per_deg_roll(tw_v):  # m(suspension deflection)/deg roll, initial slope making small-angle assumption.
    
    return np.arctan(np.pi/180) * tw_v/2  # Can alternatively be expressed as (2*np.pi*tw_v/2)/360 to express m(circumference)/deg. At small angles, these are approximate. This formulation is not used since the roll model does not prescribe circular motion about the cg.

def get_disp_per_deg_pitch(wb):  # m(suspension deflection)/deg pitch, initial slope making small-angle assumption. Takes as inputs the results of the get_wheelbase_arms() method.
    
    return np.arctan(np.pi/180) * wb/2  # Can alternatively be expressed as (2*np.pi*tw_v/2)/360 to express m(circumference)/deg. At small angles, these are approximate. This formulation is not used since the roll model does not prescribe circular motion about the cg.

def get_roll_moment_1g(rc_height_r, weight_dist_f, rc_height_f, cm_height, m):  # Nm at 1g, roll moment for whole chassis

    rc_height_at_cm = rc_height_r + weight_dist_f * (rc_height_f - rc_height_r)

    return m * 9.80665 * (cm_height - rc_height_at_cm)  # Nm at 1g

def get_pitch_moment_1g(m, cm_height, pc_height):

    return m * 9.80665 * (cm_height - pc_height)

def get_K_side(K_s_f_v, K_s_r_v, K_arb_f_v, K_arb_r_v, K_t_f, K_t_r):

    return K_eq_series(K_s_f_v+K_arb_f_v, K_t_f) + K_eq_series(K_s_r_v+K_arb_r_v, K_t_r)  # N/m, left-or-right-only rate

def get_K_end(K_s_v, K_t):
    return K_eq_series(K_s_v, K_t) * 2

def get_K_roll(tw_v, K_side):  # Nm/deg, the "roll spring" stiffness

    disp_per_deg = get_disp_per_deg_roll(tw_v)

    return 2 * tw_v/2 * K_side * disp_per_deg

def get_K_pitch(sm_f, wheel_base, K_s_f_v, K_s_r_v, K_t_f, K_t_r):  # Nm/deg, the "pitch spring" stiffness
    
    wb_f, wb_r = get_wheelbase_arms(sm_f, wheel_base)

    K_end_f = get_K_end(K_s_f_v, K_t_f)

    K_end_r = get_K_end(K_s_r_v, K_t_r)

    disp_per_deg_pitch_dive = get_disp_per_deg_pitch(wb_f)

    disp_per_deg_pitch_squat = get_disp_per_deg_pitch(wb_r)

    return wb_f * K_end_f * disp_per_deg_pitch_dive + wb_r * K_end_r * disp_per_deg_pitch_squat

def roll_gradient(
        tw_v, rc_height_f, rc_height_r, cm_height, m, weight_dist_f,
        K_s_f_v, K_s_r_v, K_arb_f_v, K_arb_r_v, K_t_f, K_t_r
    ):

    roll_moment = get_roll_moment_1g(rc_height_r, weight_dist_f, rc_height_f, cm_height, m)

    K_combined_side = get_K_side(K_s_f_v, K_s_r_v, K_arb_f_v, K_arb_r_v, K_t_f, K_t_r)

    disp = roll_moment / (2* K_combined_side * tw_v/2)  # chassis edge displacement relative to ground (m) at 1 G

    return 180 * np.arctan(disp / (tw_v / 2)) / np.pi  # deg/G

def roll_frequency(K_s_f_v, K_s_r_v, K_arb_f_v, K_arb_r_v, K_t_f, K_t_r, tw_v, I_roll):

    K_side = get_K_side(K_s_f_v, K_s_r_v, K_arb_f_v, K_arb_r_v, K_t_f, K_t_r)

    K_roll = get_K_roll(tw_v, K_side)

    return (1/(2*np.pi)) * np.sqrt((180/np.pi) * K_roll / I_roll)  # Formula adapted from Optimum G tech tip 6

def roll_damping(
    K_s_f_v, K_s_r_v, K_arb_f_v, K_arb_r_v, K_t_f, K_t_r, tw_v, I_roll,
    C_c_f_v, C_c_r_v, C_r_f_v, C_r_r_v
):

    K_side = get_K_side(K_s_f_v, K_s_r_v, K_arb_f_v, K_arb_r_v, K_t_f, K_t_r)

    K_roll = get_K_roll(tw_v, K_side)  # Nm/deg

    C_cr_pol = 2 * np.sqrt((180/np.pi) * K_roll * I_roll)  # Nm/(rad/s)

    C_avg = (C_c_f_v + C_c_r_v + C_r_f_v + C_r_r_v) / 4  #N/(m/s)
    
    C_pol = 2 * C_avg * tw_v * get_disp_per_deg_roll(tw_v) * 180 / np.pi  # Nm/(rad/s)

    roll_damping_ratio = C_pol / C_cr_pol

    return roll_damping_ratio

def roll_LatLD_per_g(
    usm_f, usm_r, sm_f, sm_r,
    tw_v, tw_f, tw_r, tire_diam_f, tire_diam_r, rc_height_f, rc_height_r, cm_height, m, weight_dist_f,
    K_s_f_v, K_s_r_v, K_arb_f_v, K_arb_r_v, K_t_f, K_t_r, df_f, df_r
):  # Lateral Load Dist. ~50% = tip-over point, ~0% = no lateral G applied
    #TODO: Standardize usage of LLT/LatLD terminology

    roll_moment_per_g = get_roll_moment_1g(rc_height_r, weight_dist_f, rc_height_f, cm_height, m)

    K_combined_side = get_K_side(K_s_f_v, K_s_r_v, K_arb_f_v, K_arb_r_v, K_t_f, K_t_r)

    disp = roll_moment_per_g / (2* K_combined_side * tw_v/2)

    LLT_usm_geo_f = LatLT_usm_geometric_1g_axle(usm_f, tire_diam_f, tw_f)

    LLT_usm_geo_r = LatLT_usm_geometric_1g_axle(usm_r, tire_diam_r, tw_r)

    LLT_sm_geo_f = LatLT_sm_geometric_1g_axle(sm_f, rc_height_f, tw_f)
    
    LLT_sm_geo_r = LatLT_sm_geometric_1g_axle(sm_r, rc_height_r, tw_r)

    LLT_sm_elastic_f = LatLT_sm_elastic_alt(disp, K_s_f_v, K_arb_f_v, K_t_f)
    
    LLT_sm_elastic_r = LatLT_sm_elastic_alt(disp, K_s_r_v, K_arb_r_v, K_t_r)

    load_outside_f = (sm_f * 9.80655)/2 + LLT_usm_geo_f + LLT_sm_geo_f + LLT_sm_elastic_f + df_f/2

    load_inside_f = (sm_f * 9.80655)/2 - LLT_usm_geo_f - LLT_sm_geo_f - LLT_sm_elastic_f + df_f/2

    load_outside_r = (sm_r * 9.80655)/2 + LLT_usm_geo_r + LLT_sm_geo_r + LLT_sm_elastic_r + df_r/2

    load_inside_r = (sm_r * 9.80655)/2 - LLT_usm_geo_r - LLT_sm_geo_r - LLT_sm_elastic_r + df_r/2

    LLT_f_per_g = abs(load_outside_f) / (load_outside_f + load_inside_f) - 0.5

    LLT_r_per_g = abs(load_outside_r) / (load_outside_r + load_inside_r) - 0.5

    LLT_ratio = LLT_f_per_g / (LLT_r_per_g + LLT_f_per_g)

    return (LLT_f_per_g, LLT_r_per_g, LLT_ratio)

def LatLT_sm_elastic_alt(disp, K_s_v, K_arb_v, K_t):

    return disp * K_eq_series(K_s_v+K_arb_v, K_t)

def get_roll_tip_G(tw_f, tw_r, m_f, cm_height, m, df_f, df_r):

    tw_at_cg = tw_r + (tw_f - tw_r) * m_f

    aero_weight_ratio = get_aero_weight_ratio(m, df_f, df_r)

    return tw_at_cg * aero_weight_ratio / (2 * cm_height)

def pitch_LongLD_per_g(cm_height, wheel_base, m, df_f, df_r):

    aero_weight_ratio = get_aero_weight_ratio(m, df_f, df_r)

    return cm_height / (wheel_base * aero_weight_ratio)

def pitch_gradient(m, wheel_base, cm_height, pc_height, K_s_f_v, K_s_r_v, K_t_f, K_t_r):

    pitch_moment = get_pitch_moment_1g(m, cm_height, pc_height)

    K_end_f = get_K_end(K_s_f_v, K_t_f)

    K_end_r = get_K_end(K_s_r_v, K_t_r)

    dive = pitch_moment / (wheel_base * K_end_f)

    squat = (-1) * dive * (K_end_f/K_end_r)

    return 180 * np.arctan((dive-squat) / wheel_base)/ np.pi

def pitch_frquency(I_pitch, sm_f, wheel_base, K_s_f_v, K_s_r_v, K_t_f, K_t_r):

    K_pitch = get_K_pitch(sm_f, wheel_base, K_s_f_v, K_s_r_v, K_t_f, K_t_r)
    
    return (1/(2*np.pi)) * np.sqrt((180/np.pi) * K_pitch / I_pitch)  # Formula adapted from Optimum G tech tip 6

def pitch_damping(I_pitch, sm_f, wheel_base, K_s_f_v, K_s_r_v, K_t_f, K_t_r, C_c_f_v, C_c_r_v, C_r_f_v, C_r_r_v):

    wb_f, wb_r = get_wheelbase_arms(sm_f, wheel_base)

    K_pitch = get_K_pitch(sm_f, wheel_base, K_s_f_v, K_s_r_v, K_t_f, K_t_r)  # Nm/deg

    C_cr_pol = 2 * np.sqrt((180/np.pi) * K_pitch * I_pitch)  # Nm/(rad/s)

    C_pol_f = (C_c_f_v + C_r_f_v) * wb_f * get_disp_per_deg_pitch(wb_f) * 180 / np.pi  # Nm/(rad/s)

    C_pol_r = (C_c_r_v + C_r_r_v) * wb_r * get_disp_per_deg_pitch(wb_r) * 180 / np.pi  # Nm/(rad/s)

    return (C_pol_f + C_pol_r) / C_cr_pol

def aero_platform_response(test_df_f, test_df_r, m_f, wheel_base, K_s_f_v, K_s_r_v, K_t_f, K_t_r):

    K_end_f = get_K_end(K_s_f_v, K_t_f) #TODO: Must add heave spring effects

    K_end_r = get_K_end(K_s_r_v, K_t_r)

    dive = test_df_f / K_end_f

    squat = test_df_r / K_end_r

    pitch = 180 * np.arctan((dive-squat) / wheel_base)/ np.pi

    aero_cp_f = test_df_f / (test_df_f + test_df_r)  # % aero load distribution on the front axle

    stability_margin = wheel_base * (m_f - aero_cp_f)  # m, inches aero center of pressure is behind the center of mass

    return(pitch, dive, squat, stability_margin)  # In degrees dive, m, m, and m respectively

'''
SECTION 2. Begin functions for supporting time-dependent solving below.
#TODO:Dummy functions still to finalize.
'''

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

def get_spring_disp(a, b, WS_motion_ratio):
    'Convert wheel-to-body displacement to spring displacement'
    return (a-b) / WS_motion_ratio

def get_damper_disp(a, b, WD_motion_ratio):
    'Convert wheel-to-body displacement to damper displacement'
    return (a-b) / WD_motion_ratio

def get_ideal_damper_force(
        C_lsc, C_hsc, C_lsr, C_hsr, a_d, b_d, knee_c, knee_r
    ):
    'Inputs are given at the wheel. Returns hysteresis-free and champer-spring-free damper force at the wheel.'
    if (a_d-b_d) > 0:  # Compression domain
        if (a_d-b_d) > knee_c:  # High-speed compression domain
            return (C_hsc * (a_d - b_d - knee_c) + C_lsc * knee_c)
        else:  # Low-speed compression domain
            return (C_lsc * (a_d - b_d))
    else:  # Rebound domain
        if (a_d-b_d) < -knee_r:  # High-speed rebound domain
            return (C_hsr * (a_d - b_d + knee_r) - C_lsr * knee_r)
        else:  # Low-speed rebound domain
            return (C_lsr * (a_d - b_d))

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

def get_ride_spring_F(K_s, a, b):
    return max(K_s * (a - b), 0)

def get_ARB_F(K_arb, a_r, b_r, a_l, b_l):
    return K_arb * ((a_r - b_r) - (a_l - b_l))

def get_tire_spring_F(K_t, b, c):
    return max(K_t * (b - c), 0)

def get_tire_damper_F(C_t, b_d, c_d):
    return C_t * (b_d - c_d)

def get_tire_load(tire_spring_F, tire_damper_F):
    return max(tire_spring_F + tire_damper_F, 0)

def get_damper_vel(a_d, b_d, WD_motion_ratio):
    'Inputs are given at the wheel. Returns damper velocity at the damper.'
    return (a_d - b_d) / WD_motion_ratio
    
def get_damper_force(ride_damper_F_ideal, WD_motion_ratio):
    'Inputs are given at the wheel. Returns damper force at the damper.'
    return ride_damper_F_ideal / WD_motion_ratio**2

def get_roll_angle_deg_per_axle(a_r, a_l, tw):
    # TODO: Needs review of small angle assumption.
    return 180 * np.arctan((a_r-a_l)/tw) / np.pi

def get_pitch_angle_deg(a_fr, a_fl, a_rr, a_rl, wb):
    # TODO: Needs review of small angle assumption.
    disp_f = (a_fr + a_fl)/2
    disp_r = (a_rr + a_rl)/2
    return 180 * np.arctan((disp_f - disp_r)/wb) / np.pi

def get_roll_angle_rate_deg_per_axle(a_r_d, a_l_d, tw):
    # TODO: Needs review of small angle assumption.
    return 180 * np.arctan((a_r_d-a_l_d)/tw) / np.pi

def get_pitch_angle_rate_deg(a_fr_d, a_fl_d, a_rr_d, a_rl_d, wb):
    # TODO: Needs review of small angle assumption.
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

def get_pre_init_b(sm, usm, K_t):
    'Returns at-rest tire-to-ground deflection, taken from the unloaded, free-spring position.'
    return (sm + usm) * G / K_t

def get_pre_init_a(sm, usm, K_s, K_t):
    'Returns at-rest chassis-to-ground deflection, taken from the unloaded, free-spring position.'
    return sm * G / K_s + get_pre_init_b(sm, usm, K_t)

def get_bump_stop_F(K_bs, compression_to_bumpstop, init_a, a, init_b, b):
    'Returns bump stop engagement force. All inputs are taken at the wheel.'
    return max(K_bs * ((a-init_a) - (b-init_b) - compression_to_bumpstop), 0)

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

def get_travel_limit_stop_force(init_a, a, init_b, b, travel_limit):
    return max(K_TRAVEL_LIMIT * ((a-init_a)-(b-init_b) - travel_limit), 0)

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

def get_RsqCorr_v_time(control, results, window_s=4):
    print('Calculating R-sq correlation w.r.t. time...')

    window = window_s * FREQ_DATA

    corr_series=[]
    for i in range(len(control[:-window])):
        corr_series.append(stats.linregress(control[i:i+window], results[i:i+window])[2])
    
    # print(corr_series)
    return corr_series