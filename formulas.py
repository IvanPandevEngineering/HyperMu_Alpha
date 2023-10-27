import numpy as np

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

    LLT_ratio = LLT_f_per_g / LLT_r_per_g

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
    
    return 9.80665 * sm * (cm_height - rc_height) / tw

def LatLT_sm_geometric_1g_axle(sm, rc_height, tw):  # N, transferred to outside OR lifted from inside tire
    
    return 9.80665 * sm * rc_height / tw

def LatLT_usm_geometric_1g_axle(usm, tire_diameter, tw):  # N, transferred to outside OR lifted from inside tire
    
    return 9.80665 * usm * (tire_diameter/2) / tw

#TODO: Finalize and draw out the below equations. Not done yet.
#TODO: Geo arms are approximation, spend time thinking through physics
def LongLT_sm_elastic_1g_end(sm, sm_dist_opposite, anti_geo, cm_height, wb_end):  # N, transferred to outside OR lifted from ONE end tire
    
    return 9.80665 * sm * sm_dist_opposite * (cm_height * (1-anti_geo)) / (wb_end * 2)

def LongLT_sm_geometric_1g_end(sm, sm_dist_opposite, anti_geo, cm_height, wb_end):  # N, transferred to outside OR lifted from One end tire
    
    return 9.80665 * sm * sm_dist_opposite * (cm_height * (1-anti_geo)) / (wb_end * 2)

def LongLT_usm_geometric_1g_end(usm_f, usm_r, tire_diameter_f, tire_diameter_r, wb_end):  # N, transferred to outside OR lifted from One end tire
    
    return 9.80665 * ((usm_f * tire_diameter_f/2 + usm_r * tire_diameter_r/2)/2) / wb_end

#TODO: anti-dive and anti-squat averaging
def LongLT_sm_elastic_1g(sm, anti_dive, anti_squat, cm_height, wb_end):  # N, transferred to outside OR lifted from ONE end tire
    
    return 9.80665 * sm * (cm_height * (1-(anti_dive+anti_squat)/2)) / (wb_end * 4)

def LongLT_sm_geometric_1g(sm, anti_dive, anti_squat, cm_height, wb_end):  # N, transferred to outside OR lifted from One end tire
    
    return 9.80665 * sm * (cm_height * (1-(anti_dive+anti_squat)/2)) / (wb_end * 4)

def LongLT_usm_geometric_1g(usm_f, usm_r, tire_diameter_f, tire_diameter_r, wb_end):  # N, transferred to outside OR lifted from One end tire
    
    return 9.80665 * ((usm_f * tire_diameter_f/2 + usm_r * tire_diameter_r/2)/2) / wb_end


def get_ideal_damper_force(
        C_lsc, C_hsc, C_lsr, C_hsr, a_d, b_d, knee_c, knee_r
    ):
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

def get_damper_force(
        C_lsc, C_hsc, C_lsr, C_hsr, a_d, b_d, knee_c, knee_r
    ):
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

def get_inst_I_roll_properties(I_roll, a_d_r, a_d_l, tw):
    return I_roll, tw/2

def get_inst_I_pitch_properties(I_pitch, wheel_base, sm_f):
    return I_pitch, wheel_base*(1-sm_f), wheel_base*sm_f

# TO BE DEPRICATED.
def get_tire_load_dep(self, b, b_d, c, c_d, m_dist):
    return (b - c) * self.K_t_f + (b_d - c_d) * self.C_t_f

def get_chassis_flex_LT(K_ch, a_fr, a_fl, a_rr, a_rl, tw):
    return K_ch * ((a_fr - a_fl) - (a_rr - a_rl)) / (tw / 2)

def get_ride_spring_F(K_s, a, b):
    return max(K_s * (a - b), 0)

def get_ARB_F(K_arb, a_r, b_r, a_l, b_l):
    return K_arb * ((a_r - b_r) - (a_l - b_l))

def get_tire_spring_F(K_t, b, c):
    return K_t * (b - c)

def get_tire_damper_F(C_t, b_d, c_d):
    return C_t * (b_d - c_d)

def get_tire_load(tire_spring_F, tire_damper_F):
    return tire_spring_F + tire_damper_F

def get_init_b(sm, usm, K_t):
    return (sm + usm) * 9.80655 / K_t

def get_init_a(sm, usm, K_s, K_t):
    return sm * 9.80655 / K_s + get_init_b(sm, usm, K_t)

def get_bump_stop_F(K_bs, max_compression, init_a, a, init_b, b):
    return max(K_bs * ((a-init_a) - (b-init_b) - max_compression)  , 0)