'''
Ivan Pandev, March 2023

This document defines the chassis model used in ChassisDyne, as a system of equations
relating the wheel and tire displacements over time to chassis parameters such as
springs, dampers, geometry, mass and inertia, etc.

'''

import numpy as np
import formulas as f

def get_x_matrix_1Dtest(
    m: float,
    F_sm: float, F_usm: float, K_s: float, C_s: float, K_t: float, C_t: float,
    a: float, a_d: float, b: float, b_d: float, c: float, c_d: float
) -> np.array:
    
    '''
    returns the x_dd = F(t, x, x_d) matrix:

    [
    [a_dd]
    [b_dd]
    ]
    '''

    A_mat = np.array([
        [m, 0],
        [0, m]
    ])

    B_mat = np.array([
        [F_sm - C_s*(a_d - b_d) - K_s*(a - b)],
        [F_usm + K_s*(a - b) + C_s*(a_d - b_d) - K_t*(b - c) - C_t*(b_d - c_d)]
    ])

    return np.matmul(np.linalg.inv(A_mat), B_mat)

def get_x_matrix(
    m: float,
    F_sm: float, F_usm: float, K_s: float, C_s: float, K_t: float, C_t: float,
    a: float, a_d: float, b: float, b_d: float, c: float, c_d: float
) -> np.array:
    
    '''
    returns the x_dd = F(t, x, x_d) matrix:

    [
    [a_fr_dd]
    [a_fl_dd]
    [a_rr_dd]
    [a_rl_dd]
    [b_fr_dd]
    [b_fl_dd]
    [b_rr_dd]
    [b_rl_dd]
    ]
    '''

    #  Load transfers from lat- and long- acceleration
    lat_sm_elastic_LT_f = f.LatLT_sm_elastic_1g_axle(sm_f, rc_height_f, cm_height, tw_f)
    lat_sm_geo_LT_f = f.LatLT_sm_geometric_1g_axle(sm_f, rc_height_f, tw_f)
    lat_usm_geo_LT_f = f.LatLT_usm_geometric_1g_axle(usm_f, tire_diam_f, tw_f)

    lat_sm_elastic_LT_r = f.LatLT_sm_elastic_1g_axle(sm_r, rc_height_r, cm_height, tw_r)
    lat_sm_geo_LT_r = f.LatLT_sm_geometric_1g_axle(sm_r, rc_height_r, tw_r)
    lat_usm_geo_LT_r = f.LatLT_usm_geometric_1g_axle(usm_r, tire_diam_r, tw_r)

    long_sm_elastic_LT_f = f.LongLT_sm_elastic_1g_end(sm, pc_height, cm_height, pitch_arm_f)
    long_sm_geo_LT_f = f.LongLT_sm_geometric_1g_end(sm, pc_height, pitch_arm_f) 
    long_usm_geo_LT_f = f.LongLT_usm_geometric_1g_end(usm_f, usm_r, tire_diameter_f, tire_diameter_r, pitch_arm_f)

    long_sm_elastic_LT_r = f.LongLT_sm_elastic_1g_end(sm, pc_height, cm_height, pitch_arm_r)
    long_sm_geo_LT_r = f.LongLT_sm_geometric_1g_end(sm, pc_height, pitch_arm_r)
    long_usm_geo_LT_r = f.LongLT_usm_geometric_1g_end(usm_f, usm_r, tire_diameter_f, tire_diameter_r, pitch_arm_r)

    #  Load transfers from body inertias
    # roll_inr_LT_f = I_roll_inst_f * (a_dd_fr - a_dd_fl) / I_roll_arm_inst_f**2
    # roll_inr_LT_r = I_roll_inst_r * (a_dd_rr - a_dd_rl) / I_roll_arm_inst_r**2
    # pitch_inr_LT_f = I_pitch_inst * (a_dd_f - a_dd_r) / I_pitch_arm_inst_f**2
    # pitch_inr_LT_r = I_pitch_inst * (a_dd_f - a_dd_r) / I_pitch_arm_inst_r**2
    # height_delta_LT_f = m_f * (a_dd_fr + a_dd_fl) / 2
    # height_delta_LT_r = m_r * (a_dd_rr + a_dd_rl) / 2

    #  Load transfers from springs and dampers
    chassis_flex_LT_f = K_ch * ((a_fr - a_fl) - (a_rr - a_rl)) / (tw_f / 2)
    chassis_flex_LT_r = K_ch * ((a_fr - a_fl) - (a_rr - a_rl)) / (tw_r / 2)
    ride_spring_F_fr = K_s_f * (a_fr - b_fr)
    ride_spring_F_fl = K_s_f * (a_fl - b_fl)
    ride_spring_F_rr = K_s_r * (a_rr - b_rr)
    ride_spring_F_rl = K_s_r * (a_rl - b_rl)
    ride_damper_F_fr = C_s_fr * (a_d_fr - b_d_fr)
    ride_damper_F_fl = C_s_fl * (a_d_fl - b_d_fl)
    ride_damper_F_rr = C_s_rr * (a_d_rr - b_d_rr)
    ride_damper_F_rl = C_s_rl * (a_d_rl - b_d_rl)

    #ARB
    #Heave spring
    #Heave damper

    tire_spring_F_fr = K_t_f * (b_fr - c_fr)
    tire_spring_F_fl = K_t_f * (b_fl - c_fl)
    tire_spring_F_rr = K_t_r * (b_rr - c_rr)
    tire_spring_F_rl = K_t_r * (b_rl - c_rl)
    tire_damper_F_fr = C_t_f * (b_d_fr - c_d_fr)
    tire_damper_F_fl = C_t_f * (b_d_fl - c_d_fl)
    tire_damper_F_rr = C_t_r * (b_d_rr - c_d_rr)
    tire_damper_F_rl = C_t_r * (b_d_rl - c_d_rl)

    A_mat = np.array([
        [( - I_roll_inst_f/(I_roll_arm_inst_f**2) - I_pitch_inst_f/(4*I_pitch_arm_inst_f**2) - m_f/2), ( + I_roll_inst_f/(I_roll_arm_inst_f**2) - I_pitch_inst_f/(4*I_pitch_arm_inst_f**2) - m_f/2), ( + I_pitch_inst_f/(4*I_pitch_arm_inst_f**2)), ( + I_pitch_inst_f/(4*I_pitch_arm_inst_f**2)), 0, 0, 0, 0],  # Node a_fr
        [( + I_roll_inst_f/(I_roll_arm_inst_f**2) - I_pitch_inst_f/(4*I_pitch_arm_inst_f**2) - m_f/2), ( - I_roll_inst_f/(I_roll_arm_inst_f**2) - I_pitch_inst_f/(4*I_pitch_arm_inst_f**2) - m_f/2), ( + I_pitch_inst_f/(4*I_pitch_arm_inst_f**2)), ( + I_pitch_inst_f/(4*I_pitch_arm_inst_f**2)), 0, 0, 0, 0],  # Node a_fl, compared to a_fr, (-) on roll terms and chassis flex terms, but not pitch terms
        [( - I_roll_inst_r/(I_roll_arm_inst_r**2) + I_pitch_inst_r/(4*I_pitch_arm_inst_r**2) - m_r/2), ( + I_roll_inst_r/(I_roll_arm_inst_r**2) + I_pitch_inst_r/(4*I_pitch_arm_inst_r**2) - m_r/2), ( - I_pitch_inst_r/(4*I_pitch_arm_inst_r**2)), ( - I_pitch_inst_r/(4*I_pitch_arm_inst_r**2)), 0, 0, 0, 0],  # Node a_rr, compared to a_fr, (-) on pitch and chassis flex terms, but not roll terms
        [( + I_roll_inst_r/(I_roll_arm_inst_r**2) + I_pitch_inst_r/(4*I_pitch_arm_inst_r**2) - m_r/2), ( - I_roll_inst_r/(I_roll_arm_inst_r**2) + I_pitch_inst_r/(4*I_pitch_arm_inst_r**2) - m_r/2), ( - I_pitch_inst_r/(4*I_pitch_arm_inst_r**2)), ( - I_pitch_inst_r/(4*I_pitch_arm_inst_r**2)), 0, 0, 0, 0],  # Node a_rl, compared to a_fr, (-) on pitch and roll terms, but not chassis flex termms 
        [0, 0, 0, 0, (- usm_fr), 0, 0, 0],  # Node b_fr
        [0, 0, 0, 0, 0, (- usm_fl), 0, 0],  # Node b_fl
        [0, 0, 0, 0, 0, 0, (- usm_rr), 0],  # Node b_rr
        [0, 0, 0, 0, 0, 0, 0, (- usm_rl)]  # Node b_rl
    ])

    B_mat = np.array([
        [ - lat_sm_elastic_LT_f - long_sm_elastic_LT_f + chassis_flex_LT_f + ride_spring_F_fr + ride_damper_F_fr],
        [ + lat_sm_elastic_LT_f - long_sm_elastic_LT_f - chassis_flex_LT_f + ride_spring_F_fl + ride_damper_F_fl],
        [ - lat_sm_elastic_LT_r + long_sm_elastic_LT_r - chassis_flex_LT_r + ride_spring_F_rr + ride_damper_F_rr],
        [ + lat_sm_elastic_LT_r + long_sm_elastic_LT_r + chassis_flex_LT_r + ride_spring_F_rl + ride_damper_F_rl],
        [ - ride_spring_F_fr - ride_damper_F_fr - lat_sm_geo_LT_f - lat_usm_geo_LT_f - long_sm_geo_LT_f - long_usm_geo_LT_f + tire_spring_F_fr + tire_damper_F_fr],
        [ - ride_spring_F_fl - ride_damper_F_fl + lat_sm_geo_LT_f + lat_usm_geo_LT_f - long_sm_geo_LT_f - long_usm_geo_LT_f + tire_spring_F_fl + tire_damper_F_fl],
        [ - ride_spring_F_rr - ride_damper_F_rr - lat_sm_geo_LT_r - lat_usm_geo_LT_r + long_sm_geo_LT_r + long_usm_geo_LT_r + tire_spring_F_rr + tire_damper_F_rr],
        [ - ride_spring_F_rl - ride_damper_F_rl + lat_sm_geo_LT_r + lat_usm_geo_LT_r + long_sm_geo_LT_r + long_usm_geo_LT_r + tire_spring_F_rl + tire_damper_F_rl]
    ])

    return np.matmul(np.linalg.inv(A_mat), B_mat)

