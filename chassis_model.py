'''
Ivan Pandev, March 2023

This document defines the chassis model used in ChassisDyne, as a system of equations
relating the wheel and body displacements over time to chassis parameters such as
springs, dampers, geometry, mass and inertia, etc.

'''

import numpy as np
import formulas as f

def get_x_matrix(
    a_fr, a_fl, a_rr, a_rl, b_fr, b_fl, b_rr, b_rl, c_fr, c_fl, c_rr, c_rl,  # Node position inputs
    a_d_fr, a_d_fl, a_d_rr, a_d_rl, b_d_fr, b_d_fl, b_d_rr, b_d_rl, c_d_fr, c_d_fl, c_d_rr, c_d_rl,  # Node velocity inputs
    sm, sm_f, sm_r, usm_f, usm_r,  # Masses
    I_roll_inst_f, I_roll_inst_r, I_pitch_inst, I_roll_arm_inst_f, I_roll_arm_inst_r, I_pitch_arm_inst_f, I_pitch_arm_inst_r,  # Inertias, radii of rotation
    tw_f, tw_r, wheel_base_f, wheel_base_r, rc_height_f, rc_height_r, anti_dive, anti_squat, cm_height, tire_diam_f, tire_diam_r,  # Vehicle geometries
    K_ch, K_s_f, K_s_r, K_t_f, K_t_r, C_s_fr, C_s_fl, C_s_rr, C_s_rl, C_t_f, C_t_r,  # Springs and Dampers
    G_lat, G_long  # lateral and longitudinal acceleration in G
) -> np.array:
    
    '''
    This method returns the x_dd = F(t, x, x_d) matrix for use in the RK4 iterator:

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
    lat_sm_elastic_LT_f = G_lat * f.LatLT_sm_elastic_1g_axle(sm_f*sm, rc_height_f, cm_height, tw_f) #TODO: please clean up this notation sm_f*sm
    lat_sm_geo_LT_f = G_lat * f.LatLT_sm_geometric_1g_axle(sm_f*sm, rc_height_f, tw_f)
    lat_usm_geo_LT_f = G_lat * f.LatLT_usm_geometric_1g_axle(usm_f, tire_diam_f, tw_f)

    lat_sm_elastic_LT_r = G_lat * f.LatLT_sm_elastic_1g_axle(sm_r*sm, rc_height_r, cm_height, tw_r)
    lat_sm_geo_LT_r = G_lat * f.LatLT_sm_geometric_1g_axle(sm_r*sm, rc_height_r, tw_r)
    lat_usm_geo_LT_r = G_lat * f.LatLT_usm_geometric_1g_axle(usm_r, tire_diam_r, tw_r)

    long_sm_elastic_LT_f = G_long * f.LongLT_sm_elastic_1g_end(sm, sm_r, anti_dive, cm_height, wheel_base_f)
    long_sm_geo_LT_f = G_long * f.LongLT_sm_geometric_1g_end(sm, sm_r, anti_dive, cm_height, wheel_base_f)
    long_usm_geo_LT_f = G_long * f.LongLT_usm_geometric_1g_end(usm_f, usm_r, tire_diam_f, tire_diam_r, wheel_base_f)

    long_sm_elastic_LT_r = G_long * f.LongLT_sm_elastic_1g_end(sm, sm_f, anti_squat, cm_height, wheel_base_r)
    long_sm_geo_LT_r = G_long * f.LongLT_sm_geometric_1g_end(sm, sm_f, anti_squat, cm_height, wheel_base_r)
    long_usm_geo_LT_r = G_long * f.LongLT_usm_geometric_1g_end(usm_f, usm_r, tire_diam_f, tire_diam_r, wheel_base_r)

    #  Load transfers from springs and dampers
    #TODO: Check K_ch implementation.
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

    #  ARB
    #  Heave spring
    #  Heave damper

    tire_spring_F_fr = K_t_f * (b_fr - c_fr)
    tire_spring_F_fl = K_t_f * (b_fl - c_fl)
    tire_spring_F_rr = K_t_r * (b_rr - c_rr)
    tire_spring_F_rl = K_t_r * (b_rl - c_rl)
    tire_damper_F_fr = C_t_f * (b_d_fr - c_d_fr)
    tire_damper_F_fl = C_t_f * (b_d_fl - c_d_fl)
    tire_damper_F_rr = C_t_r * (b_d_rr - c_d_rr)
    tire_damper_F_rl = C_t_r * (b_d_rl - c_d_rl)

    '''
    The first 4 rows of A_mat are adaptations of the load transfers from sprung body inertias.
    The last 4 rows are unsprung body inertias.
    '''

    #TODO: check implementation of vertical mass inertia, eliminates need for dynamic I and I_arms

    A_mat = np.array([
        [( - I_roll_inst_f/(I_roll_arm_inst_f**2) - I_pitch_inst/(4*I_pitch_arm_inst_f**2) - sm_f*sm/2),\
         ( + I_roll_inst_f/(I_roll_arm_inst_f**2) - I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         0, 0, 0, 0],  # Node a_dd_fr

        [( + I_roll_inst_f/(I_roll_arm_inst_f**2) - I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         ( - I_roll_inst_f/(I_roll_arm_inst_f**2) - I_pitch_inst/(4*I_pitch_arm_inst_f**2) - sm_f*sm/2),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         0, 0, 0, 0],  # Node a_dd_fl

        [( + I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         ( - I_roll_inst_r/(I_roll_arm_inst_r**2) - I_pitch_inst/(4*I_pitch_arm_inst_r**2) - sm_r*sm/2),\
         ( + I_roll_inst_r/(I_roll_arm_inst_r**2) - I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         0, 0, 0, 0],  # Node a_dd_rr

        [( + I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         ( + I_roll_inst_r/(I_roll_arm_inst_r**2) - I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         ( - I_roll_inst_r/(I_roll_arm_inst_r**2) - I_pitch_inst/(4*I_pitch_arm_inst_r**2) - sm_r*sm/2),\
         0, 0, 0, 0],  # Node a_dd_rl

        [0, 0, 0, 0, (- usm_f), 0, 0, 0],  # Node b_fr
        [0, 0, 0, 0, 0, (- usm_f), 0, 0],  # Node b_fl
        [0, 0, 0, 0, 0, 0, (- usm_r), 0],  # Node b_rr
        [0, 0, 0, 0, 0, 0, 0, (- usm_r)]  # Node b_rl
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