'''
Ivan Pandev, March 2023

This document defines the chassis model used in ChassisDyne, as a system of equations
relating the wheel and body displacements over time to chassis parameters such as
springs, dampers, geometry, mass and inertia, etc.

'''

import numpy as np
import formulas as f
from collections import namedtuple

chassis_state = namedtuple('chassis_state',
    ['a_fr', 'a_fl', 'a_rr', 'a_rl', 'b_fr', 'b_fl', 'b_rr', 'b_rl', 'c_fr', 'c_fl', 'c_rr', 'c_rl',\
    'a_d_fr', 'a_d_fl', 'a_d_rr', 'a_d_rl', 'b_d_fr', 'b_d_fl', 'b_d_rr', 'b_d_rl', 'c_d_fr', 'c_d_fl', 'c_d_rr', 'c_d_rl']
)

def get_dd_matrix(
    self,  # Instance of ChassisDyne's vehicle() class, containing spring constants, damper rates, masses, and inertias
    state,
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

    #  Get instantaneous body inertias
    I_roll_inst_f, I_roll_arm_inst_f = f.get_inst_I_roll_properties(self.I_roll/2, state.a_d_fr, state.a_d_fl, self.tw_f)
    I_roll_inst_r, I_roll_arm_inst_r = f.get_inst_I_roll_properties(self.I_roll/2, state.a_d_rr, state.a_d_rl, self.tw_r)
    I_pitch_inst, I_pitch_arm_inst_f, I_pitch_arm_inst_r = f.get_inst_I_pitch_properties(self.I_pitch, self.wheel_base, self.sm_f)

    #  Load transfers from lat- and long- acceleration
    lat_sm_elastic_LT_f = G_lat * f.LatLT_sm_elastic_1g_axle(self.sm_f*self.sm, self.rc_height_f, self.cm_height, self.tw_f) #TODO: please clean up this notation sm_f*sm
    lat_sm_geo_LT_f = G_lat * f.LatLT_sm_geometric_1g_axle(self.sm_f*self.sm, self.rc_height_f, self.tw_f)
    lat_usm_geo_LT_f = G_lat * f.LatLT_usm_geometric_1g_axle(self.usm_f, self.tire_diam_f, self.tw_f)

    lat_sm_elastic_LT_r = G_lat * f.LatLT_sm_elastic_1g_axle(self.sm_r*self.sm, self.rc_height_r, self.cm_height, self.tw_r)
    lat_sm_geo_LT_r = G_lat * f.LatLT_sm_geometric_1g_axle(self.sm_r*self.sm, self.rc_height_r, self.tw_r)
    lat_usm_geo_LT_r = G_lat * f.LatLT_usm_geometric_1g_axle(self.usm_r, self.tire_diam_r, self.tw_r)

    long_sm_elastic_LT_f = G_long * f.LongLT_sm_elastic_1g_end(self.sm, self.sm_r, self.anti_dive, self.cm_height, self.wheel_base_f)
    long_sm_geo_LT_f = G_long * f.LongLT_sm_geometric_1g_end(self.sm, self.sm_r, self.anti_dive, self.cm_height, self.wheel_base_f)
    long_usm_geo_LT_f = G_long * f.LongLT_usm_geometric_1g_end(self.usm_f, self.usm_r, self.tire_diam_f, self.tire_diam_r, self.wheel_base_f)

    long_sm_elastic_LT_r = G_long * f.LongLT_sm_elastic_1g_end(self.sm, self.sm_f, self.anti_squat, self.cm_height, self.wheel_base_r)
    long_sm_geo_LT_r = G_long * f.LongLT_sm_geometric_1g_end(self.sm, self.sm_f, self.anti_squat, self.cm_height, self.wheel_base_r)
    long_usm_geo_LT_r = G_long * f.LongLT_usm_geometric_1g_end(self.usm_f, self.usm_r, self.tire_diam_f, self.tire_diam_r, self.wheel_base_r)

    #  Load transfers from springs and dampers
    #TODO: Check K_ch implementation.
    chassis_flex_LT_f = self.K_ch * ((state.a_fr - state.a_fl) - (state.a_rr - state.a_rl)) / (self.tw_f / 2)
    chassis_flex_LT_r = self.K_ch * ((state.a_fr - state.a_fl) - (state.a_rr - state.a_rl)) / (self.tw_r / 2)
    ride_spring_F_fr = self.K_s_f * (state.a_fr - state.b_fr)
    ride_spring_F_fl = self.K_s_f * (state.a_fl - state.b_fl)
    ride_spring_F_rr = self.K_s_r * (state.a_rr - state.b_rr)
    ride_spring_F_rl = self.K_s_r * (state.a_rl - state.b_rl)
    
    ride_damper_F_ideal_fr = f.get_ideal_damper_force(
        C_lsc = self.C_lsc_f, C_hsc = self.C_hsc_f, C_lsr = self.C_lsr_f, C_hsr = self.C_hsr_f, a_d = state.a_d_fr, b_d = state.b_d_fr, knee_c = self.knee_c_f, knee_r = self.knee_r_f
    )
    ride_damper_F_ideal_fl = f.get_ideal_damper_force(
        C_lsc = self.C_lsc_f, C_hsc = self.C_hsc_f, C_lsr = self.C_lsr_f, C_hsr = self.C_hsr_f, a_d = state.a_d_fl, b_d = state.b_d_fl, knee_c = self.knee_c_f, knee_r = self.knee_r_f
    )
    ride_damper_F_ideal_rr = f.get_ideal_damper_force(
        C_lsc = self.C_lsc_r, C_hsc = self.C_hsc_r, C_lsr = self.C_lsr_r, C_hsr = self.C_hsr_r, a_d = state.a_d_rr, b_d = state.b_d_rr, knee_c = self.knee_c_r, knee_r = self.knee_r_r
    )
    ride_damper_F_ideal_rl = f.get_ideal_damper_force(
        C_lsc = self.C_lsc_r, C_hsc = self.C_hsc_r, C_lsr = self.C_lsr_r, C_hsr = self.C_hsr_r, a_d = state.a_d_rl, b_d = state.b_d_rl, knee_c = self.knee_c_r, knee_r = self.knee_r_r
    )

    #  ARB
    #  Heave spring
    #  Heave damper

    tire_spring_F_fr = self.K_t_f * (state.b_fr - state.c_fr)
    tire_spring_F_fl = self.K_t_f * (state.b_fl - state.c_fl)
    tire_spring_F_rr = self.K_t_r * (state.b_rr - state.c_rr)
    tire_spring_F_rl = self.K_t_r * (state.b_rl - state.c_rl)
    tire_damper_F_fr = self.C_t_f * (state.b_d_fr - state.c_d_fr)
    tire_damper_F_fl = self.C_t_f * (state.b_d_fl - state.c_d_fl)
    tire_damper_F_rr = self.C_t_r * (state.b_d_rr - state.c_d_rr)
    tire_damper_F_rl = self.C_t_r * (state.b_d_rl - state.c_d_rl)

    '''
    The first 4 rows of A_mat are adaptations of the load transfers from sprung body inertias.
    The last 4 rows are unsprung body inertias.
    '''

    #TODO: check implementation of vertical mass inertia, eliminates need for dynamic I and I_arms

    A_mat = np.array([
        [( - I_roll_inst_f/(I_roll_arm_inst_f**2) - I_pitch_inst/(4*I_pitch_arm_inst_f**2) - self.sm_f*self.sm/2),\
         ( + I_roll_inst_f/(I_roll_arm_inst_f**2) - I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         0, 0, 0, 0],  # Node a_dd_fr

        [( + I_roll_inst_f/(I_roll_arm_inst_f**2) - I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         ( - I_roll_inst_f/(I_roll_arm_inst_f**2) - I_pitch_inst/(4*I_pitch_arm_inst_f**2) - self.sm_f*self.sm/2),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         0, 0, 0, 0],  # Node a_dd_fl

        [( + I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         ( - I_roll_inst_r/(I_roll_arm_inst_r**2) - I_pitch_inst/(4*I_pitch_arm_inst_r**2) - self.sm_r*self.sm/2),\
         ( + I_roll_inst_r/(I_roll_arm_inst_r**2) - I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         0, 0, 0, 0],  # Node a_dd_rr

        [( + I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         ( + I_roll_inst_r/(I_roll_arm_inst_r**2) - I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         ( - I_roll_inst_r/(I_roll_arm_inst_r**2) - I_pitch_inst/(4*I_pitch_arm_inst_r**2) - self.sm_r*self.sm/2),\
         0, 0, 0, 0],  # Node a_dd_rl

        [0, 0, 0, 0, - self.usm_f, 0, 0, 0],  # Node b_fr
        [0, 0, 0, 0, 0, - self.usm_f, 0, 0],  # Node b_fl
        [0, 0, 0, 0, 0, 0, - self.usm_r, 0],  # Node b_rr
        [0, 0, 0, 0, 0, 0, 0, - self.usm_r]  # Node b_rl
    ])

    B_mat = np.array([
        [ - lat_sm_elastic_LT_f - long_sm_elastic_LT_f + chassis_flex_LT_f + ride_spring_F_fr + ride_damper_F_ideal_fr],
        [ + lat_sm_elastic_LT_f - long_sm_elastic_LT_f - chassis_flex_LT_f + ride_spring_F_fl + ride_damper_F_ideal_fl],
        [ - lat_sm_elastic_LT_r + long_sm_elastic_LT_r - chassis_flex_LT_r + ride_spring_F_rr + ride_damper_F_ideal_rr],
        [ + lat_sm_elastic_LT_r + long_sm_elastic_LT_r + chassis_flex_LT_r + ride_spring_F_rl + ride_damper_F_ideal_rl],
        [ - ride_spring_F_fr - ride_damper_F_ideal_fr - lat_sm_geo_LT_f - lat_usm_geo_LT_f - long_sm_geo_LT_f - long_usm_geo_LT_f + tire_spring_F_fr + tire_damper_F_fr],
        [ - ride_spring_F_fl - ride_damper_F_ideal_fl + lat_sm_geo_LT_f + lat_usm_geo_LT_f - long_sm_geo_LT_f - long_usm_geo_LT_f + tire_spring_F_fl + tire_damper_F_fl],
        [ - ride_spring_F_rr - ride_damper_F_ideal_rr - lat_sm_geo_LT_r - lat_usm_geo_LT_r + long_sm_geo_LT_r + long_usm_geo_LT_r + tire_spring_F_rr + tire_damper_F_rr],
        [ - ride_spring_F_rl - ride_damper_F_ideal_rl + lat_sm_geo_LT_r + lat_usm_geo_LT_r + long_sm_geo_LT_r + long_usm_geo_LT_r + tire_spring_F_rl + tire_damper_F_rl]
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