'''
Copyright 2024 Ivan Pandev

This document defines the chassis model used in ChassisDyne, as a system of equations
relating the wheel and body displacements over time to chassis parameters such as
springs, dampers, geometry, mass and inertia, etc.

This document is also the most efficient place to define and varibles of interset
(each variable should only be calculated once throughout the program) which will
be plotted later.

Dimensional variables in HyperMu follow these conventions:
- a_fr = displacement of the chassis at the front-right corner, taken from the (tire and springs') free-spring length, positive downward
- a_d_fr = first time-derivative of the displacement of the chassis at the front-right corner, taken from the (tire and springs') free-spring length, positive downward
- a_dd_fr = second time-derivative of the displacement of the chassis at the front-right corner, taken from the (tire and springs') free-spring length, positive downward
- b_rl = displacement of the wheel at the right-left corner, taken from the (tire's) free-spring length, positive downward
- b_d_rl = first time-derivative of the displacement of the wheel at the right-left corner, taken from the (tire's) free-spring length, positive downward
- b_dd_rl = second time-derivative of the displacement of the wheel at the right-left corner, taken from the (tire's) free-spring length, positive downward
- c_fr = road height change at the front-right tire, positive downward
- c_d_fr = first time-derivative of road height at the front-right tire, positive downward

A consequence of these conventions is that no suspension travel ratios will be used in solving the vehicle model.
'''

import numpy as np
import formulas as f
from collections import namedtuple

chassis_state = namedtuple('chassis_state',
    ['a_fr', 'a_fl', 'a_rr', 'a_rl',
     'b_fr', 'b_fl', 'b_rr', 'b_rl',
     'c_fr', 'c_fl', 'c_rr', 'c_rl',
     'a_d_fr', 'a_d_fl', 'a_d_rr', 'a_d_rl',
     'b_d_fr', 'b_d_fl', 'b_d_rr', 'b_d_rl',
     'c_d_fr', 'c_d_fl', 'c_d_rr', 'c_d_rl']
)

state_for_plotting = namedtuple('variables_of_interest',
    ['a_fr', 'a_fl', 'a_rr', 'a_rl',
     'b_fr', 'b_fl', 'b_rr', 'b_rl',
     'a_dd_rear_axle',
     'spring_disp_fr', 'spring_disp_fl', 'spring_disp_rl', 'spring_disp_rr',
     'damper_disp_fr', 'damper_disp_fl', 'damper_disp_rl', 'damper_disp_rr',
     'tire_load_fr', 'tire_load_fl', 'tire_load_rr', 'tire_load_rl',
     'damper_vel_fr', 'damper_vel_fl', 'damper_vel_rr', 'damper_vel_rl',
     'damper_force_fr', 'damper_force_fl', 'damper_force_rr', 'damper_force_rl',
     'bump_stop_F_fr', 'bump_stop_F_fl', 'bump_stop_F_rr', 'bump_stop_F_rl', 
     'roll_angle_f', 'roll_angle_r', 'pitch_angle',
     'roll_angle_rate_f', 'roll_angle_rate_r', 'pitch_angle_rate',
     'lateral_load_dist_f', 'lateral_load_dist_r', 'lateral_load_dist_ratio']
)

def enforce_droop_limit(a, a_d, b, b_d, max_droop):
    if a - b < - max_droop:
        b = a + max_droop
        b_d = a_d
    return b, b_d

def enforce_compression_limit():
    return None

def enforce_sidewall_compression_limit():
    return None

def enforce_total_body_compression_limit():
    return None

def enforce_dimensional_boundary_conditions(self, state):

    '''
    Enforce maximum droop condition.
    Hard limits here will force appropriate changes in forces and accelerations downstream.
    Max_droop values are distance between a, b taken from unloaded condition.
    The function enforces b stays permissible distance from a.
    '''

    constrained_b_fr, constrained_b_d_fr = enforce_droop_limit(state.a_fr, state.a_d_fr, state.b_fr, state.b_d_fr, self.max_droop_f)
    constrained_b_fl, constrained_b_d_fl = enforce_droop_limit(state.a_fl, state.a_d_fl, state.b_fl, state.b_d_fl, self.max_droop_f)
    constrained_b_rr, constrained_b_d_rr = enforce_droop_limit(state.a_rr, state.a_d_rr, state.b_rr, state.b_d_rr, self.max_droop_r)
    constrained_b_rl, constrained_b_d_rl = enforce_droop_limit(state.a_rl, state.a_d_rl, state.b_rl, state.b_d_rl, self.max_droop_r)

    # TODO: Enforce compression and sidewall compression limits.

    constrained_state = chassis_state(
        a_fr = state.a_fr, a_fl = state.a_fl, a_rr = state.a_rr, a_rl = state.a_rl,
        b_fr = constrained_b_fr, b_fl = constrained_b_fl, b_rr = constrained_b_rr, b_rl = constrained_b_rl,
        c_fr = state.c_fr, c_fl = state.c_fl, c_rr = state.c_rr, c_rl = state.c_rl,
        a_d_fr = state.a_d_fr, a_d_fl = state.a_d_fl, a_d_rr = state.a_d_rr, a_d_rl = state.a_d_rl,
        b_d_fr = constrained_b_d_fr, b_d_fl = constrained_b_d_fl, b_d_rr = constrained_b_d_rr, b_d_rl = constrained_b_d_rl,
        c_d_fr = state.c_d_fr, c_d_fl = state.c_d_fl, c_d_rr = state.c_d_rr, c_d_rl = state.c_d_rl,
    )

    return constrained_state

def solve_chassis_model(
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

    state = enforce_dimensional_boundary_conditions(self, state)

    #  Get instantaneous body inertias
    I_roll_inst_f, I_roll_arm_inst_f = f.get_inst_I_roll_properties(self.I_roll/2, self.tw_f)
    I_roll_inst_r, I_roll_arm_inst_r = f.get_inst_I_roll_properties(self.I_roll/2, self.tw_r)
    I_pitch_inst, I_pitch_arm_inst_f, I_pitch_arm_inst_r = f.get_inst_I_pitch_properties(self.I_pitch, self.wheel_base, self.sm_f)

    #  Load transfers from lat- and long- acceleration
    lat_sm_elastic_LT_f = G_lat * f.LatLT_sm_elastic_1g_axle(self.sm_f*self.sm, self.rc_height_f, self.cm_height, self.tw_f) #TODO: please clean up this notation sm_f*sm
    lat_sm_geo_LT_f = G_lat * f.LatLT_sm_geometric_1g_axle(self.sm_f*self.sm, self.rc_height_f, self.tw_f)
    lat_usm_geo_LT_f = G_lat * f.LatLT_usm_geometric_1g_axle(self.usm_f, self.tire_diam_f, self.tw_f)

    lat_sm_elastic_LT_r = G_lat * f.LatLT_sm_elastic_1g_axle(self.sm_r*self.sm, self.rc_height_r, self.cm_height, self.tw_r)
    lat_sm_geo_LT_r = G_lat * f.LatLT_sm_geometric_1g_axle(self.sm_r*self.sm, self.rc_height_r, self.tw_r)
    lat_usm_geo_LT_r = G_lat * f.LatLT_usm_geometric_1g_axle(self.usm_r, self.tire_diam_r, self.tw_r)

    # TODO: Need review, top priprity
    long_sm_elastic_LT = G_long * f.LongLT_sm_elastic_1g_v2(G_long, self.sm, self.anti_dive, self.anti_squat, self.cm_height, self.wheel_base, self.tire_diam_r)
    long_sm_geo_LT = G_long * f.LongLT_sm_geometric_1g_v2(G_long, self.sm, self.anti_dive, self.anti_squat, self.cm_height, self.wheel_base, self.tire_diam_r)
    long_usm_geo_LT = G_long * f.LongLT_usm_geometric_1g(self.usm_f, self.usm_r, self.tire_diam_f, self.tire_diam_r, self.wheel_base_f)

    #  Load transfers from springs and dampers
    #TODO: Check K_ch implementation.
    chassis_flex_LT_f = f.get_chassis_flex_LT(self.K_ch, state.a_fr, state.a_fl, state.a_rr, state.a_rl, self.tw_f)
    chassis_flex_LT_r = f.get_chassis_flex_LT(self.K_ch, state.a_fr, state.a_fl, state.a_rr, state.a_rl, self.tw_r)

    bump_stop_F_fr = f.get_bump_stop_F(self.K_bs_f, self.max_compression_f, self.init_a_fr, state.a_fr, self.init_b_fr, state.b_fr)
    bump_stop_F_fl = f.get_bump_stop_F(self.K_bs_f, self.max_compression_f, self.init_a_fl, state.a_fl, self.init_b_fl, state.b_fl)
    bump_stop_F_rr = f.get_bump_stop_F(self.K_bs_r, self.max_compression_r, self.init_a_rr, state.a_rr, self.init_b_rr, state.b_rr)
    bump_stop_F_rl = f.get_bump_stop_F(self.K_bs_r, self.max_compression_r, self.init_a_rl, state.a_rl, self.init_b_rl, state.b_rl)

    ride_spring_F_fr = f.get_ride_spring_F(self.K_s_f, state.a_fr, state.b_fr) + bump_stop_F_fr
    ride_spring_F_fl = f.get_ride_spring_F(self.K_s_f, state.a_fl, state.b_fl) + bump_stop_F_fl
    ride_spring_F_rr = f.get_ride_spring_F(self.K_s_r, state.a_rr, state.b_rr) + bump_stop_F_rr
    ride_spring_F_rl = f.get_ride_spring_F(self.K_s_r, state.a_rl, state.b_rl) + bump_stop_F_rl

    ARB_F_f = f.get_ARB_F(self.K_arb_f, state.a_fr, state.b_fr, state.a_fl, state.b_fl)
    ARB_F_r = f.get_ARB_F(self.K_arb_r, state.a_rr, state.b_rr, state.a_rl, state.b_rl)

    ride_damper_F_ideal_fr = f.get_ideal_damper_force(C_lsc = self.C_lsc_f, C_hsc = self.C_hsc_f, C_lsr = self.C_lsr_f, C_hsr = self.C_hsr_f, a_d = state.a_d_fr, b_d = state.b_d_fr, knee_c = self.knee_c_f, knee_r = self.knee_r_f)
    ride_damper_F_ideal_fl = f.get_ideal_damper_force(C_lsc = self.C_lsc_f, C_hsc = self.C_hsc_f, C_lsr = self.C_lsr_f, C_hsr = self.C_hsr_f, a_d = state.a_d_fl, b_d = state.b_d_fl, knee_c = self.knee_c_f, knee_r = self.knee_r_f)
    ride_damper_F_ideal_rr = f.get_ideal_damper_force(C_lsc = self.C_lsc_r, C_hsc = self.C_hsc_r, C_lsr = self.C_lsr_r, C_hsr = self.C_hsr_r, a_d = state.a_d_rr, b_d = state.b_d_rr, knee_c = self.knee_c_r, knee_r = self.knee_r_r)
    ride_damper_F_ideal_rl = f.get_ideal_damper_force(C_lsc = self.C_lsc_r, C_hsc = self.C_hsc_r, C_lsr = self.C_lsr_r, C_hsr = self.C_hsr_r, a_d = state.a_d_rl, b_d = state.b_d_rl, knee_c = self.knee_c_r, knee_r = self.knee_r_r)

    tire_spring_F_fr = f.get_tire_spring_F(self.K_t_f, state.b_fr, state.c_fr)
    tire_spring_F_fl = f.get_tire_spring_F(self.K_t_f, state.b_fl, state.c_fl)
    tire_spring_F_rr = f.get_tire_spring_F(self.K_t_r, state.b_rr, state.c_rr)
    tire_spring_F_rl = f.get_tire_spring_F(self.K_t_r, state.b_rl, state.c_rl)
    tire_damper_F_fr = f.get_tire_damper_F(self.C_t_f, state.b_d_fr, state.c_d_fr)
    tire_damper_F_fl = f.get_tire_damper_F(self.C_t_f, state.b_d_fl, state.c_d_fl)
    tire_damper_F_rr = f.get_tire_damper_F(self.C_t_r, state.b_d_rr, state.c_d_rr)
    tire_damper_F_rl = f.get_tire_damper_F(self.C_t_r, state.b_d_rl, state.c_d_rl)

    Hy=0
    Hy_fr = f.get_hysteresis_coef(Hy, a_d=state.a_d_fr, b_d=state.b_d_fr)
    Hy_fl = f.get_hysteresis_coef(Hy, a_d=state.a_d_fl, b_d=state.b_d_fl)
    Hy_rr = f.get_hysteresis_coef(Hy, a_d=state.a_d_rr, b_d=state.b_d_rr)
    Hy_rl = f.get_hysteresis_coef(Hy, a_d=state.a_d_rl, b_d=state.b_d_rl)

    '''
    The first 4 rows of A_mat are adaptations of the load transfers from sprung body inertias.
    The last 4 rows are unsprung body inertias.
    '''

    #TODO: check implementation of vertical mass inertia, eliminates need for dynamic I and I_arms

    A_mat = np.array([
        [( - I_roll_inst_f/(I_roll_arm_inst_f**2) - I_pitch_inst/(4*I_pitch_arm_inst_f**2) - self.sm_f*self.sm/2) - Hy_fr,\
         ( + I_roll_inst_f/(I_roll_arm_inst_f**2) - I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         Hy_fr, 0, 0, 0],  # sprung x_dd-dependent forces, front-right

        [( + I_roll_inst_f/(I_roll_arm_inst_f**2) - I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         ( - I_roll_inst_f/(I_roll_arm_inst_f**2) - I_pitch_inst/(4*I_pitch_arm_inst_f**2) - self.sm_f*self.sm/2) - Hy_fl,\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_f**2)),\
         0, Hy_fl, 0, 0],  # sprung x_dd-dependent forces, front-left

        [( + I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         ( - I_roll_inst_r/(I_roll_arm_inst_r**2) - I_pitch_inst/(4*I_pitch_arm_inst_r**2) - self.sm_r*self.sm/2) - Hy_rr,\
         ( + I_roll_inst_r/(I_roll_arm_inst_r**2) - I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         0, 0, Hy_rr, 0],  # sprung x_dd-dependent forces, rear-right

        [( + I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         ( + I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         ( + I_roll_inst_r/(I_roll_arm_inst_r**2) - I_pitch_inst/(4*I_pitch_arm_inst_r**2)),\
         ( - I_roll_inst_r/(I_roll_arm_inst_r**2) - I_pitch_inst/(4*I_pitch_arm_inst_r**2) - self.sm_r*self.sm/2) - Hy_rl,\
         0, 0, 0, Hy_rl],  # sprung x_dd-dependent forces, rear-left

        [Hy_fr, 0, 0, 0, - self.usm_f - Hy_fr, 0, 0, 0],  # unsprung x_dd-dependent forces, front-right
        [0, Hy_fl, 0, 0, 0, - self.usm_f - Hy_fl, 0, 0],  # unsprung x_dd-dependent forces, front-left
        [0, 0, Hy_rr, 0, 0, 0, - self.usm_r - Hy_rr, 0],  # unsprung x_dd-dependent forces, rear-right
        [0, 0, 0, Hy_rl, 0, 0, 0, - self.usm_r - Hy_rl]  # unsprung x_dd-dependent forces, rear-left
    ])

    #TODO: unsprung mass must be considered in last 4 rows, check to make sure it is/isn't and correct. 
    B_mat = np.array([
        [ - lat_sm_elastic_LT_f - long_sm_elastic_LT + chassis_flex_LT_f + ride_spring_F_fr + ARB_F_f + ride_damper_F_ideal_fr - self.sm_fr*9.80655],
        [ + lat_sm_elastic_LT_f - long_sm_elastic_LT - chassis_flex_LT_f + ride_spring_F_fl - ARB_F_f + ride_damper_F_ideal_fl - self.sm_fl*9.80655],
        [ - lat_sm_elastic_LT_r + long_sm_elastic_LT - chassis_flex_LT_r + ride_spring_F_rr + ARB_F_r + ride_damper_F_ideal_rr - self.sm_rr*9.80655],
        [ + lat_sm_elastic_LT_r + long_sm_elastic_LT + chassis_flex_LT_r + ride_spring_F_rl - ARB_F_r + ride_damper_F_ideal_rl - self.sm_rl*9.80655],
        [ - ride_spring_F_fr - ARB_F_f - ride_damper_F_ideal_fr - lat_sm_geo_LT_f - lat_usm_geo_LT_f - long_sm_geo_LT - long_usm_geo_LT + tire_spring_F_fr + tire_damper_F_fr - self.usm_fr*9.80655],
        [ - ride_spring_F_fl + ARB_F_f - ride_damper_F_ideal_fl + lat_sm_geo_LT_f + lat_usm_geo_LT_f - long_sm_geo_LT - long_usm_geo_LT + tire_spring_F_fl + tire_damper_F_fl - self.usm_fl*9.80655],
        [ - ride_spring_F_rr - ARB_F_r - ride_damper_F_ideal_rr - lat_sm_geo_LT_r - lat_usm_geo_LT_r + long_sm_geo_LT + long_usm_geo_LT + tire_spring_F_rr + tire_damper_F_rr - self.usm_rr*9.80655],
        [ - ride_spring_F_rl + ARB_F_r - ride_damper_F_ideal_rl + lat_sm_geo_LT_r + lat_usm_geo_LT_r + long_sm_geo_LT + long_usm_geo_LT + tire_spring_F_rl + tire_damper_F_rl - self.usm_rl*9.80655]
    ])

    #  Solve for accelerations of all 8 bodies
    body_accelerations = np.matmul(np.linalg.inv(A_mat), B_mat)

    #  Capture the variables of interest which will be gathered in time-series and plotted in vehicle.py
    state_for_plotting_return = state_for_plotting(
        a_fr=state.a_fr,
        a_fl=state.a_fl,
        a_rr=state.a_rr,
        a_rl=state.a_rl,
        b_fr=state.b_fr,
        b_fl=state.b_fl,
        b_rr=state.b_rr,
        b_rl=state.b_rl,
        a_dd_rear_axle = float((body_accelerations[2] + body_accelerations[3])/2),
        # TODO: Spring forces calculated here need to have gas chamber, bump stop, and spring forces separated before motion ratios applied.
        spring_disp_fr = f.get_spring_disp(a = state.a_fr, b = state.b_fr, WS_motion_ratio = self.WS_motion_ratio_f),
        spring_disp_fl = f.get_spring_disp(a = state.a_fl, b = state.b_fl, WS_motion_ratio = self.WS_motion_ratio_f),
        spring_disp_rr = f.get_spring_disp(a = state.a_rr, b = state.b_rr, WS_motion_ratio = self.WS_motion_ratio_r),
        spring_disp_rl = f.get_spring_disp(a = state.a_rl, b = state.b_rl, WS_motion_ratio = self.WS_motion_ratio_r),
        damper_disp_fr = f.get_damper_disp(a = state.a_fr, b = state.b_fr, WD_motion_ratio = self.WD_motion_ratio_f),
        damper_disp_fl = f.get_damper_disp(a = state.a_fl, b = state.b_fl, WD_motion_ratio = self.WD_motion_ratio_f),
        damper_disp_rr = f.get_damper_disp(a = state.a_rr, b = state.b_rr, WD_motion_ratio = self.WD_motion_ratio_r),
        damper_disp_rl = f.get_damper_disp(a = state.a_rl, b = state.b_rl, WD_motion_ratio = self.WD_motion_ratio_r),
        tire_load_fr = f.get_tire_load(tire_spring_F_fr, tire_damper_F_fr),
        tire_load_fl = f.get_tire_load(tire_spring_F_fl, tire_damper_F_fl),
        tire_load_rr = f.get_tire_load(tire_spring_F_rr, tire_damper_F_rr),
        tire_load_rl = f.get_tire_load(tire_spring_F_rl, tire_damper_F_rl),
        damper_vel_fr = f.get_damper_vel(a_d = state.a_d_fr, b_d = state.b_d_fr, WD_motion_ratio = self.WD_motion_ratio_f),
        damper_vel_fl = f.get_damper_vel(a_d = state.a_d_fl, b_d = state.b_d_fl, WD_motion_ratio = self.WD_motion_ratio_f),
        damper_vel_rr = f.get_damper_vel(a_d = state.a_d_rr, b_d = state.b_d_rr, WD_motion_ratio = self.WD_motion_ratio_r),
        damper_vel_rl = f.get_damper_vel(a_d = state.a_d_rl, b_d = state.b_d_rl, WD_motion_ratio = self.WD_motion_ratio_r),
        damper_force_fr = f.get_damper_force(ride_damper_F_ideal = ride_damper_F_ideal_fr, WD_motion_ratio = self.WD_motion_ratio_f)
                        + f.get_hysteresis_force(Hy=Hy, a_d=state.a_d_fr, b_d=state.b_d_fr, a_dd=body_accelerations[0], b_dd=body_accelerations[4]),
        damper_force_fl = f.get_damper_force(ride_damper_F_ideal = ride_damper_F_ideal_fl, WD_motion_ratio = self.WD_motion_ratio_f)
                        + f.get_hysteresis_force(Hy=Hy, a_d=state.a_d_fl, b_d=state.b_d_fl,  a_dd=body_accelerations[1], b_dd=body_accelerations[5]),
        damper_force_rr = f.get_damper_force(ride_damper_F_ideal = ride_damper_F_ideal_rr, WD_motion_ratio = self.WD_motion_ratio_r)
                        + f.get_hysteresis_force(Hy=Hy, a_d=state.a_d_rr, b_d=state.b_d_rr,  a_dd=body_accelerations[2], b_dd=body_accelerations[6]),
        damper_force_rl = f.get_damper_force(ride_damper_F_ideal = ride_damper_F_ideal_rl, WD_motion_ratio = self.WD_motion_ratio_r)
                        + f.get_hysteresis_force(Hy=Hy, a_d=state.a_d_rl, b_d=state.b_d_rl,  a_dd=body_accelerations[3], b_dd=body_accelerations[7]),
        bump_stop_F_fr = bump_stop_F_fr,  # TODO: We don't need to apply WD/WS motion ratios to this?
        bump_stop_F_fl = bump_stop_F_fl,
        bump_stop_F_rr = bump_stop_F_rr,
        bump_stop_F_rl = bump_stop_F_rl,
        roll_angle_f = f.get_roll_angle_deg_per_axle(a_r = state.a_fr, a_l = state.a_fl),
        roll_angle_r = f.get_roll_angle_deg_per_axle(a_r = state.a_rr, a_l = state.a_rl),
        pitch_angle = f.get_pitch_angle_deg(a_fr = state.a_fr, a_fl = state.a_fl, a_rr = state.a_rr, a_rl = state.a_rl),
        roll_angle_rate_f = f.get_roll_angle_rate_deg_per_axle(a_r_d = state.a_d_fr, a_l_d = state.a_d_fl),
        roll_angle_rate_r = f.get_roll_angle_rate_deg_per_axle(a_r_d = state.a_d_rr, a_l_d = state.a_d_rl),
        pitch_angle_rate = f.get_pitch_angle_rate_deg(a_fr_d = state.a_d_fr, a_fl_d = state.a_d_fl, a_rr_d = state.a_d_rr, a_rl_d = state.a_d_rl),
        lateral_load_dist_f = 100 * f.get_lateral_load_dist_axle(tire_load_r = f.get_tire_load(tire_spring_F_fr, tire_damper_F_fr), 
                                                           tire_load_l = f.get_tire_load(tire_spring_F_fl, tire_damper_F_fl)),
        lateral_load_dist_r = 100 * f.get_lateral_load_dist_axle(tire_load_r = f.get_tire_load(tire_spring_F_rr, tire_damper_F_rr),
                                                           tire_load_l = f.get_tire_load(tire_spring_F_rl, tire_damper_F_rl)),
        lateral_load_dist_ratio = 100 * f.get_lateral_load_dist_ratio(
            lateral_load_dist_f = f.get_lateral_load_dist_axle(tire_load_r = f.get_tire_load(tire_spring_F_fr, tire_damper_F_fr), 
                                                                tire_load_l = f.get_tire_load(tire_spring_F_fl, tire_damper_F_fl)),
            lateral_load_dist_r = f.get_lateral_load_dist_axle(tire_load_r = f.get_tire_load(tire_spring_F_rr, tire_damper_F_rr),
                                                                tire_load_l = f.get_tire_load(tire_spring_F_rl, tire_damper_F_rl)))
    )

    return body_accelerations, state_for_plotting_return

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