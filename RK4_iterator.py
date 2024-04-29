'''
Copyright 2024 Ivan Pandev

This document defines the application of the Runge-Kutta 4th Order, 2nd derivative
numerical iteration method to solve the x_dd matrix over time.
'''

import chassis_model as model
from collections import namedtuple

time_dependent_inputs = namedtuple('time_dependent_inputs',
    ['G_lat', 'G_lat_half_next', 'G_lat_next', 'G_long', 'G_long_half_next', 'G_long_next', 'G_vert', 'G_vert_half_next', 'G_vert_next',
     'c_fr', 'c_fl', 'c_rr', 'c_rl', 'c_d_fr', 'c_d_fl', 'c_d_rr', 'c_d_rl',
     'c_fr_next', 'c_fl_next', 'c_rr_next', 'c_rl_next', 'c_d_fr_next', 'c_d_fl_next', 'c_d_rr_next', 'c_d_rl_next']
)

def RK4_step(
    dt, self, state, inputs_dt
) -> tuple:

    dd_mat = model.solve_chassis_model(
        self = self, state = state,
        G_lat = inputs_dt.G_lat, G_long = inputs_dt.G_long  # lateral and longitudinal acceleration in G
    )[0]

    x1_a_fr = dt * state.a_d_fr
    x1_a_fl = dt * state.a_d_fl
    x1_a_rr = dt * state.a_d_rr
    x1_a_rl = dt * state.a_d_rl
    x1_b_fr = dt * state.b_d_fr
    x1_b_fl = dt * state.b_d_fl
    x1_b_rr = dt * state.b_d_rr
    x1_b_rl = dt * state.b_d_rl

    v1_a_d_fr = dt * dd_mat[0][0]
    v1_a_d_fl = dt * dd_mat[1][0]
    v1_a_d_rr = dt * dd_mat[2][0]
    v1_a_d_rl = dt * dd_mat[3][0]
    v1_b_d_fr = dt * dd_mat[4][0]
    v1_b_d_fl = dt * dd_mat[5][0]
    v1_b_d_rr = dt * dd_mat[6][0]
    v1_b_d_rl = dt * dd_mat[7][0]

    RK_state_1 = model.chassis_state(
        a_fr = state.a_fr + x1_a_fr/2, a_fl = state.a_fl + x1_a_fl/2, a_rr = state.a_rr + x1_a_rr/2, a_rl = state.a_rl + x1_a_rl/2,
        b_fr = state.b_fr + x1_b_fr/2, b_fl = state.b_fl + x1_b_fl/2, b_rr = state.b_rr + x1_b_rr/2, b_rl = state.b_rl + x1_b_rl/2,
        a_d_fr = state.a_d_fr + v1_a_d_fr/2, a_d_fl = state.a_d_fl + v1_a_d_fl/2, a_d_rr = state.a_d_rr + v1_a_d_rr/2, a_d_rl = state.a_d_rl + v1_a_d_rl/2,
        b_d_fr = state.b_d_fr + v1_b_d_fr/2, b_d_fl = state.b_d_fl + v1_b_d_fl/2, b_d_rr = state.b_d_rr + v1_b_d_rr/2, b_d_rl = state.b_d_rl + v1_b_d_rl/2,
        c_fr = (inputs_dt.c_fr+inputs_dt.c_fr_next)/2, c_fl = (inputs_dt.c_fl+inputs_dt.c_fl_next)/2, c_rr = (inputs_dt.c_rr+inputs_dt.c_rr_next)/2, c_rl = (inputs_dt.c_rl+inputs_dt.c_rl_next)/2,
        c_d_fr = (inputs_dt.c_d_fr+inputs_dt.c_d_fr_next)/2, c_d_fl = (inputs_dt.c_d_fl+inputs_dt.c_d_fl_next)/2, c_d_rr = (inputs_dt.c_d_rr+inputs_dt.c_d_rr_next)/2, c_d_rl = (inputs_dt.c_d_rl+inputs_dt.c_d_rl_next)/2
    )

    dd_mat_half_next_1 = model.solve_chassis_model(
        self = self,
        state = RK_state_1,
        G_lat = inputs_dt.G_lat_half_next, G_long = inputs_dt.G_long_half_next  # lateral and longitudinal acceleration in G
    )[0]

    x2_a_fr = dt * (state.a_d_fr + v1_a_d_fr/2)
    x2_a_fl = dt * (state.a_d_fl + v1_a_d_fl/2)
    x2_a_rr = dt * (state.a_d_rr + v1_a_d_rr/2)
    x2_a_rl = dt * (state.a_d_rl + v1_a_d_rl/2)
    x2_b_fr = dt * (state.b_d_fr + v1_b_d_fr/2)
    x2_b_fl = dt * (state.b_d_fl + v1_b_d_fl/2)
    x2_b_rr = dt * (state.b_d_rr + v1_b_d_rr/2)
    x2_b_rl = dt * (state.b_d_rl + v1_b_d_rl/2)

    v2_a_d_fr = dt * dd_mat_half_next_1[0][0]
    v2_a_d_fl = dt * dd_mat_half_next_1[1][0]
    v2_a_d_rr = dt * dd_mat_half_next_1[2][0]
    v2_a_d_rl = dt * dd_mat_half_next_1[3][0]
    v2_b_d_fr = dt * dd_mat_half_next_1[4][0]
    v2_b_d_fl = dt * dd_mat_half_next_1[5][0]
    v2_b_d_rr = dt * dd_mat_half_next_1[6][0]
    v2_b_d_rl = dt * dd_mat_half_next_1[7][0]

    RK_state_2 = model.chassis_state(
        a_fr = state.a_fr + x2_a_fr/2, a_fl = state.a_fl + x2_a_fl/2, a_rr = state.a_rr + x2_a_rr/2, a_rl = state.a_rl + x2_a_rl/2,
        b_fr = state.b_fr + x2_b_fr/2, b_fl = state.b_fl + x2_b_fl/2, b_rr = state.b_rr + x2_b_rr/2, b_rl = state.b_rl + x2_b_rl/2,
        a_d_fr = state.a_d_fr + v2_a_d_fr/2, a_d_fl = state.a_d_fl + v2_a_d_fl/2, a_d_rr = state.a_d_rr + v2_a_d_rr/2, a_d_rl = state.a_d_rl + v2_a_d_rl/2,
        b_d_fr = state.b_d_fr + v2_b_d_fr/2, b_d_fl = state.b_d_fl + v2_b_d_fl/2, b_d_rr = state.b_d_rr + v2_b_d_rr/2, b_d_rl = state.b_d_rl + v2_b_d_rl/2,
        c_fr = (inputs_dt.c_fr+inputs_dt.c_fr_next)/2, c_fl = (inputs_dt.c_fl+inputs_dt.c_fl_next)/2, c_rr = (inputs_dt.c_rr+inputs_dt.c_rr_next)/2, c_rl = (inputs_dt.c_rl+inputs_dt.c_rl_next)/2,
        c_d_fr = (inputs_dt.c_d_fr+inputs_dt.c_d_fr_next)/2, c_d_fl = (inputs_dt.c_d_fl+inputs_dt.c_d_fl_next)/2, c_d_rr = (inputs_dt.c_d_rr+inputs_dt.c_d_rr_next)/2, c_d_rl = (inputs_dt.c_d_rl+inputs_dt.c_d_rl_next)/2
    )
    
    dd_mat_half_next_2 = model.solve_chassis_model(
        self = self,
        state = RK_state_2,
        G_lat = inputs_dt.G_lat_half_next, G_long = inputs_dt.G_long_half_next  # lateral and longitudinal acceleration in G
    )[0]

    x3_a_fr = dt * (state.a_d_fr + v2_a_d_fr/2)
    x3_a_fl = dt * (state.a_d_fl + v2_a_d_fl/2)
    x3_a_rr = dt * (state.a_d_rr + v2_a_d_rr/2)
    x3_a_rl = dt * (state.a_d_rl + v2_a_d_rl/2)
    x3_b_fr = dt * (state.b_d_fr + v2_b_d_fr/2)
    x3_b_fl = dt * (state.b_d_fl + v2_b_d_fl/2)
    x3_b_rr = dt * (state.b_d_rr + v2_b_d_rr/2)
    x3_b_rl = dt * (state.b_d_rl + v2_b_d_rl/2)
    v3_a_d_fr = dt * dd_mat_half_next_2[0][0]
    v3_a_d_fl = dt * dd_mat_half_next_2[1][0]
    v3_a_d_rr = dt * dd_mat_half_next_2[2][0]
    v3_a_d_rl = dt * dd_mat_half_next_2[3][0]
    v3_b_d_fr = dt * dd_mat_half_next_2[4][0]
    v3_b_d_fl = dt * dd_mat_half_next_2[5][0]
    v3_b_d_rr = dt * dd_mat_half_next_2[6][0]
    v3_b_d_rl = dt * dd_mat_half_next_2[7][0]

    RK_state_3 = model.chassis_state(
        a_fr = state.a_fr + x3_a_fr, a_fl = state.a_fl + x3_a_fl, a_rr = state.a_rr + x3_a_rr, a_rl = state.a_rl + x3_a_rl,
        b_fr = state.b_fr + x3_b_fr, b_fl = state.b_fl + x3_b_fl, b_rr = state.b_rr + x3_b_rr, b_rl = state.b_rl + x3_b_rl,
        a_d_fr = state.a_d_fr + v3_a_d_fr, a_d_fl = state.a_d_fl + v3_a_d_fl, a_d_rr = state.a_d_rr + v3_a_d_rr, a_d_rl = state.a_d_rl + v3_a_d_rl,
        b_d_fr = state.b_d_fr + v3_b_d_fr, b_d_fl = state.b_d_fl + v3_b_d_fl, b_d_rr = state.b_d_rr + v3_b_d_rr, b_d_rl = state.b_d_rl + v3_b_d_rl,
        c_fr = inputs_dt.c_fr_next, c_fl = inputs_dt.c_fl_next, c_rr = inputs_dt.c_rr_next, c_rl = inputs_dt.c_rl_next,
        c_d_fr = inputs_dt.c_d_fr_next, c_d_fl = inputs_dt.c_d_fl_next, c_d_rr = inputs_dt.c_d_rr_next, c_d_rl = inputs_dt.c_d_rl_next
    )

    dd_mat_next = model.solve_chassis_model(
        self = self,
        state = RK_state_3,
        G_lat = inputs_dt.G_lat_next, G_long = inputs_dt.G_long_next  # lateral and longitudinal acceleration in G
    )[0]

    x4_a_fr = dt * (state.a_d_fr + v3_a_d_fr)
    x4_a_fl = dt * (state.a_d_fl + v3_a_d_fl)
    x4_a_rr = dt * (state.a_d_rr + v3_a_d_rr)
    x4_a_rl = dt * (state.a_d_rl + v3_a_d_rl)
    x4_b_fr = dt * (state.b_d_fr + v3_b_d_fr)
    x4_b_fl = dt * (state.b_d_fl + v3_b_d_fl)
    x4_b_rr = dt * (state.b_d_rr + v3_b_d_rr)
    x4_b_rl = dt * (state.b_d_rl + v3_b_d_rl)

    v4_a_d_fr = dt * dd_mat_next[0][0]
    v4_a_d_fl = dt * dd_mat_next[1][0]
    v4_a_d_rr = dt * dd_mat_next[2][0]
    v4_a_d_rl = dt * dd_mat_next[3][0]
    v4_b_d_fr = dt * dd_mat_next[4][0]
    v4_b_d_fl = dt * dd_mat_next[5][0]
    v4_b_d_rr = dt * dd_mat_next[6][0]
    v4_b_d_rl = dt * dd_mat_next[7][0]

    a_fr_next = state.a_fr + (x1_a_fr + 2*x2_a_fr + 2*x3_a_fr + x4_a_fr) / 6
    a_fl_next = state.a_fl + (x1_a_fl + 2*x2_a_fl + 2*x3_a_fl + x4_a_fl) / 6
    a_rr_next = state.a_rr + (x1_a_rr + 2*x2_a_rr + 2*x3_a_rr + x4_a_rr) / 6
    a_rl_next = state.a_rl + (x1_a_rl + 2*x2_a_rl + 2*x3_a_rl + x4_a_rl) / 6
    b_fr_next = state.b_fr + (x1_b_fr + 2*x2_b_fr + 2*x3_b_fr + x4_b_fr) / 6
    b_fl_next = state.b_fl + (x1_b_fl + 2*x2_b_fl + 2*x3_b_fl + x4_b_fl) / 6
    b_rr_next = state.b_rr + (x1_b_rr + 2*x2_b_rr + 2*x3_b_rr + x4_b_rr) / 6
    b_rl_next = state.b_rl + (x1_b_rl + 2*x2_b_rl + 2*x3_b_rl + x4_b_rl) / 6

    a_d_fr_next = state.a_d_fr + (v1_a_d_fr + 2*v2_a_d_fr + 2*v3_a_d_fr + v4_a_d_fr) / 6
    a_d_fl_next = state.a_d_fl + (v1_a_d_fl + 2*v2_a_d_fl + 2*v3_a_d_fl + v4_a_d_fl) / 6
    a_d_rr_next = state.a_d_rr + (v1_a_d_rr + 2*v2_a_d_rr + 2*v3_a_d_rr + v4_a_d_rr) / 6
    a_d_rl_next = state.a_d_rl + (v1_a_d_rl + 2*v2_a_d_rl + 2*v3_a_d_rl + v4_a_d_rl) / 6
    b_d_fr_next = state.b_d_fr + (v1_b_d_fr + 2*v2_b_d_fr + 2*v3_b_d_fr + v4_b_d_fr) / 6
    b_d_fl_next = state.b_d_fl + (v1_b_d_fl + 2*v2_b_d_fl + 2*v3_b_d_fl + v4_b_d_fl) / 6
    b_d_rr_next = state.b_d_rr + (v1_b_d_rr + 2*v2_b_d_rr + 2*v3_b_d_rr + v4_b_d_rr) / 6
    b_d_rl_next = state.b_d_rl + (v1_b_d_rl + 2*v2_b_d_rl + 2*v3_b_d_rl + v4_b_d_rl) / 6

    next_state = model.chassis_state(
        a_fr = a_fr_next, a_fl = a_fl_next, a_rr = a_rr_next, a_rl = a_rl_next,
        b_fr = b_fr_next, b_fl = b_fl_next, b_rr = b_rr_next, b_rl = b_rl_next,
        a_d_fr = a_d_fr_next, a_d_fl = a_d_fl_next, a_d_rr = a_d_rr_next, a_d_rl = a_d_rl_next,
        b_d_fr = b_d_fr_next, b_d_fl = b_d_fl_next, b_d_rr = b_d_rr_next, b_d_rl = b_d_rl_next,
        c_fr = inputs_dt.c_fr_next, c_fl = inputs_dt.c_fl_next, c_rr = inputs_dt.c_rr_next, c_rl = inputs_dt.c_rl_next,
        c_d_fr = inputs_dt.c_d_fr_next, c_d_fl = inputs_dt.c_d_fl_next, c_d_rr = inputs_dt.c_d_rr_next, c_d_rl = inputs_dt.c_d_rl_next
    )

    next_variables_of_interest = model.solve_chassis_model(
        self = self,
        state = next_state,
        G_lat = inputs_dt.G_lat_next, G_long = inputs_dt.G_long_next
    )[1]

    return next_state, next_variables_of_interest

def RK4_iterator_1Dtest(
    self, dt,
    a, a_d, b, b_d, c, c_d,
    F_sm, F_usm, F_sm_half_next, F_usm_half_next, F_sm_next, F_usm_next
) -> tuple:

    F_mat = model.get_x_matrix_1Dtest(
        self.m, F_sm, F_usm, self.K_s_f, self.C_lsc_f, self.K_t_f, self.C_t_f, a, a_d, b, b_d, c, c_d
    )
    x1_a = dt * a_d
    v1_a_d = dt * F_mat[0][0]
    x1_b = dt * b_d
    v1_b_d = dt * F_mat[1][0]

    F_mat_half_next_1 = model.get_x_matrix_1Dtest(
        self.m, F_sm_half_next, F_usm_half_next, self.K_s_f, self.C_lsc_f, self.K_t_f, self.C_t_f, a + x1_a/2, a_d + v1_a_d/2, b + x1_b/2, b_d + v1_b_d/2, c, c_d
    )
    x2_a = dt * (a_d + v1_a_d / 2)
    v2_a_d = dt * F_mat_half_next_1[0][0]
    x2_b = dt * (b_d + v1_b_d / 2)
    v2_b_d = dt * F_mat_half_next_1[1][0]
    
    F_mat_half_next_2 = model.get_x_matrix_1Dtest(
        self.m, F_sm_half_next, F_usm_half_next, self.K_s_f, self.C_lsc_f, self.K_t_f, self.C_t_f, a + x2_a/2, a_d + v2_a_d/2, b + x2_b/2, b_d + v2_b_d/2, c, c_d
    )
    x3_a = dt * (a_d + v2_a_d / 2)
    v3_a_d = dt * F_mat_half_next_2[0][0]
    x3_b = dt * (b_d + v2_b_d / 2)
    v3_b_d = dt * F_mat_half_next_2[1][0]

    F_mat_next = model.get_x_matrix_1Dtest(
        self.m, F_sm_next, F_usm_next, self.K_s_f, self.C_lsc_f, self.K_t_f, self.C_t_f, a + x3_a, a_d + v3_a_d, b + x3_b, b_d + v3_b_d, c, c_d
    )
    x4_a = dt * (a_d + v3_a_d)
    v4_a_d = dt * F_mat_next[0][0]
    x4_b = dt * (b_d + v3_b_d)
    v4_b_d = dt * F_mat_next[1][0]

    a_next = a + (x1_a + 2*x2_a + 2*x3_a + x4_a) / 6
    a_d_next = a_d + (v1_a_d + 2*v2_a_d + 2*v3_a_d + v4_a_d) / 6
    b_next = b + (x1_b + 2*x2_b + 2*x3_b + x4_b) / 6
    b_d_next = b_d + (v1_b_d + 2*v2_b_d + 2*v3_b_d + v4_b_d) / 6

    return a_next, a_d_next, b_next, b_d_next