'''
Ivan Pandev, March 2023

This document defines the application of the Runge-Kutta 4th Order, 2nd derivative
numerical iteration method to solve the x_dd matrix over time.
'''

import chassis_model as model

def RK4_iterator(
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