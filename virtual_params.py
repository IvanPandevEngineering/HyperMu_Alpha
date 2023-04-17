def virtual_params(
    tw_f, tw_r,
    K_s_f, K_s_r, K_arb_f, K_arb_r,
    C_lsc_f, C_lsc_r,
    C_lsr_f, C_lsr_r,
    C_hsc_f, C_hsc_r,
    C_hsr_f, C_hsr_r
):

    tw_v = (tw_f + tw_r) / 2

    K_s_f_v = K_s_f * (tw_v/tw_f)**2
    K_s_r_v = K_s_r * (tw_v/tw_r)**2
    K_arb_f_v = K_arb_f * (tw_v/tw_r)**2
    K_arb_r_v = K_arb_r * (tw_v/tw_r)**2

    C_lsc_f_v = C_lsc_f * (tw_v/tw_f)**2
    C_lsc_r_v = C_lsc_r * (tw_v/tw_r)**2

    C_lsr_f_v = C_lsr_f * (tw_v/tw_f)**2
    C_lsr_r_v = C_lsr_r * (tw_v/tw_r)**2

    C_hsc_f_v = C_hsc_f * (tw_v/tw_f)**2
    C_hsc_r_v = C_hsc_r * (tw_v/tw_r)**2

    C_hsr_f_v = C_hsr_f * (tw_v/tw_f)**2
    C_hsr_r_v = C_hsr_r * (tw_v/tw_r)**2

    return (
        tw_v, K_s_f_v, K_s_r_v,
        K_arb_f_v, K_arb_r_v,
        C_lsc_f_v, C_lsc_r_v,
        C_lsr_f_v, C_lsr_r_v,
        C_hsc_f_v, C_hsc_r_v,
        C_hsr_f_v, C_hsr_r_v
    )