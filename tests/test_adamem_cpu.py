from adamixture.model.em_adam import optimize_parameters

from tests.config import CPU_ADAMEM_EXPECTED_DIR, EXPECTED_ATOL, EXPECTED_LOGL_ATOL, EXPECTED_RTOL, K
from tests.helpers import UPDATE_EXPECTED, assert_matches_expected, assert_model_close_to_expected

LR = 0.005
BETA1 = 0.80
BETA2 = 0.88
REG_ADAM = 1e-8
MAX_ITER = 10000
CHECK = 5
LR_DECAY = 0.5
MIN_LR = 1e-4
PATIENCE = 3
TOL_ADAM = 0.1


def test_adamem_optimization_matches_expected(demo_bed_data, als_step) -> None:
    G, N, M = demo_bed_data
    P, Q = als_step

    P_opt, Q_opt = optimize_parameters(
        G,
        P.copy(),
        Q.copy(),
        LR,
        BETA1,
        BETA2,
        REG_ADAM,
        MAX_ITER,
        CHECK,
        K,
        M,
        N,
        LR_DECAY,
        MIN_LR,
        PATIENCE,
        TOL_ADAM,
    )

    assert P_opt.shape == (8451, K)
    assert Q_opt.shape == (105, K)
    if UPDATE_EXPECTED:
        assert_matches_expected(CPU_ADAMEM_EXPECTED_DIR, "demo_run.adamem.P.expected", P_opt, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)
        assert_matches_expected(CPU_ADAMEM_EXPECTED_DIR, "demo_run.adamem.Q.expected", Q_opt, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)
    else:
        assert_model_close_to_expected(
            G,
            P_opt,
            Q_opt,
            CPU_ADAMEM_EXPECTED_DIR,
            "demo_run.adamem.P.expected",
            "demo_run.adamem.Q.expected",
            matrix_rtol=EXPECTED_RTOL,
            matrix_atol=EXPECTED_ATOL,
            logl_atol=EXPECTED_LOGL_ATOL,
        )
