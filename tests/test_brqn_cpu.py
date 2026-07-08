from adamixture.model.br_qn import optimize_original

from tests.config import CPU_BRQN_EXPECTED_DIR, EXPECTED_ATOL, EXPECTED_RTOL, K
from tests.helpers import assert_matches_expected, canonicalize_svd

MAX_ITER = 10000
TOL = 0.1
Q_HIST = 3
PATIENCE = 3


def test_svd_matches_expected(svd_step) -> None:
    _, U, S, V = svd_step
    U, V = canonicalize_svd(U, V)

    assert U.shape == (8451, K)
    assert S.shape == (K,)
    assert V.shape == (105, K)
    assert_matches_expected(CPU_BRQN_EXPECTED_DIR, "demo_run.svd.U.expected", U, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)
    assert_matches_expected(CPU_BRQN_EXPECTED_DIR, "demo_run.svd.S.expected", S, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)
    assert_matches_expected(CPU_BRQN_EXPECTED_DIR, "demo_run.svd.V.expected", V, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)


def test_als_matches_expected(als_step) -> None:
    P, Q = als_step

    assert P.shape == (8451, K)
    assert Q.shape == (105, K)
    assert_matches_expected(CPU_BRQN_EXPECTED_DIR, "demo_run.als.P.expected", P, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)
    assert_matches_expected(CPU_BRQN_EXPECTED_DIR, "demo_run.als.Q.expected", Q, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)


def test_brqn_optimization_matches_expected(demo_bed_data, als_step) -> None:
    G, N, M = demo_bed_data
    P, Q = als_step

    P_opt, Q_opt = optimize_original(
        G,
        P.copy(),
        Q.copy(),
        MAX_ITER,
        K,
        M,
        N,
        TOL,
        Q_HIST,
        PATIENCE,
    )

    assert P_opt.shape == (8451, K)
    assert Q_opt.shape == (105, K)
    assert_matches_expected(CPU_BRQN_EXPECTED_DIR, "demo_run.brqn.P.expected", P_opt, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)
    assert_matches_expected(CPU_BRQN_EXPECTED_DIR, "demo_run.brqn.Q.expected", Q_opt, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)
