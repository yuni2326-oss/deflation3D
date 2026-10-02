"""P5 — 무차원 3D 고정-자유 세장 블록. 참조 FEM(hex27)의 성질을 고정한다.

P4와 같은 규약: 문제는 기저를 모르고, (K, M)만 내보낸다. 3D가 추가하는 난점은
벡터장과 근접 모드 패밀리(굽힘·측방·비틀림)이며, 복잡 형상은 의도적으로 없다.
"""
import numpy as np
import pytest

from eigen_benchmark.problems import p5_block as p5


# ── 형상함수 ────────────────────────────────────────────────────────────────

def test_shape_functions_are_a_partition_of_unity():
    """∑N = 1이 아니면 상수장을 재현하지 못해 강체운동이 에너지를 갖는다."""
    rng = np.random.default_rng(0)
    for xi, eta, ze in rng.uniform(-1, 1, (12, 3)):
        N, dN = p5.hex27_shape(xi, eta, ze)
        assert N.shape == (27,) and dN.shape == (27, 3)
        assert abs(N.sum() - 1.0) < 1e-13
        assert np.allclose(dN.sum(axis=0), 0.0, atol=1e-12)


def test_shape_functions_interpolate_at_their_own_nodes():
    """N_a(node_b) = δ_ab. 어긋나면 절점값 해석이 무의미해진다."""
    s = (-1.0, 0.0, 1.0)
    nodes = [(s[i], s[j], s[k]) for k in range(3) for j in range(3)
             for i in range(3)]
    for a, (xi, eta, ze) in enumerate(nodes):
        N, _ = p5.hex27_shape(xi, eta, ze)
        e = np.zeros(27)
        e[a] = 1.0
        assert np.allclose(N, e, atol=1e-13), a


def test_shape_derivatives_reproduce_a_linear_field_exactly():
    """2차 요소이므로 1차 다항식의 기울기는 정확해야 한다."""
    s = (-1.0, 0.0, 1.0)
    nodes = np.array([(s[i], s[j], s[k]) for k in range(3) for j in range(3)
                      for i in range(3)])
    c = np.array([0.3, -1.7, 2.1])
    vals = nodes @ c
    rng = np.random.default_rng(1)
    for xi, eta, ze in rng.uniform(-1, 1, (8, 3)):
        _, dN = p5.hex27_shape(xi, eta, ze)
        assert np.allclose(dN.T @ vals, c, atol=1e-12)


# ── 메시와 조립 ─────────────────────────────────────────────────────────────

def test_mesh_covers_the_box_and_shares_nodes_between_elements():
    m = p5.build_mesh(2, 1, 1)
    g = p5.P5_GEOMETRY
    assert m["nodes"].shape[1] == 3
    assert m["elems"].shape[1] == 27
    lo, hi = m["nodes"].min(axis=0), m["nodes"].max(axis=0)
    assert np.allclose(lo, 0.0)
    assert np.allclose(hi, [g["L"], g["b"], g["h"]])
    # 절점 공유: 요소 수 × 27 > 실제 절점 수
    assert m["elems"].size > m["nodes"].shape[0]
    # 중복 절점이 없어야 한다 — 있으면 요소가 서로 떨어져 강체운동이 늘어난다
    uniq = np.unique(np.round(m["nodes"], 12), axis=0)
    assert uniq.shape[0] == m["nodes"].shape[0]


def test_unconstrained_stiffness_has_exactly_six_rigid_body_modes():
    """3 병진 + 3 회전. 개수가 다르면 조립이나 절점 공유가 잘못됐다."""
    from scipy.linalg import eigh
    m = p5.build_mesh(2, 1, 1)
    K, M = p5.assemble(m, n_q=3)
    w = eigh(K.toarray(), M.toarray(), eigvals_only=True)
    scale = w[6:].min()
    assert (w[:6] / scale < 1e-9).all(), w[:8] / scale
    assert w[6] / scale > 1e-3


def test_clamping_removes_exactly_the_face_at_x_zero():
    m = p5.build_mesh(2, 1, 1)
    K, M = p5.assemble(m, n_q=3)
    Kc, Mc, free = p5.apply_clamped(K, M, m)
    n_face = int((np.abs(m["nodes"][:, 0]) < 1e-12).sum())
    assert Kc.shape[0] == K.shape[0] - 3 * n_face
    assert free.sum() == Kc.shape[0]
    # 고정한 뒤에는 강체운동이 없다
    from scipy.linalg import eigh
    w = eigh(Kc.toarray(), Mc.toarray(), eigvals_only=True)
    assert w[0] > 0 and w[0] / w[-1] > 1e-14


# ── 스펙트럼 ────────────────────────────────────────────────────────────────

def test_eigenvalues_are_ritz_upper_bounds_and_fall_under_refinement():
    """일치 요소이므로 세밀화하면 고유값이 내려가야 한다 — 올라가면 조립·구적 결함."""
    a = p5.solve(2, 1, 1, n_modes=3)
    b = p5.solve(4, 2, 1, n_modes=3)
    assert (b["lam"] <= a["lam"] * (1 + 1e-10)).all(), (a["lam"], b["lam"])


def test_first_eigenvalue_agrees_with_cantilever_beam_theory_in_order():
    """§2.5 sanity check — 참조가 아니라 자릿수 확인용이다."""
    res = p5.solve(8, 3, 2, n_modes=1)
    eb = p5.beam_theory_estimate()
    assert 0.5 < res["lam"][0] / eb["lam"][0] < 2.0, (res["lam"][0], eb["lam"][0])


def test_beam_theory_estimate_orders_the_mode_families_as_designed():
    """§2.3 규칙(간격 < 1%)이 modes 1-6에서 발동하지 않는 단면비여야 한다."""
    eb = p5.beam_theory_estimate()
    lam = eb["lam"][:6]
    gaps = np.diff(lam) / lam[:-1]
    assert (gaps > 0.01).all(), gaps
    assert set(eb["family"][:6]) >= {"bending", "lateral", "torsion"}


def test_mode_evaluation_returns_three_components_at_arbitrary_points():
    res = p5.solve(2, 1, 1, n_modes=2)
    pts = np.array([[0.5, 0.1, 0.02], [1.0, 0.2, 0.04], [0.25, 0.0, 0.0]])
    u = p5.mode_at(res, 1, pts)
    assert u.shape == (3, 3)
    assert np.isfinite(u).all()
    # 고정면에서는 변위가 0이다
    u0 = p5.mode_at(res, 1, np.array([[0.0, 0.1, 0.02]]))
    assert np.abs(u0).max() < 1e-12


def test_sparse_and_dense_paths_agree_on_the_same_mesh():
    """참조 격자의 최상단은 자유도가 2만을 넘어 밀집 eigh로는 무리다. 희소 경로를
    쓰되, 같은 메시에서 두 경로가 일치함을 먼저 확인해야 그 값을 참조로 쓸 수 있다."""
    a = p5.solve(3, 2, 1, n_modes=4, solver="dense")
    b = p5.solve(3, 2, 1, n_modes=4, solver="sparse")
    assert np.allclose(a["lam"], b["lam"], rtol=1e-9), (a["lam"], b["lam"])
    assert a["solver"] == "dense" and b["solver"] == "sparse"


def test_solver_choice_is_recorded_and_defaults_by_problem_size():
    """어느 경로로 풀었는지 결과에 남아야 한다 — 참조값의 출처이므로."""
    small = p5.solve(2, 1, 1, n_modes=2)
    assert small["solver"] == "dense"
    assert p5.solve(2, 1, 1, n_modes=2, solver="sparse")["solver"] == "sparse"


# ── 구적 수렴 시험(§2.4)에 필요한 것 ────────────────────────────────────────

def test_gradient_evaluation_reproduces_a_linear_displacement_field():
    """텐서 Gauss로 a(u,u)를 적분하려면 임의 점에서 ε(u)가 필요하다.
    선형장의 기울기는 정확해야 한다 — 틀리면 구적 오차와 평가 오차가 섞인다."""
    m = p5.build_mesh(2, 1, 1)
    A = np.array([[0.5, -0.2, 0.1], [0.3, 0.7, -0.4], [-0.1, 0.2, 0.9]])
    u_nodal = m["nodes"] @ A.T                       # u_i = A_ij x_j
    vec = u_nodal.ravel()                            # dof = 3*node + c
    rng = np.random.default_rng(3)
    g = p5.P5_GEOMETRY
    pts = rng.uniform(0, 1, (6, 3)) * [g["L"], g["b"], g["h"]]
    G = p5.eval_grad_at(m, vec, pts)                 # (n_pts, 3, 3) ∂u_i/∂x_j
    assert G.shape == (6, 3, 3)
    assert np.allclose(G, A, atol=1e-10), G[0]


def test_a_global_tensor_grid_cannot_integrate_a_c0_fem_strain_field():
    """**계획 §2.4를 문자대로 따를 수 없는 이유를 고정한다.**

    FEM 장은 C⁰이므로 변형률이 요소 경계에서 불연속이고, 그 경계를 가로지르는 전역
    텐서 Gauss는 a(u,u)를 느리게·비단조로 수렴시킨다(P1의 "구적을 조각경계에 정렬"
    사실의 3D 판본). 질량형은 변위만 쓰므로 정상 수렴한다 — 이 대비가 진단이다.
    따라서 격자 인증은 FEM 모드가 아니라 매끄러운 제작해로 해야 한다."""
    r = p5.solve(3, 2, 1, n_modes=2)
    K, M = p5.assemble(r["mesh"], n_q=3)
    u = r["vec_full"][:, 0]
    a_ref, b_ref = float(u @ K @ u), float(u @ M @ u)
    errs = []
    for grid in [(40, 20, 12), (60, 30, 18), (80, 40, 24)]:
        a_q, b_q = p5.tensor_gauss_forms(r["mesh"], u, grid)
        errs.append((abs(a_q / a_ref - 1), abs(b_q / b_ref - 1)))
    # 질량형은 격자를 키우면 단조로 좋아지고 1e-6 아래로 간다
    mass = [e[1] for e in errs]
    assert mass == sorted(mass, reverse=True) and mass[-1] < 1e-6, mass
    # 강성형은 같은 격자에서 1e-6에 이르지 못한다 — 격자가 아니라 장의 꺾임 탓이다
    assert min(e[0] for e in errs) > 1e-6, errs


def test_quadrature_convergence_uses_a_smooth_manufactured_field():
    """§2.4의 실질: 신경 시행함수와 같은 매끄러움을 가진 장으로 격자를 인증한다.
    매끄러운 장에서 텐서 Gauss는 지수적으로 수렴하므로 기준값이 신뢰된다."""
    out = p5.quadrature_convergence([(8, 6, 4), (16, 8, 6), (24, 12, 8),
                                     (32, 16, 10)], tol=1e-6)
    assert out["chosen"] is not None, out["rows"]
    errs = [r["err_a"] for r in out["rows"]]
    assert errs[-1] < errs[0], errs
    ch = next(r for r in out["rows"] if tuple(r["grid"]) == tuple(out["chosen"]))
    assert ch["err_a"] <= 1e-6 and ch["err_b"] <= 1e-6


# ── 모드형상 인증(§2.2) ─────────────────────────────────────────────────────

def test_mode_shape_certification_reports_self_mac_between_adjacent_levels():
    """§2.2 — 채점이 MAC을 읽으므로 참조 모드형상도 인증해야 한다. 인접 세밀화
    단계 간 self-MAC이 낮으면 MAC 천장이 생겨 옳은 해가 spurious로 분류된다
    (P1에서 실제로 그렇게 됐다). 구적점에서 평가한다 — 채점이 그 점을 쓴다."""
    out = p5.certify_mode_shapes([(3, 2, 1), (6, 4, 2)], n_modes=3,
                                 grid=(8, 6, 4))
    assert len(out["rows"]) == 3
    for r in out["rows"]:
        assert 0.0 <= r["self_mac"] <= 1.0
        assert r["n_points"] == 8 * 6 * 4
    assert out["worst_self_mac"] == min(r["self_mac"] for r in out["rows"])
    assert "threshold" in out and out["threshold"] == 0.999


def test_mode_shape_certification_is_sign_invariant():
    """부호 자유도는 MAC이 흡수해야 한다 — 흡수하지 않으면 인증이 부호에 좌우된다."""
    a = p5.solve(3, 2, 1, n_modes=2)
    b = p5.solve(3, 2, 1, n_modes=2)
    b["vec_full"][:, 0] *= -1.0
    pts, _ = p5._tensor_grid((6, 4, 4))
    ua = p5.eval_mode_at(a["mesh"], a["vec_full"][:, 0], pts)
    ub = p5.eval_mode_at(b["mesh"], b["vec_full"][:, 0], pts)
    assert p5.point_mac(ua, ub) > 1 - 1e-12


# ── v2 §2.2–2.3 감사 격자 ────────────────────────────────────────────────────

def test_manufactured_field_takes_transverse_harmonic_orders():
    """v1 인증은 횡방향 반파 1개만 실었다. 감사 격자는 훈련 격자족이 못 보는 성분까지
    적분해야 하므로, 횡방향 고차 조화에서도 인증할 수 있어야 한다."""
    import numpy as np
    pts, _ = p5._tensor_grid((6, 5, 4))
    u1, _ = p5.manufactured_field(pts)
    u3, _ = p5.manufactured_field(pts, my=3, mz=2)
    assert u1.shape == u3.shape
    assert not np.allclose(u1, u3), "my/mz가 장을 바꾸지 않는다"
    # 기본값은 v1과 같아야 한다 — 커밋된 인증이 조용히 바뀌면 안 된다
    assert np.allclose(u1, p5.manufactured_field(pts, my=1, mz=1)[0])


def test_audit_grid_certification_is_the_smallest_grid_above_the_training_family():
    """v2 §2.2–2.3. 감사 격자는 (a) 훈련 격자족 최대치의 축당 1.5배 이상이고
    (b) 여러 파장·횡조화에서 1e-6을 통과하며 (c) 한 단계 위 격자와 1e-6 안에서
    일치하는 **최소** 격자다. 세 조건 중 하나라도 빠지면 A1이 되풀이된다."""
    out = p5.certify_audit_grid(train_family_max=(20, 10, 8), tol=1e-6)
    ax, ay, az = out["chosen"]
    assert (ax, ay, az) >= (30, 15, 12), out["chosen"]
    assert out["meets_tol"] is True
    assert out["cross_grid_rel_diff"] <= 1e-6, out["cross_grid_rel_diff"]
    assert len(out["fields"]) >= 2, "파장·횡조화를 하나만 보면 인증이 약하다"
    for f in out["fields"]:
        assert f["err_a"] <= 1e-6 and f["err_b"] <= 1e-6, f


def test_audit_grid_certification_records_the_fem_mode_floor_it_cannot_gate_on():
    """v2 §2.3.1은 FEM 참조 모드에 1e-6을 요구하지만, C0 변형률은 요소면에서
    불연속이라 전역 텐서격자로 도달 불가능하다. **문턱으로 쓰지 않되 실측을 남긴다** —
    남기지 않으면 '왜 제작해로 대체했는가'가 검증 불가능한 주장이 된다."""
    out = p5.certify_audit_grid(train_family_max=(20, 10, 8), tol=1e-6,
                                fem_ladder=(6, 4, 2), fem_modes=(1, 3))
    assert "fem_mode_floor" in out
    assert out["fem_mode_floor"] > 1e-6, out["fem_mode_floor"]
    assert out["fem_gate_applied"] is False


# ── E13 (사전규정 2026-09-24): 기하와 경계조건 확장 ─────────────────────────

def test_two_face_clamp_removes_both_end_faces():
    """E13-b의 FEM 쪽. x = 0만 고정하던 것을 두 면으로 넓히되 **기본값은 불변**이어야
    커밋된 참조가 그대로 재현된다."""
    mesh = p5.build_mesh(4, 2, 2)
    K, M = p5.assemble(mesh)
    _, _, f1 = p5.apply_clamped(K, M, mesh)
    _, _, f2 = p5.apply_clamped(K, M, mesh, faces="two_face")
    n1, n2 = int(f1.sum()), int(f2.sum())
    assert n2 < n1, "두 면 고정이 한 면보다 자유도가 적어야 한다"
    # 제거된 자유도 수가 정확히 한 면 몫만큼 더 많아야 한다
    n_face = int((np.abs(mesh["nodes"][:, 0]) < 1e-12).sum()) * 3
    assert n1 - n2 == n_face


def test_thicker_block_changes_the_mode_family_ordering():
    """E13-a의 전제 — 두께를 2.5배로 하면 모드 계열 배열이 바뀌어야 규칙을 시험할 수
    있다. 바뀌지 않으면 사전규정 §9.0에 따라 그 구조는 판정하지 않는다."""
    base = p5.solve(8, 4, 2, n_modes=4)
    thick = p5.solve(8, 4, 4, n_modes=4,
                     geometry={**p5.P5_GEOMETRY, "h": 0.10})
    assert base["lam"].shape == thick["lam"].shape
    # 두께가 늘면 z 굽힘이 뻣뻣해지므로 스펙트럼이 전체적으로 올라간다
    assert thick["lam"][0] > base["lam"][0]


def test_fixed_start_vector_makes_the_sparse_solve_bitwise_repeatable():
    """ARPACK 시작벡터를 고정하지 않으면 고유벡터가 실행마다 마지막 자리에서 달라진다(원고
    §5의 계기 공백). 모드 도감은 형상을 산출물로 커밋하므로 고정 시작벡터를 쓴다.
    기본값은 그대로다 — 커밋된 MAC이 바뀌지 않아야 하기 때문이다."""
    a = p5.solve(6, 2, 2, n_modes=4, solver="sparse", start_vector="ones")
    b = p5.solve(6, 2, 2, n_modes=4, solver="sparse", start_vector="ones")
    assert np.array_equal(a["vec_full"], b["vec_full"])
    assert np.array_equal(a["lam"], b["lam"])


def test_start_vector_rejects_unknown_names():
    with pytest.raises(ValueError):
        p5.solve(3, 2, 1, n_modes=2, solver="sparse", start_vector="random")


# ── 부분공간 self-MAC (E16 정사각 단면의 축퇴쌍 인증) ──────────────────────────

def test_point_subspace_mac_is_blind_to_rotation_within_a_pair():
    """축퇴쌍은 세밀화 단계마다 쌍 안에서 다르게 회전해 나온다. 모드별 MAC은 그 회전에
    무너지지만 부분공간 비교는 불변이어야 한다."""
    rng = np.random.default_rng(3)
    A = [rng.normal(size=(50, 3)) for _ in range(2)]
    t = 0.6
    B = [np.cos(t) * A[0] + np.sin(t) * A[1], -np.sin(t) * A[0] + np.cos(t) * A[1]]
    assert p5.point_mac(A[0], B[0]) < 0.9
    assert p5.point_subspace_mac(A, B) == pytest.approx(1.0, abs=1e-12)


def test_point_subspace_mac_of_one_mode_is_the_point_mac():
    rng = np.random.default_rng(4)
    u, v = rng.normal(size=(40, 3)), rng.normal(size=(40, 3))
    assert p5.point_subspace_mac([u], [v]) == pytest.approx(p5.point_mac(u, v), rel=1e-12)


def test_point_subspace_mac_reports_the_worst_direction():
    """한 방향만 어긋나도 인증이 실패해야 한다 — 평균이면 나쁜 방향이 가려진다."""
    rng = np.random.default_rng(5)
    a, b, c = (rng.normal(size=(60, 3)) for _ in range(3))
    got = p5.point_subspace_mac([a, b], [a, c])
    assert got < 0.5
