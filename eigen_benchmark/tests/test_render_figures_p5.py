"""논문4.P5의 그림 시험 — 2026-10-02 test_render_figures.py에서 떼어 냈다(P5 공개본이 이 파일만
담을 수 있도록). 그림은 data/paper4의 커밋된 산출물에서만 그린다."""
import os

import pytest

pytest.importorskip("matplotlib")
from eigen_benchmark.render import figures_p5      # noqa: E402

DATA_P5 = "docs/_generated/data/paper4"


@pytest.mark.parametrize("name", sorted(figures_p5.ALL))
def test_every_p5_figure_writes_a_nonempty_png(tmp_path, name):
    p = figures_p5.ALL[name](DATA_P5, str(tmp_path))
    assert os.path.exists(p) and os.path.getsize(p) > 5000


def test_p5_ladder_threshold_matches_the_driver():
    """그림이 쓰는 해상 문턱은 드라이버가 판정에 쓰는 값과 같아야 한다 — 다르면
    그림과 표가 다른 셀 집합을 그린다."""
    from eigen_benchmark.drivers import run_p5_neural as run
    assert figures_p5.LADDER_RESOLVED == run.LADDER_RESOLVED


def test_p5_atlas_wall_rule_reproduces_the_committed_e13_predictions():
    """그림 1이 테두리로 표시하는 모드는 성분 공유 규칙의 예측이다. E13 두 구조의 예측은
    신경 실행 전에 커밋되었으므로(49460de) 렌더러가 같은 규칙으로 같은 모드를 짚어야 한다."""
    import json
    for tag in ("e13a_thick", "e13b_clamped2"):
        with open(os.path.join(DATA_P5, f"p5_reference_{tag}.json"), encoding="utf-8") as f:
            ref = json.load(f)
        assert figures_p5.predicted_wall(ref["dominant"]) == ref["wall_predicted_at"], tag
    assert figures_p5.predicted_wall(list("zyzzzy")) == 4        # 본실험(관측)


def test_p5_atlas_reads_its_shapes_from_the_committed_artifact_only():
    """렌더러는 FEM을 풀지 않는다 — 문제 모듈을 불러오지 않아야 한다."""
    import inspect
    src = inspect.getsource(figures_p5)
    assert "p5_block" not in src and "solve(" not in src


def test_p5_atlas_adds_the_two_further_structures_with_the_wall_at_mode_four():
    """그림 1 확장(2026-10-09) — L-브래킷과 두 층 구조 행. 두 구조 모두 성분 공유 규칙이 벽을 모드 4로
    짚는다(원고 4.7절, 사전규정 2026-10-04)."""
    import json
    tags = [t for t, *_ in figures_p5.ATLAS]
    assert tags[-2:] == ["e21_lbracket", "e22_bilayer"]
    assert set(figures_p5.ATLAS_EXTRA) == set(tags[-2:])
    for tag in figures_p5.ATLAS_EXTRA:
        with open(os.path.join(DATA_P5, f"p5_mode_atlas_{tag}.json"), encoding="utf-8") as f:
            atlas = json.load(f)
        assert figures_p5.predicted_wall([m["dominant"] for m in atlas["modes"]]) == 4, tag


def test_block_faces_skip_inner_faces_and_keep_the_default_order():
    import numpy as np
    axes = (np.linspace(0, 1, 3), np.linspace(0, 1, 2), np.linspace(0, 1, 2))
    P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    full = figures_p5._block_faces(P)
    assert len(full) == 10
    assert len(figures_p5._block_faces(P, frozenset({"x1"}))) == 9
    assert np.array_equal(figures_p5._block_faces(P, frozenset()), full)
