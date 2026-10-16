

def test_head_sha_is_found_even_without_the_git_executable(monkeypatch):
    """GPU 컨테이너에는 git이 없어 manifest에 'unknown'이 남았다 — 표 캡션이 그 sha를
    인용하므로 추적선이 끊긴다. .git을 직접 읽는 폴백을 회귀로 지킨다."""
    import os
    import pytest
    from eigen_benchmark.drivers import manifest as mf
    if not os.path.isdir(os.path.join(mf._root(), ".git")):
        pytest.skip("저장소(.git)가 없는 배포본(압축 내려받기·보관본) — .git을 읽는 폴백을 잴 대상이 없다")
    monkeypatch.setattr(mf, "_git", lambda *a: "unknown")
    sha = mf._head_sha()
    assert sha != "unknown" and len(sha) == 40 and all(
        c in "0123456789abcdef" for c in sha), sha


def test_head_sha_falls_back_to_the_launch_environment_without_a_repository(monkeypatch, tmp_path):
    """dgx02는 rsync 사본이라 .git이 없다 — E15-b의 manifest에 unknown이 남는다. 실행한 쪽이
    넘겨준 SHA를 쓰되, 저장소가 있으면 저장소가 이긴다."""
    from eigen_benchmark.drivers import manifest as mf
    monkeypatch.setattr(mf, "_git", lambda *a: "unknown")
    monkeypatch.setattr(mf, "_root", lambda: str(tmp_path))
    monkeypatch.delenv(mf.LAUNCH_SHA_ENV, raising=False)
    assert mf._head_sha() == "unknown"
    monkeypatch.setenv(mf.LAUNCH_SHA_ENV, "a" * 40)
    assert mf._head_sha() == "a" * 40


def test_launch_provenance_does_not_call_an_unknown_tree_dirty(monkeypatch):
    """git 실행파일이 없는 컨테이너에서 status는 'unknown'이고, bool('unknown')은 참이다 —
    모르는 것을 '더럽다'로 적으면 안 된다."""
    from eigen_benchmark.drivers import manifest as mf
    monkeypatch.setattr(mf, "_git", lambda *a: "unknown")
    monkeypatch.delenv(mf.LAUNCH_DIRTY_ENV, raising=False)
    got = mf.launch_provenance()
    assert set(got) == {"git_sha", "git_dirty", "timestamp_utc"}
    assert got["git_dirty"] is None
    monkeypatch.setenv(mf.LAUNCH_DIRTY_ENV, "0")
    assert mf.launch_provenance()["git_dirty"] is False


def test_p5_outputs_have_their_own_root_and_paper2_keeps_its_default(tmp_path):
    """논문4.P5는 2026-10-02에 data/paper2에서 data/paper4로 옮겼다. 논문2 드라이버의 기본값은 그대로다."""
    import os
    from eigen_benchmark.drivers import manifest as mf
    assert mf.P5_OUTDIR.replace(os.sep, "/").endswith("docs/_generated/data/paper4")
    assert mf.ensure_outdir() == os.path.abspath(mf.DEFAULT_OUTDIR)
    assert mf.ensure_outdir(default=str(tmp_path)) == os.path.abspath(str(tmp_path))
    assert mf.ensure_outdir(str(tmp_path / "x"), default=mf.P5_OUTDIR) == str(tmp_path / "x")
