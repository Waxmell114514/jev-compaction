from jevctx.supersede import ANY_FILE, Footprint, SupersessionIndex, footprints, note_for


def kinds(tool, args, cwd="/testbed"):
    return {(p.kind, p.key, p.span) for p in footprints(tool, args, cwd=cwd)}


def test_footprints_of_views_writes_runs_and_searches():
    assert kinds("read", {"filePath": "/testbed/a.py", "offset": 10, "limit": 20}) == {
        ("view", "/testbed/a.py", (10.0, 29.0))}
    assert ("view", "/testbed/a.py", (5.0, 9.0)) in kinds("bash", {"command": "sed -n '5,9p' a.py"})
    assert ("view", "/testbed/pkg/a.py", (1.0, float("inf"))) in kinds(
        "bash", {"command": "cd /testbed/pkg && cat a.py"})
    assert ("view", "/testbed/a.py", (1.0, 40.0)) in kinds("bash", {"command": "head -n 40 a.py"})
    assert kinds("edit", {"filePath": "/testbed/a.py", "oldString": "x", "newString": "y"}) == {
        ("write", "/testbed/a.py", None)}
    writes = {k for kind, k, _ in kinds("bash", {"command": "sed -i 's/a/b/' a.py b.py && echo hi > out.txt 2>&1"})
              if kind == "write"}
    assert writes == {"/testbed/a.py", "/testbed/b.py", "/testbed/out.txt"}
    assert ("write", ANY_FILE, None) in kinds("bash", {"command": "git stash"})
    assert not any(k == "write" for k, _, _ in kinds("bash", {"command": "git show X | git apply --check"}))
    # A pipeline is not a plain view, and a heredoc's body is not parsed as shell.
    assert not any(k == "view" for k, _, _ in kinds("bash", {"command": "cat a.py | grep def"}))
    heredoc = kinds("bash", {"command": "python - <<'EOF'\nprint('>' * 3) > 'x'\nEOF"})
    assert {k for k, _, _ in heredoc} == {"run"}
    assert kinds("grep", {"pattern": "def f"}) == kinds("grep", {"pattern": "def f"})


def test_a_rerun_or_a_covering_view_supersedes():
    index = SupersessionIndex(cwd="/testbed")
    index.observe("t1", "bash", {"command": "python -m pytest tests/test_a.py -q"}, turn=1)
    index.observe("r1", "read", {"filePath": "/testbed/a.py", "offset": 100, "limit": 50}, turn=2)
    index.observe("r2", "read", {"filePath": "/testbed/a.py", "offset": 120, "limit": 10}, turn=3)
    assert index.status("r1") is None          # a narrower view does not replace a wider one
    made = index.observe("r3", "bash", {"command": "cat a.py"}, turn=4)
    assert {(r.older, r.kind) for r in made} == {("r1", "superseded"), ("r2", "superseded")}
    made = index.observe("t2", "bash", {"command": "cd /testbed &&  python -m pytest tests/test_a.py -q"},
                         turn=5)
    assert [(r.older, r.newer, r.kind) for r in made] == [("t1", "t2", "superseded")]


def test_a_write_makes_views_of_that_file_stale():
    index = SupersessionIndex(cwd="/testbed")
    index.observe("r1", "read", {"filePath": "/testbed/a.py"}, turn=1)
    index.observe("r2", "read", {"filePath": "/testbed/b.py"}, turn=1)
    index.observe("g1", "grep", {"pattern": "parse"}, turn=1)
    made = index.observe("e1", "edit", {"filePath": "/testbed/a.py", "oldString": "a", "newString": "b"},
                         turn=2)
    assert [(r.older, r.kind) for r in made] == [("r1", "stale")]
    assert "may" in note_for(index.status("r1")) or "out of date" in note_for(index.status("r1"))
    # Reading it again after the edit supersedes the stale copy, and that is what status shows.
    index.observe("r3", "read", {"filePath": "/testbed/a.py"}, turn=3)
    assert index.status("r1").kind == "superseded"
    assert index.status("r3") is None
    # git stash may have changed every file.
    assert {r.older for r in index.observe("s", "bash", {"command": "git stash"}, turn=4)} == {"r2", "r3"}


def test_relations_round_trip():
    index = SupersessionIndex()
    index.observe("a", "bash", {"command": "ls"}, turn=1)
    index.observe("b", "bash", {"command": "ls"}, turn=2)
    saved = [r.to_dict() for rs in index.relations.values() for r in rs]
    restored = SupersessionIndex()
    restored.load(saved)
    assert restored.status("a") == index.status("a")
    assert isinstance(footprints("bash", {"command": "ls"})[0], Footprint)


def test_narrower_views_together_supersede_a_wider_one():
    index = SupersessionIndex(cwd="/testbed")
    index.observe("r1", "read", {"filePath": "/testbed/a.py", "offset": 100, "limit": 50}, turn=1)
    index.observe("s1", "bash", {"command": "sed -n '90,130p' a.py"}, turn=2)
    assert index.status("r1") is None
    made = index.observe("s2", "bash", {"command": "sed -n '131,160p' a.py"}, turn=3)
    assert [(r.older, r.kind) for r in made] == [("r1", "superseded")]


def test_output_filters_and_stash_round_trips():
    index = SupersessionIndex(cwd="/testbed")
    index.observe("r1", "read", {"filePath": "/testbed/a.py"}, turn=1)
    index.observe("t1", "bash", {"command": "python -m pytest tests/t.py 2>&1 | tail -20"}, turn=1)
    # Checking the old code and restoring the change leaves every file as it was.
    made = index.observe("t2", "bash", {"command": "git stash && python -m pytest tests/t.py; git stash pop"},
                         turn=2)
    assert made == []
    assert index.observe("l", "bash", {"command": "git stash list"}, turn=3) == []
    made = index.observe("t3", "bash", {"command": "python -m pytest tests/t.py | grep FAIL"}, turn=4)
    assert [(r.older, r.kind) for r in made] == [("t1", "superseded")]
