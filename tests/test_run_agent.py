"""CLI integration: only selected files can reach the model."""

import json

import httpx
import pytest

import run_agent as cli
from tests.test_agent import call, response


def configure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_MODEL", "configured-model")
    for var in ("TYPESAFE_API_KEY", "JEV_API_KEY", "JEV_BASE_URL", "OPENROUTER_API_KEY",
                "JEVCTX_JUDGE"):
        monkeypatch.delenv(var, raising=False)


def test_cli_reads_only_allowlisted_files_and_writes_report(tmp_path, monkeypatch):
    configure(monkeypatch)
    source = tmp_path / "selected.txt"
    source.write_bytes(b"original\r\nbytes\r\n")
    output = tmp_path / "run"
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        assert request.headers["authorization"] == "Bearer test-secret"
        messages = json.loads(request.content)["messages"]
        if count == 1:
            return response([call("a", "read_file", json.dumps({"path": "../.env"})),
                             call("b", "read_file", json.dumps({"path": str(source)}))])
        assert "Tool error" in messages[-2]["content"]
        assert messages[-1]["content"] == "original\r\nbytes\r\n"
        return response()

    client_class = httpx.Client
    monkeypatch.setattr(cli.httpx, "Client", lambda **kw: client_class(
        transport=httpx.MockTransport(handler), **kw))
    assert cli.main(["Read selected file", "--file", str(source), "--output", str(output),
                     "--mode", "off"]) == 0
    text = (output / "run.json").read_text(encoding="utf-8")
    report = json.loads(text)
    assert report["status"] == "completed"
    assert report["metrics"]["host_usage"]["requests"] == 2
    assert report["metrics"]["total_cost_usd"] is None
    assert "test-secret" not in text
    with pytest.raises(SystemExit):
        cli.main(["Read", "--file", str(source), "--output", str(output), "--mode", "off"])
    assert (output / "run.json").read_text(encoding="utf-8") == text


def test_shadow_requires_jev_before_creating_output(tmp_path, monkeypatch):
    configure(monkeypatch)
    output = tmp_path / "run"
    with pytest.raises(SystemExit):
        cli.main(["Read", "--file", "README.md", "--output", str(output)])
    assert not output.exists()


def test_cli_turns_on_the_profiled_gate_and_the_work_area(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from jevctx.testing import FakeJevClient
    from tests.test_profile import by_dimension

    configure(monkeypatch)
    monkeypatch.setenv("TYPESAFE_API_KEY", "jev-secret")
    monkeypatch.setattr(cli, "make_judge", lambda: nullcontext(FakeJevClient(by_dimension)))
    source = tmp_path / "notes.txt"
    source.write_text("line\n" * 400)
    seen = []

    def handler(request):
        seen.append([t["function"]["name"] for t in json.loads(request.content)["tools"]])
        return response()

    client_class = httpx.Client
    monkeypatch.setattr(cli.httpx, "Client", lambda **kw: client_class(
        transport=httpx.MockTransport(handler), **kw))
    output = tmp_path / "run"
    assert cli.main(["Read it", "--file", str(source), "--output", str(output), "--mode", "on",
                     "--profile", "--gate-on", "role:change_site", "--workarea"]) == 0
    report = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert report["metrics"]["workarea"] is not None and "recall" in seen[0]
    with pytest.raises(SystemExit):   # a role gate needs the profile
        cli.main(["Read it", "--file", str(source), "--output", str(tmp_path / "r2"),
                  "--mode", "on", "--gate-on", "role:change_site"])
