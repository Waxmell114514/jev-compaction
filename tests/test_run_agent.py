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
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)


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
