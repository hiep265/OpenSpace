import json
from pathlib import Path

from openspace.host_integration.kimi_trace_adapter import KimiTraceAdapter
from openspace.host_integration.supervisor import HostTraceSupervisor


def _sample_trace() -> dict:
    return {
        "version": 1,
        "trace_id": "cw_123_default_20260401T102233Z_ab12cd",
        "created_at": "2026-04-01T10:22:33Z",
        "agent_name": "default",
        "conversation_id": "123",
        "session_id": "123_default",
        "source": "chatwoot",
        "user_message": "[CHATWOOT_CONTEXT]\nconversation_id=123\n[/CHATWOOT_CONTEXT]\n\nUSER_MESSAGE:\nXin gia khoa hoc",
        "system_prompt": {
            "path": "/tmp/system.md",
            "sha256": "abc123",
        },
        "skills_runtime": {
            "compiled_dir": "/tmp/compiled/default/current",
            "base_dirs": ["/tmp/skills"],
            "managed_dir": "/tmp/managed/default",
            "selected_skill_ids": ["captured-reply__v0_ab12cd34"],
            "selected_skill_paths": ["/tmp/managed/default/captured-reply/SKILL.md"],
            "used_skill_ids": ["captured-reply__v0_ab12cd34"],
            "used_skill_paths": ["/tmp/managed/default/captured-reply/SKILL.md"],
            "active_skill_paths": [
                "/tmp/skills/faq/SKILL.md",
                "/tmp/managed/default/captured-reply/SKILL.md",
            ],
        },
        "history_snapshot": [
            {"role": "user", "content": "Xin gia khoa hoc"},
            {"role": "assistant", "content": "Ben em gui thong tin"},
        ],
        "execution": {
            "assistant_text": "Da gui thong tin",
            "event_types": ["TurnBegin", "ToolCall", "ToolResult", "TurnEnd"],
            "tool_calls": ["chatwoot_faq_search", "chatwoot_reply"],
            "tool_results": {
                "chatwoot_faq_search": {"ok": True, "items": [{"title": "Bang gia"}]},
                "chatwoot_reply": {"ok": True},
            },
            "wire_file": "/tmp/wire.jsonl",
            "context_file": "/tmp/context.jsonl",
            "token_count": 321,
            "max_context_size": 100000,
            "reserved_context_size": 1000,
        },
        "chatwoot": {
            "account_id": "1",
            "inbox_id": "2",
            "contact_id": "3",
            "labels": ["khach_moi"],
        },
    }


def test_kimi_trace_adapter_writes_analyzer_compatible_recording(tmp_path: Path):
    output_dir = tmp_path / "recording"

    result = KimiTraceAdapter.write_recording(
        trace_payload=_sample_trace(),
        output_dir=output_dir,
    )

    assert result["recording_dir"] == str(output_dir.resolve())
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    conversations = [
        json.loads(line)
        for line in (output_dir / "conversations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    traj = [
        json.loads(line)
        for line in (output_dir / "traj.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert metadata["task_id"] == "cw_123_default_20260401T102233Z_ab12cd"
    assert metadata["task_description"] == "Xin gia khoa hoc"
    assert metadata["execution_outcome"]["status"] == "success"
    assert metadata["kimi_trace"]["agent_name"] == "default"
    assert metadata["skill_selection"]["selected"] == ["captured-reply__v0_ab12cd34"]
    assert "captured-reply__v0_ab12cd34" in metadata["skill_selection"]["available_skills"]

    assert conversations[0]["type"] == "setup"
    assert conversations[0]["agent_name"] == "GroundingAgent"
    assert conversations[0]["messages"][0]["role"] == "system"
    assert "Kimi host trace replay" in conversations[0]["messages"][0]["content"]
    assert conversations[1]["type"] == "iteration"
    assert conversations[1]["delta_messages"][-1]["role"] == "assistant"

    assert len(traj) == 2
    assert traj[0]["backend"] == "mcp"
    assert traj[0]["tool"] == "chatwoot_faq_search"
    assert traj[1]["tool"] == "chatwoot_reply"
    assert traj[1]["result"]["status"] == "success"


def test_host_trace_supervisor_builds_openspace_analysis_context(tmp_path: Path):
    output_dir = tmp_path / "recording"

    result = HostTraceSupervisor().analyze_trace(
        trace_payload=_sample_trace(),
        output_dir=output_dir,
    )

    assert result["recording"]["trace_id"] == "cw_123_default_20260401T102233Z_ab12cd"
    assert result["analysis_context"]["task_description"] == "Xin gia khoa hoc"
    assert result["analysis_context"]["execution_status"] == "success"
    assert result["analysis_context"]["conversation_count"] == 2
    assert "mcp:chatbotlevan:chatwoot_reply" in result["analysis_context"]["used_tool_keys"]
    assert result["analysis_context"]["formatted_conversation"]
    assert result["execution_analysis"]["task_completed"] is True
    assert result["execution_analysis"]["analyzed_by"] == "host_trace_heuristic"
    assert result["execution_analysis"]["evolution_suggestions"][0]["type"] == "captured"


def test_host_trace_supervisor_uses_injected_analysis_runner(tmp_path: Path):
    output_dir = tmp_path / "recording"

    def _runner(*, trace_payload, analysis_context):
        assert trace_payload["trace_id"] == "cw_123_default_20260401T102233Z_ab12cd"
        assert analysis_context["task_description"] == "Xin gia khoa hoc"
        return {
            "task_completed": False,
            "execution_note": "Injected analyzer says this trace should be reviewed.",
            "tool_issues": ["mcp:chatbotlevan:chatwoot_reply"],
            "skill_judgments": [],
            "evolution_suggestions": [],
            "analyzed_by": "injected_runner",
        }

    result = HostTraceSupervisor(analysis_runner=_runner).analyze_trace(
        trace_payload=_sample_trace(),
        output_dir=output_dir,
    )

    assert result["execution_analysis"]["task_completed"] is False
    assert result["execution_analysis"]["execution_note"] == "Injected analyzer says this trace should be reviewed."
    assert result["execution_analysis"]["tool_issues"] == ["mcp:chatbotlevan:chatwoot_reply"]
    assert result["execution_analysis"]["analyzed_by"] == "injected_runner"


def test_host_trace_supervisor_prefers_env_enabled_execution_analyzer_path(
    tmp_path: Path,
    monkeypatch,
):
    output_dir = tmp_path / "recording"
    monkeypatch.setenv("OPENSPACE_HOST_ENABLE_EXECUTION_ANALYZER", "1")

    def _fake_builtin_runner(*, trace_payload, analysis_context, recording_dir):
        assert trace_payload["trace_id"] == "cw_123_default_20260401T102233Z_ab12cd"
        assert analysis_context["task_description"] == "Xin gia khoa hoc"
        assert recording_dir == output_dir
        return {
            "task_completed": False,
            "execution_note": "ExecutionAnalyzer requested manual review.",
            "tool_issues": ["mcp:chatbotlevan:chatwoot_reply"],
            "skill_judgments": [],
            "evolution_suggestions": [],
            "analyzed_by": "host_trace_execution_analyzer",
        }

    monkeypatch.setattr(
        HostTraceSupervisor,
        "_try_run_builtin_execution_analyzer",
        staticmethod(_fake_builtin_runner),
    )

    result = HostTraceSupervisor().analyze_trace(
        trace_payload=_sample_trace(),
        output_dir=output_dir,
    )

    assert result["execution_analysis"]["task_completed"] is False
    assert result["execution_analysis"]["execution_note"] == "ExecutionAnalyzer requested manual review."
    assert result["execution_analysis"]["tool_issues"] == ["mcp:chatbotlevan:chatwoot_reply"]
    assert result["execution_analysis"]["analyzed_by"] == "host_trace_execution_analyzer"


def test_host_trace_supervisor_prefers_env_enabled_skill_evolver_path(
    tmp_path: Path,
    monkeypatch,
):
    output_dir = tmp_path / "recording"
    monkeypatch.setenv("OPENSPACE_HOST_ENABLE_SKILL_EVOLVER", "1")

    def _fake_execution_runner(*, trace_payload, analysis_context, recording_dir):
        assert trace_payload["trace_id"] == "cw_123_default_20260401T102233Z_ab12cd"
        assert analysis_context["task_description"] == "Xin gia khoa hoc"
        assert recording_dir == output_dir
        return {
            "task_completed": True,
            "execution_note": "ExecutionAnalyzer captured a reusable workflow.",
            "tool_issues": [],
            "skill_judgments": [],
            "evolution_suggestions": [
                {
                    "type": "captured",
                    "target_skills": [],
                    "category": "workflow",
                    "direction": "Capture this successful customer-facing reply workflow.",
                }
            ],
            "analyzed_by": "host_trace_execution_analyzer",
        }

    def _fake_skill_evolver(*, trace_payload, execution_analysis, recording_dir):
        assert trace_payload["trace_id"] == "cw_123_default_20260401T102233Z_ab12cd"
        assert execution_analysis["analyzed_by"] == "host_trace_execution_analyzer"
        assert recording_dir == output_dir
        return [
            {
                "skill_id": "captured-reply__v0_ab12cd34",
                "name": "captured-reply",
                "description": "Captured reply flow",
                "path": str(output_dir / "managed" / "captured-reply" / "SKILL.md"),
                "lineage": {
                    "origin": "captured",
                    "created_at": "2026-04-02T10:00:00",
                },
            }
        ]

    monkeypatch.setattr(
        HostTraceSupervisor,
        "_try_run_builtin_execution_analyzer",
        staticmethod(_fake_execution_runner),
    )
    monkeypatch.setattr(
        HostTraceSupervisor,
        "_try_run_builtin_skill_evolver",
        staticmethod(_fake_skill_evolver),
    )

    result = HostTraceSupervisor().analyze_trace(
        trace_payload=_sample_trace(),
        output_dir=output_dir,
    )

    assert result["execution_analysis"]["analyzed_by"] == "host_trace_execution_analyzer"
    assert result["evolved_skill_records"][0]["skill_id"] == "captured-reply__v0_ab12cd34"
    assert result["evolved_skill_records"][0]["name"] == "captured-reply"
