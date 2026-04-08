from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from .trace_models import HostTraceContext


def _extract_user_instruction(user_message: str) -> str:
    text = str(user_message or "").strip()
    if not text:
        return ""
    match = re.search(r"USER_MESSAGE:\s*(.*)", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _tool_result_status(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("error"):
            return "error"
        message = str(result.get("message") or "").strip().lower()
        if message.startswith("error") or "failed" in message:
            return "error"
        ok_value = result.get("ok")
        if ok_value is False:
            return "error"
    if isinstance(result, str):
        lowered = result.strip().lower()
        if lowered.startswith("error") or "failed" in lowered:
            return "error"
    return "success"


class KimiTraceAdapter:
    @staticmethod
    def write_recording(
        *,
        trace_payload: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Any]:
        safe_output_dir = Path(output_dir).resolve()
        safe_output_dir.mkdir(parents=True, exist_ok=True)

        trace_id = str(trace_payload.get("trace_id") or "").strip()
        user_instruction = _extract_user_instruction(str(trace_payload.get("user_message") or ""))
        metadata = KimiTraceAdapter._build_metadata(
            trace_payload=trace_payload,
            task_description=user_instruction,
        )
        conversations = KimiTraceAdapter._build_conversations(trace_payload=trace_payload)
        traj_records = KimiTraceAdapter._build_traj_records(trace_payload=trace_payload)

        metadata_file = safe_output_dir / "metadata.json"
        conversations_file = safe_output_dir / "conversations.jsonl"
        traj_file = safe_output_dir / "traj.jsonl"

        metadata_file.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        conversations_file.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in conversations) + "\n",
            encoding="utf-8",
        )
        traj_file.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in traj_records) + "\n",
            encoding="utf-8",
        )

        context = HostTraceContext(
            trace_id=trace_id,
            recording_dir=safe_output_dir,
            metadata_file=metadata_file,
            conversations_file=conversations_file,
            traj_file=traj_file,
            task_description=user_instruction,
        )
        return context.as_dict()

    @staticmethod
    def _build_metadata(
        *,
        trace_payload: Dict[str, Any],
        task_description: str,
    ) -> Dict[str, Any]:
        execution = dict(trace_payload.get("execution") or {})
        skills_runtime = dict(trace_payload.get("skills_runtime") or {})
        tool_calls = [
            str(item).strip()
            for item in (execution.get("tool_calls") or [])
            if str(item).strip()
        ]
        selected_skill_ids = [
            str(item).strip()
            for item in (skills_runtime.get("selected_skill_ids") or [])
            if str(item).strip()
        ]
        used_skill_ids = [
            str(item).strip()
            for item in (skills_runtime.get("used_skill_ids") or [])
            if str(item).strip()
        ]
        available_skill_ids = list(dict.fromkeys([*selected_skill_ids, *used_skill_ids]))
        tool_defs = [
            {
                "name": tool_name,
                "backend": "mcp",
                "server_name": "chatbotlevan",
            }
            for tool_name in tool_calls
        ]
        return {
            "task_id": str(trace_payload.get("trace_id") or "").strip(),
            "task_description": task_description,
            "start_time": str(trace_payload.get("created_at") or "").strip(),
            "backends": ["mcp"],
            "execution_outcome": {
                "status": "success" if str(execution.get("assistant_text") or "").strip() else "unknown",
                "iterations": 1,
            },
            "retrieved_tools": {
                "instruction": task_description,
                "count": len(tool_defs),
                "tools": tool_defs,
            },
            "skill_selection": {
                "method": "host_trace_adapter",
                "task": task_description,
                "selected": selected_skill_ids,
                "available_skills": available_skill_ids,
            },
            "kimi_trace": {
                "version": trace_payload.get("version"),
                "trace_id": str(trace_payload.get("trace_id") or "").strip(),
                "agent_name": str(trace_payload.get("agent_name") or "").strip(),
                "conversation_id": str(trace_payload.get("conversation_id") or "").strip(),
                "session_id": str(trace_payload.get("session_id") or "").strip(),
                "source": str(trace_payload.get("source") or "").strip(),
                "system_prompt": dict(trace_payload.get("system_prompt") or {}),
                "skills_runtime": skills_runtime,
                "chatwoot": dict(trace_payload.get("chatwoot") or {}),
            },
        }

    @staticmethod
    def _build_conversations(
        *,
        trace_payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        execution = dict(trace_payload.get("execution") or {})
        tool_calls = [
            str(item).strip()
            for item in (execution.get("tool_calls") or [])
            if str(item).strip()
        ]
        user_instruction = _extract_user_instruction(str(trace_payload.get("user_message") or ""))
        history_snapshot = list(trace_payload.get("history_snapshot") or [])
        setup_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Kimi host trace replay. This setup was adapted from an external runtime "
                    "trace so OpenSpace can analyze the execution."
                ),
            },
            {
                "role": "user",
                "content": user_instruction,
            },
        ]
        for item in history_snapshot:
            role = str(item.get("role") or "").strip()
            content = str(item.get("content") or "").strip()
            if role and content:
                setup_messages.append({"role": role, "content": content})

        tool_call_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"tool-call-{index}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": "{}",
                    },
                }
                for index, tool_name in enumerate(tool_calls, start=1)
            ],
        }
        delta_messages: list[dict[str, Any]] = [tool_call_message]
        tool_results = dict(execution.get("tool_results") or {})
        for tool_name in tool_calls:
            if tool_name not in tool_results:
                continue
            tool_result = tool_results.get(tool_name)
            delta_messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )
        delta_messages.append(
            {
                "role": "assistant",
                "content": str(execution.get("assistant_text") or "").strip(),
            }
        )

        return [
            {
                "type": "setup",
                "agent_name": "GroundingAgent",
                "timestamp": str(trace_payload.get("created_at") or "").strip(),
                "messages": setup_messages,
                "tools": [
                    {
                        "name": tool_name,
                        "backend": "mcp",
                        "description": f"[MCP] Adapted Kimi runtime tool {tool_name}",
                    }
                    for tool_name in tool_calls
                ],
                "extra": {
                    "source": "kimi_host_trace",
                    "trace_id": str(trace_payload.get("trace_id") or "").strip(),
                },
            },
            {
                "type": "iteration",
                "agent_name": "GroundingAgent",
                "iteration": 1,
                "timestamp": str(trace_payload.get("created_at") or "").strip(),
                "response_metadata": {
                    "has_tool_calls": bool(tool_calls),
                    "tool_calls_count": len(tool_calls),
                },
                "delta_messages": delta_messages,
                "extra": {
                    "source": "kimi_host_trace",
                    "trace_id": str(trace_payload.get("trace_id") or "").strip(),
                },
            },
        ]

    @staticmethod
    def _build_traj_records(
        *,
        trace_payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        execution = dict(trace_payload.get("execution") or {})
        tool_calls = [
            str(item).strip()
            for item in (execution.get("tool_calls") or [])
            if str(item).strip()
        ]
        tool_results = dict(execution.get("tool_results") or {})
        records: list[dict[str, Any]] = []
        timestamp = str(trace_payload.get("created_at") or "").strip()
        for index, tool_name in enumerate(tool_calls, start=1):
            tool_result = tool_results.get(tool_name)
            records.append(
                {
                    "step": index,
                    "timestamp": timestamp,
                    "backend": "mcp",
                    "server": "chatbotlevan",
                    "tool": tool_name,
                    "command": tool_name,
                    "parameters": {},
                    "result": {
                        "status": _tool_result_status(tool_result),
                        "output": tool_result,
                    },
                }
            )
        return records
