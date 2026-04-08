from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from openspace.skill_engine.types import (
    EvolutionSuggestion,
    EvolutionType,
    ExecutionAnalysis,
    SkillCategory,
)

from .kimi_trace_adapter import KimiTraceAdapter


class HostTraceSupervisor:
    """Prepare analyzer-compatible recording packages from external host traces."""

    def __init__(
        self,
        *,
        analysis_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self._analysis_runner = analysis_runner

    def prepare_recording(
        self,
        *,
        trace_payload: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Any]:
        return KimiTraceAdapter.write_recording(
            trace_payload=trace_payload,
            output_dir=output_dir,
        )

    def analyze_trace(
        self,
        *,
        trace_payload: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Any]:
        recording = self.prepare_recording(
            trace_payload=trace_payload,
            output_dir=output_dir,
        )
        context = self._build_analysis_context(
            recording_dir=Path(recording["recording_dir"]),
        )
        execution_analysis = self._resolve_execution_analysis(
            trace_payload=trace_payload,
            analysis_context=context,
            recording_dir=Path(recording["recording_dir"]),
        )
        evolved_skill_records = self._resolve_evolved_skills(
            trace_payload=trace_payload,
            execution_analysis=execution_analysis,
            recording_dir=Path(recording["recording_dir"]),
        )
        return {
            "recording": recording,
            "analysis_context": context,
            "execution_analysis": execution_analysis,
            "evolved_skill_records": evolved_skill_records,
        }

    def _build_analysis_context(
        self,
        *,
        recording_dir: Path,
    ) -> Dict[str, Any]:
        analyzer_context = self._try_build_context_with_openspace_analyzer(recording_dir=recording_dir)
        if analyzer_context is not None:
            return analyzer_context
        return self._build_fallback_context(recording_dir=recording_dir)

    def _try_build_context_with_openspace_analyzer(
        self,
        *,
        recording_dir: Path,
    ) -> Dict[str, Any] | None:
        try:
            from openspace.skill_engine.analyzer import ExecutionAnalyzer
            from openspace.skill_engine.conversation_formatter import format_conversations
        except Exception:
            return None

        analyzer = ExecutionAnalyzer(
            store=None,
            llm_client=None,
            enabled=False,
        )
        context = analyzer._load_recording_context(
            recording_dir,
            {},
        )
        if context is None:
            return None
        formatted_conversation = format_conversations(
            list(context.get("conversations") or []),
            budget=12000,
        )
        return {
            "task_description": str(context.get("task_description") or "").strip(),
            "execution_status": str(context.get("execution_status") or "").strip() or "unknown",
            "conversation_count": len(list(context.get("conversations") or [])),
            "used_tool_keys": sorted(str(item).strip() for item in (context.get("used_tool_keys") or set()) if str(item).strip()),
            "formatted_conversation": formatted_conversation,
            "analysis_mode": "openspace_analyzer",
        }

    def _build_fallback_context(
        self,
        *,
        recording_dir: Path,
    ) -> Dict[str, Any]:
        metadata_file = recording_dir / "metadata.json"
        conversations_file = recording_dir / "conversations.jsonl"
        traj_file = recording_dir / "traj.jsonl"
        metadata = (
            json.loads(metadata_file.read_text(encoding="utf-8"))
            if metadata_file.exists()
            else {}
        )
        conversations = []
        if conversations_file.exists():
            for line in conversations_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    conversations.append(json.loads(line))
        traj_records = []
        if traj_file.exists():
            for line in traj_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    traj_records.append(json.loads(line))
        used_tool_keys: set[str] = set()
        for entry in traj_records:
            backend = str(entry.get("backend") or "").strip()
            server = str(entry.get("server") or "").strip()
            tool = str(entry.get("tool") or "").strip()
            if backend and tool:
                used_tool_keys.add(f"{backend}:{tool}")
            if backend and server and tool:
                used_tool_keys.add(f"{backend}:{server}:{tool}")
        formatted_chunks: list[str] = []
        for conversation in conversations:
            if conversation.get("type") == "setup":
                for message in conversation.get("messages") or []:
                    role = str(message.get("role") or "").strip()
                    content = str(message.get("content") or "").strip()
                    if role and content:
                        formatted_chunks.append(f"[{role.upper()}] {content}")
            elif conversation.get("type") == "iteration":
                for message in conversation.get("delta_messages") or []:
                    role = str(message.get("role") or "").strip()
                    content = str(message.get("content") or "").strip()
                    if role == "assistant" and content:
                        formatted_chunks.append(f"[ASSISTANT] {content}")
                    elif role == "tool" and content:
                        formatted_chunks.append(f"[TOOL] {content}")
        return {
            "task_description": str(metadata.get("task_description") or "").strip(),
            "execution_status": str(((metadata.get("execution_outcome") or {}).get("status")) or "unknown").strip() or "unknown",
            "conversation_count": len(conversations),
            "used_tool_keys": sorted(used_tool_keys),
            "formatted_conversation": "\n".join(formatted_chunks).strip(),
            "analysis_mode": "fallback_parser",
        }

    def _build_execution_analysis(
        self,
        *,
        trace_payload: Dict[str, Any],
        analysis_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        execution = dict(trace_payload.get("execution") or {})
        tool_calls = [
            str(item).strip()
            for item in (execution.get("tool_calls") or [])
            if str(item).strip()
        ]
        assistant_text = str(execution.get("assistant_text") or "").strip()
        tool_results = dict(execution.get("tool_results") or {})
        has_tool_error = any(self._tool_result_has_error(value) for value in tool_results.values())
        task_completed = bool(assistant_text) and not has_tool_error

        suggestions: list[EvolutionSuggestion] = []
        if task_completed and any(name in {"chatwoot_reply", "chatwoot_reply_comment"} for name in tool_calls):
            suggestions.append(
                EvolutionSuggestion(
                    evolution_type=EvolutionType.CAPTURED,
                    target_skill_ids=[],
                    category=SkillCategory.WORKFLOW,
                    direction="Capture this successful customer-facing reply workflow as a reusable managed skill draft.",
                )
            )

        analysis = ExecutionAnalysis(
            task_id=str(trace_payload.get("trace_id") or "").strip(),
            timestamp=datetime.now(),
            task_completed=task_completed,
            execution_note=(
                "Host trace indicates a successful assistant outcome."
                if task_completed
                else "Host trace needs review because it contains tool errors or no clear assistant outcome."
            ),
            tool_issues=sorted(
                tool_key
                for tool_key in (analysis_context.get("used_tool_keys") or [])
                if has_tool_error and str(tool_key).strip()
            ) if has_tool_error else [],
            skill_judgments=[],
            evolution_suggestions=suggestions,
            analyzed_by="host_trace_heuristic",
            analyzed_at=datetime.now(),
        )
        return analysis.to_dict()

    def _resolve_execution_analysis(
        self,
        *,
        trace_payload: Dict[str, Any],
        analysis_context: Dict[str, Any],
        recording_dir: Path,
    ) -> Dict[str, Any]:
        if callable(self._analysis_runner):
            try:
                raw = self._analysis_runner(
                    trace_payload=trace_payload,
                    analysis_context=analysis_context,
                )
                normalized = self._normalize_execution_analysis(raw, trace_payload=trace_payload)
                if normalized is not None:
                    return normalized
            except Exception:
                pass
        try:
            raw = self._try_run_builtin_execution_analyzer(
                trace_payload=trace_payload,
                analysis_context=analysis_context,
                recording_dir=recording_dir,
            )
            normalized = self._normalize_execution_analysis(raw, trace_payload=trace_payload)
            if normalized is not None:
                return normalized
        except Exception:
            pass
        return self._build_execution_analysis(
            trace_payload=trace_payload,
            analysis_context=analysis_context,
        )

    def _resolve_evolved_skills(
        self,
        *,
        trace_payload: Dict[str, Any],
        execution_analysis: Dict[str, Any],
        recording_dir: Path,
    ) -> list[Dict[str, Any]]:
        suggestions = list(execution_analysis.get("evolution_suggestions") or [])
        if not suggestions:
            return []
        try:
            raw = self._try_run_builtin_skill_evolver(
                trace_payload=trace_payload,
                execution_analysis=execution_analysis,
                recording_dir=recording_dir,
            )
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        return [dict(item) for item in raw if isinstance(item, dict)]

    @staticmethod
    def _try_run_builtin_execution_analyzer(
        *,
        trace_payload: Dict[str, Any],
        analysis_context: Dict[str, Any],
        recording_dir: Path,
    ) -> Dict[str, Any] | None:
        enabled_raw = str(os.getenv("OPENSPACE_HOST_ENABLE_EXECUTION_ANALYZER") or "").strip().lower()
        if enabled_raw not in {"1", "true", "yes", "on"}:
            return None

        try:
            from openspace.llm import LLMClient
            from openspace.skill_engine.analyzer import ExecutionAnalyzer
            from openspace.skill_engine.registry import SkillRegistry
            from openspace.skill_engine.store import SkillStore
        except Exception:
            return None

        skill_dirs: list[Path] = []
        skills_runtime = dict(trace_payload.get("skills_runtime") or {})
        for raw_path in skills_runtime.get("base_dirs") or []:
            safe_path = Path(str(raw_path)).expanduser()
            if safe_path.exists():
                skill_dirs.append(safe_path.resolve())
        managed_dir_raw = str(skills_runtime.get("managed_dir") or "").strip()
        if managed_dir_raw:
            managed_dir = Path(managed_dir_raw).expanduser()
            if managed_dir.exists():
                skill_dirs.append(managed_dir.resolve())

        model = (
            str(os.getenv("OPENSPACE_HOST_ANALYZER_MODEL") or "").strip()
            or "openrouter/anthropic/claude-sonnet-4.5"
        )
        timeout_raw = str(os.getenv("OPENSPACE_HOST_ANALYZER_TIMEOUT") or "").strip()
        try:
            timeout = float(timeout_raw or "45")
        except ValueError:
            timeout = 45.0

        async def _runner() -> Dict[str, Any] | None:
            registry = SkillRegistry(skill_dirs=skill_dirs)
            registry.discover()
            store = SkillStore(db_path=recording_dir / "host_execution_analyzer.db")
            try:
                await store.sync_from_registry(registry.list_skills())
                analyzer = ExecutionAnalyzer(
                    store=store,
                    llm_client=LLMClient(
                        model=model,
                        timeout=timeout,
                        enable_tool_result_summarization=False,
                    ),
                    model=model,
                    skill_registry=registry,
                )
                analysis = await analyzer.analyze_execution(
                    task_id=str(trace_payload.get("trace_id") or "").strip(),
                    recording_dir=str(recording_dir),
                    execution_result={
                        "instruction": str(analysis_context.get("task_description") or "").strip(),
                        "status": str(analysis_context.get("execution_status") or "").strip() or "unknown",
                        "iterations": 1,
                    },
                    available_tools=[],
                )
                return analysis.to_dict() if analysis is not None else None
            finally:
                store.close()

        return HostTraceSupervisor._run_async_blocking(_runner())

    @staticmethod
    def _try_run_builtin_skill_evolver(
        *,
        trace_payload: Dict[str, Any],
        execution_analysis: Dict[str, Any],
        recording_dir: Path,
    ) -> list[Dict[str, Any]] | None:
        enabled_raw = str(os.getenv("OPENSPACE_HOST_ENABLE_SKILL_EVOLVER") or "").strip().lower()
        if enabled_raw not in {"1", "true", "yes", "on"}:
            return None

        try:
            from openspace.llm import LLMClient
            from openspace.skill_engine.evolver import SkillEvolver
            from openspace.skill_engine.registry import SkillRegistry
            from openspace.skill_engine.store import SkillStore
            from openspace.skill_engine.types import ExecutionAnalysis
        except Exception:
            return None

        skill_dirs: list[Path] = []
        seen_dirs: set[str] = set()
        skills_runtime = dict(trace_payload.get("skills_runtime") or {})

        managed_dir_raw = str(skills_runtime.get("managed_dir") or "").strip()
        ordered_paths: list[str] = []
        if managed_dir_raw:
            ordered_paths.append(managed_dir_raw)
        ordered_paths.extend(str(item) for item in (skills_runtime.get("base_dirs") or []))

        for raw_path in ordered_paths:
            safe_path = Path(str(raw_path)).expanduser()
            if not safe_path.exists():
                continue
            resolved = str(safe_path.resolve())
            if resolved in seen_dirs:
                continue
            seen_dirs.add(resolved)
            skill_dirs.append(Path(resolved))

        model = (
            str(os.getenv("OPENSPACE_HOST_EVOLVER_MODEL") or "").strip()
            or str(os.getenv("OPENSPACE_HOST_ANALYZER_MODEL") or "").strip()
            or "openrouter/anthropic/claude-sonnet-4.5"
        )
        timeout_raw = str(os.getenv("OPENSPACE_HOST_EVOLVER_TIMEOUT") or "").strip()
        try:
            timeout = float(timeout_raw or "45")
        except ValueError:
            timeout = 45.0

        async def _runner() -> list[Dict[str, Any]]:
            registry = SkillRegistry(skill_dirs=skill_dirs)
            registry.discover()
            store = SkillStore(db_path=recording_dir / "host_skill_evolver.db")
            try:
                await store.sync_from_registry(registry.list_skills())
                evolver = SkillEvolver(
                    store=store,
                    registry=registry,
                    llm_client=LLMClient(
                        model=model,
                        timeout=timeout,
                        enable_tool_result_summarization=False,
                    ),
                    model=model,
                )
                analysis = ExecutionAnalysis.from_dict(
                    HostTraceSupervisor._normalize_execution_analysis(
                        execution_analysis,
                        trace_payload=trace_payload,
                    )
                )
                records = await evolver.process_analysis(analysis)
                return [record.to_dict() for record in records]
            finally:
                store.close()

        return HostTraceSupervisor._run_async_blocking(_runner())

    @staticmethod
    def _run_async_blocking(coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result_holder: dict[str, Any] = {}
        error_holder: dict[str, BaseException] = {}

        def _thread_runner() -> None:
            try:
                result_holder["value"] = asyncio.run(coro)
            except BaseException as exc:  # pragma: no cover - surfaced to caller
                error_holder["error"] = exc

        thread = threading.Thread(target=_thread_runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder.get("value")

    @staticmethod
    def _normalize_execution_analysis(
        raw: Any,
        *,
        trace_payload: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        return {
            "task_id": str(raw.get("task_id") or trace_payload.get("trace_id") or "").strip(),
            "timestamp": str(raw.get("timestamp") or datetime.now().isoformat()),
            "task_completed": bool(raw.get("task_completed", False)),
            "execution_note": str(raw.get("execution_note") or "").strip(),
            "tool_issues": list(raw.get("tool_issues") or []),
            "skill_judgments": list(raw.get("skill_judgments") or []),
            "evolution_suggestions": list(raw.get("evolution_suggestions") or []),
            "candidate_for_evolution": bool(raw.get("evolution_suggestions") or []),
            "analyzed_by": str(raw.get("analyzed_by") or "").strip(),
            "analyzed_at": str(raw.get("analyzed_at") or datetime.now().isoformat()),
        }

    @staticmethod
    def _tool_result_has_error(result: Any) -> bool:
        if isinstance(result, dict):
            if bool(result.get("error")):
                return True
            ok_value = result.get("ok")
            if ok_value is False:
                return True
            message = str(result.get("message") or "").strip().lower()
            if message.startswith("error") or "failed" in message:
                return True
            return any(HostTraceSupervisor._tool_result_has_error(value) for value in result.values())
        if isinstance(result, list):
            return any(HostTraceSupervisor._tool_result_has_error(value) for value in result)
        if isinstance(result, str):
            lowered = result.strip().lower()
            return lowered.startswith("error") or "failed" in lowered
        return False
