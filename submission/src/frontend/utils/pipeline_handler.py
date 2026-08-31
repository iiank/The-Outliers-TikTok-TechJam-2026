"""
Thread-isolated backend adapter for the Streamlit recommender demo.
Streamlit can execute successive reruns on different script-runner threads.
The CRIS search stack keeps long-lived resources, including a module-level
'SearchPipeline' singleton. 
PipelineHandler owns one dedicated worker thread and guarantees that Agent is 
called only on that worker. session_id keys keep dialogue states separate.
"""

from __future__ import annotations

import atexit
import copy
import importlib
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Mapping, Optional, TypeVar, cast

__all__ = [
    "PipelineExecutionError",
    "PipelineHandler",
    "close_pipeline_handler",
    "get_pipeline_handler",
]

_ResultT = TypeVar("_ResultT")


class PipelineExecutionError(RuntimeError):
    """Raised when an operation fails inside the isolated backend worker."""


class PipelineHandler:
    """Run every frontend Agent operation on one persistent worker thread."""

    def __init__(
        self,
        agent_module: str = "submission.agent",
        agent_class: str = "Agent",
    ) -> None:
        self._agent_module = str(agent_module)
        self._agent_class = str(agent_class)
        self._agent: Any = None
        self._worker_thread_id: Optional[int] = None
        self._worker_thread_name: Optional[str] = None
        self._closed = False
        self._lifecycle_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="cris-pipeline",
        )

        try:
            self._executor.submit(self._initialise_worker).result()
        except BaseException:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise

    @property
    def worker_thread_id(self) -> Optional[int]:
        """Backend thread identifier (diagnostics)"""
        return self._worker_thread_id

    @property
    def worker_thread_name(self) -> Optional[str]:
        """Backend thread name (diagnostics)"""
        return self._worker_thread_name

    @property
    def is_ready(self) -> bool:
        """Whether Agent initialisation is success and handler is open."""
        return not self._closed and self._agent is not None

    def reset(
        self,
        session_id: str,
        user_profile: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Start or replace one conversation state on the backend worker."""
        
        session_key = self._validate_session_id(session_id)
        profile_copy: Dict[str, Any] = copy.deepcopy(dict(user_profile or {}))
        self._call("Agent.reset", self._reset_on_worker, session_key, profile_copy)

    def respond_chat(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Execute one frontend turn through 'Agent.respond_chat()'."""

        session_key = self._validate_session_id(session_id)
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message must be a non-empty string")
        turn_number = self._validate_positive_int("turn", turn)
        result_limit = self._validate_positive_int("top_k", top_k)

        result = self._call(
            "Agent.respond_chat",
            self._respond_chat_on_worker,
            session_key,
            user_message.strip(),
            turn_number,
            result_limit,
        )
        if not isinstance(result, dict):
            raise PipelineExecutionError(
                "Agent.respond_chat() returned a non-dictionary result: "
                f"{type(result).__name__}"
            )
        return cast(Dict[str, Any], result)

    def close(self) -> None:
        """Stop accepting work and shut down the worker (safe to call twice)."""

        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._agent = None

    def _initialise_worker(self) -> None:
        current_thread = threading.current_thread()
        self._worker_thread_id = threading.get_ident()
        self._worker_thread_name = current_thread.name

        # keep import-time resources on worker
        module = importlib.import_module(self._agent_module)
        try:
            agent_type = getattr(module, self._agent_class)
        except AttributeError as exc:
            raise ImportError(
                f"Module {self._agent_module!r} has no "
                f"{self._agent_class!r} class"
            ) from exc
        self._agent = agent_type()

    def _reset_on_worker(
        self,
        session_id: str,
        user_profile: Dict[str, Any],
    ) -> None:
        self._assert_worker_thread()
        self._agent.reset(session_id=session_id, user_profile=user_profile)

    def _respond_chat_on_worker(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> Dict[str, Any]:
        self._assert_worker_thread()
        output = self._agent.respond_chat(
            session_id=session_id,
            user_message=user_message,
            turn=turn,
            top_k=top_k,
        )
        if not isinstance(output, dict):
            raise TypeError(
                "Agent.respond_chat() must return a dict, "
                f"got {type(output).__name__}"
            )

        # Keep snapshot on the worker
        snapshot = copy.deepcopy(cast(Dict[str, Any], output))
        raw_diagnostics = snapshot.get("diagnostics")
        diagnostics = (
            raw_diagnostics if isinstance(raw_diagnostics, dict) else {}
        )

        diagnostics.setdefault("turn", turn)
        snapshot["diagnostics"] = diagnostics
        return snapshot

    def _call(
        self,
        operation: str,
        function: Callable[..., _ResultT],
        *args: Any,
    ) -> _ResultT:
        with self._lifecycle_lock:
            if self._closed:
                raise PipelineExecutionError(
                    f"Cannot run {operation}: the pipeline handler is closed"
                )
            future: Future[_ResultT] = self._executor.submit(function, *args)

        try:
            return future.result()
        except PipelineExecutionError:
            raise
        except Exception as exc:
            thread_suffix = (
                f" on worker thread {self._worker_thread_id}"
                if self._worker_thread_id is not None
                else ""
            )
            raise PipelineExecutionError(
                f"{operation} failed{thread_suffix}: {exc}"
            ) from exc

    def _assert_worker_thread(self) -> None:
        current_thread_id = threading.get_ident()
        if current_thread_id != self._worker_thread_id:
            raise PipelineExecutionError(
                "Backend operation escaped the isolated worker: "
                f"expected thread {self._worker_thread_id}, "
                f"received {current_thread_id}"
            )
        if self._agent is None:
            raise PipelineExecutionError("CRIS Agent has not been initialized")

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        value = str(session_id).strip()
        if not value:
            raise ValueError("session_id must be a non-empty string")
        return value

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be an integer greater than or equal to 1")
        return value


_HANDLER: Optional[PipelineHandler] = None
_HANDLER_LOCK = threading.Lock()


def get_pipeline_handler() -> PipelineHandler:
    """Return the process-wide handler shared by all Streamlit reruns."""

    global _HANDLER
    if _HANDLER is None or not _HANDLER.is_ready:
        with _HANDLER_LOCK:
            if _HANDLER is None or not _HANDLER.is_ready:
                _HANDLER = PipelineHandler()
    return _HANDLER


def close_pipeline_handler() -> None:
    """Close and clear the process-wide handler during interpreter shutdown."""

    global _HANDLER
    with _HANDLER_LOCK:
        handler = _HANDLER
        _HANDLER = None
    if handler is not None:
        handler.close()


atexit.register(close_pipeline_handler)