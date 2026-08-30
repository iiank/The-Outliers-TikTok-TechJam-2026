"""Stub the heavy optional dependencies pulled in transitively by
``search.search`` (via ``reranker/reranker.py`` -> torch/sentence-transformers,
and ``embed/store.py`` -> chromadb) so tests that only need
``SearchPipeline.search()``'s own branching logic -- not the real
retrieval/reranking stack -- can run in environments where those packages
aren't installed (e.g. a lightweight local dev machine, as opposed to the
full requirements.txt environment).

Purely additive: if a package is already importable, its real module is used
untouched, so this has no effect in an environment with the full
requirements.txt installed.
"""

from __future__ import annotations

import sys
import types


def _stub_if_missing(name: str) -> None:
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = types.ModuleType(name)


def _noop_decorator_factory():
    """Stand-in for ``torch.inference_mode``/``torch.no_grad``: usable both
    as a bare decorator (``@torch.inference_mode``) and called with/without
    arguments (``@torch.inference_mode()``), returning the function/context
    unchanged either way -- reranker.reranker.py only needs the decorator to
    exist at class-definition time, never to actually disable autograd."""

    def _identity_decorator(func=None):
        if func is not None and callable(func):
            return func

        def _wrap(f):
            return f

        return _wrap

    return _identity_decorator


if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        stub = types.ModuleType("torch")
        stub.inference_mode = _noop_decorator_factory()
        stub.no_grad = _noop_decorator_factory()
        sys.modules["torch"] = stub

if "sentence_transformers" not in sys.modules:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        stub = types.ModuleType("sentence_transformers")
        stub.SentenceTransformer = object
        stub.CrossEncoder = object
        sys.modules["sentence_transformers"] = stub

_stub_if_missing("chromadb")
