"""Enrollment / verification engine and template storage.

TorchEmbedder (the trained model) lives in keyauth.auth.embedder so this
package can be imported, and tested, without PyTorch.
"""

from keyauth.auth.engine import EngineConfig, EnrollResult, KeyAuthEngine, VerifyResult
from keyauth.auth.store import TemplateStore, UserRecord
from keyauth.auth.validation import InvalidTimings, validate_timings

__all__ = [
    "EngineConfig",
    "EnrollResult",
    "InvalidTimings",
    "KeyAuthEngine",
    "TemplateStore",
    "UserRecord",
    "VerifyResult",
    "validate_timings",
]
