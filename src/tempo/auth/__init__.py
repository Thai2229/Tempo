"""Enrollment / verification engine and template storage.

TorchEmbedder (the trained model) lives in tempo.auth.embedder so this
package can be imported, and tested, without PyTorch.
"""

from tempo.auth.engine import EngineConfig, EnrollResult, TempoEngine, VerifyResult
from tempo.auth.store import SiteRecord, TemplateStore, UserRecord
from tempo.auth.validation import InvalidTimings, validate_timings

__all__ = [
    "EngineConfig",
    "EnrollResult",
    "InvalidTimings",
    "TempoEngine",
    "SiteRecord",
    "TemplateStore",
    "UserRecord",
    "VerifyResult",
    "validate_timings",
]
