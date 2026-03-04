"""
Mock remote API service — backward-compatibility re-export.

.. deprecated::
    The canonical module has moved to
    ``app.providers.mock_remote``.
    Import from there instead.
"""

from app.providers.mock_remote import (  # noqa: F401
    router,
    TaskStatus,
    TextToImageRequest,
    EvaluateImageRequest,
    RepairImageRequest,
    SubmitResponse,
    PollResponse,
    clear_mock_tasks,
    get_mock_task,
    set_mock_task_status,
)
