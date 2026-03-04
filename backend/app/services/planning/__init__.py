"""
Planning model implementations.

Contains concrete planning model classes (Kimi, Gemini, Qwen-VL)
built on the ``BasePlanningModel`` base class defined in ``base.py``.

.. note::
    The canonical provider interface lives in ``app.providers``.
    These legacy planning models extend ``BasePlanningModel`` which
    wraps the provider interface with additional helper methods.
"""
