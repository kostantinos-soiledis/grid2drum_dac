#!/usr/bin/env python3
"""
Runtime compatibility shims for third-party dependency mismatches.
"""

from __future__ import annotations


def _patch_protobuf_message_factory() -> bool:
    try:
        from google.protobuf import message_factory
    except Exception:
        return False

    factory_cls = getattr(message_factory, "MessageFactory", None)
    get_message_class = getattr(message_factory, "GetMessageClass", None)
    if factory_cls is None or get_message_class is None:
        return False
    if hasattr(factory_cls, "GetPrototype"):
        return False

    def _get_prototype(self, descriptor):
        return get_message_class(descriptor)

    factory_cls.GetPrototype = _get_prototype
    return True


def apply_runtime_compat() -> dict[str, bool]:
    return {
        "protobuf_message_factory_getprototype": _patch_protobuf_message_factory(),
    }
