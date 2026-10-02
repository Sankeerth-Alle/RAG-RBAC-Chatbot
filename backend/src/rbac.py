"""Centralized role-based access control policy."""

from typing import FrozenSet


ROLE_PERMISSIONS: dict[str, FrozenSet[str]] = {
    "hr": frozenset({"access_hr_documents"}),
    "engineering": frozenset({"access_engineering_documents"}),
    "c_level": frozenset({"access_hr_documents", "access_engineering_documents"}),
}
ROLE_HIERARCHY = ("engineering", "hr", "c_level")


def is_valid_role(role: str) -> bool:
    """Return whether role is defined by the application policy."""
    return role in ROLE_PERMISSIONS


def accessible_document_levels(role: str) -> list[str]:
    """Return upload access levels visible to a role, from the hierarchy."""
    if not is_valid_role(role):
        return []
    return [
        document_level
        for document_level in ROLE_HIERARCHY
        if can_access_documents(role, document_level)
    ]


def can_access_documents(role: str, document_access_level: str) -> bool:
    """Check whether a role may access a document's access level."""
    if not is_valid_role(role):
        return False
    if role == "c_level":
        return bool(document_access_level)
    if document_access_level not in ROLE_HIERARCHY:
        return False
    return ROLE_HIERARCHY.index(role) >= ROLE_HIERARCHY.index(document_access_level)