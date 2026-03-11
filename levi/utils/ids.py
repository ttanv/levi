"""ID generation utilities."""

import uuid


def generate_id() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())
