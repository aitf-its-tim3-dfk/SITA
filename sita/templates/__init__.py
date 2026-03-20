"""Chat template utilities — resolve template names to Jinja strings."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("sita.templates")

# Built-in templates live alongside this file
_TEMPLATES_DIR = Path(__file__).parent


def load_chat_template(name: str) -> str | None:
    """Load a chat template by name or file path.

    Resolution order:
      1. If *name* contains a path separator or ends with ``.jinja``,
         treat it as a direct file path.
      2. Otherwise look for ``<name>.jinja`` inside the built-in
         ``sita/templates/`` directory.

    Returns the template string, or ``None`` if not found.
    """
    # Direct file path
    candidate = Path(name)
    if candidate.suffix == ".jinja" or "/" in name or "\\" in name:
        if candidate.is_file():
            logger.info(f"Loading chat template from path: {candidate}")
            return candidate.read_text(encoding="utf-8")
        logger.warning(f"Chat template file not found: {candidate}")
        return None

    # Built-in lookup
    builtin = _TEMPLATES_DIR / f"{name}.jinja"
    if builtin.is_file():
        logger.info(f"Loading built-in chat template: {builtin.name}")
        return builtin.read_text(encoding="utf-8")

    logger.warning(
        f"No built-in chat template named '{name}'. "
        f"Available: {[p.stem for p in _TEMPLATES_DIR.glob('*.jinja')]}"
    )
    return None
