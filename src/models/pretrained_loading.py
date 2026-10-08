from __future__ import annotations

import logging
from collections.abc import Sequence


def log_pretrained_load_result(
    missing: Sequence[str],
    unexpected: Sequence[str],
    logger: logging.Logger,
    *,
    model_name: str = "",
    expected_missing_prefixes: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Log the result of a ``load_state_dict(..., strict=False)`` call.

    Keys that match any prefix in *expected_missing_prefixes* are considered
    intentionally absent (e.g. a freshly initialised decoder head) and are
    reported at DEBUG level only.  Everything else is reported at WARNING so
    that genuine loading problems are visible in normal log output.

    Args:
        missing: List of keys present in the model but absent from the
            checkpoint (returned by ``load_state_dict``).
        unexpected: List of keys present in the checkpoint but absent from
            the model (returned by ``load_state_dict``).
        logger: Logger to write messages to.
        model_name: Optional model name used in log messages.
        expected_missing_prefixes: Key prefixes that are expected to be
            missing (e.g. ``["decoder.", "head.", "teacher_"]``).

    Returns:
        ``(truly_missing, unexpected_list)`` — *truly_missing* excludes keys
        that match *expected_missing_prefixes*; *unexpected_list* is the
        original *unexpected* sequence as a plain list.
    """
    prefix = f"{model_name}: " if model_name else ""
    expected_prefixes: Sequence[str] = expected_missing_prefixes or []

    # Split missing into expected-absent and truly-missing
    expected_absent: list[str] = []
    truly_missing: list[str] = []
    for key in missing:
        if any(key.startswith(p) for p in expected_prefixes):
            expected_absent.append(key)
        else:
            truly_missing.append(key)

    unexpected_list = list(unexpected)

    if expected_absent:
        logger.debug(f"{prefix}Skipped {len(expected_absent)} expected-missing key(s) (intentionally reinitialised): {expected_absent}")

    if truly_missing:
        logger.warning(f"{prefix}{len(truly_missing)} key(s) missing from checkpoint (model weights were NOT loaded for these).")
        logger.debug(f"{prefix}Missing keys: {truly_missing}")
    else:
        logger.debug(f"{prefix}No truly-missing keys.")

    if unexpected_list:
        logger.warning(f"{prefix}{len(unexpected_list)} unexpected key(s) in checkpoint (ignored — not present in model).")
        logger.debug(f"{prefix}Unexpected keys: {unexpected_list}")
    else:
        logger.debug(f"{prefix}No unexpected keys.")

    return truly_missing, unexpected_list
