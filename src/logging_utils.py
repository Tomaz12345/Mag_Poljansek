from __future__ import annotations

import argparse
import json
import logging
import logging.config
import os
from typing import Any


def setup_logging(log_level: str, log_file_path: str | None = None) -> None:
    """Set up logging using the shared logging config.

    Args:
        log_level: Root logger level to use.
        log_file_path: Optional log file path. If None, file logging is disabled.
    """
    try:
        cfg_path = os.path.join(os.path.dirname(__file__), "logging_config.json")
        with open(cfg_path) as fh:
            log_cfg = json.load(fh)

        handlers = log_cfg.get("handlers")
        if isinstance(handlers, dict):
            default_handler = handlers.get("default")
            if isinstance(default_handler, dict):
                default_handler["level"] = "INFO"

            file_handler = handlers.get("file")
            if log_file_path and isinstance(file_handler, dict):
                file_handler["filename"] = log_file_path
                file_handler["level"] = str(log_level).upper()
            elif isinstance(file_handler, dict):
                handlers.pop("file", None)

        root_handlers: list[str] = ["default"]
        if log_file_path and isinstance(handlers, dict) and "file" in handlers:
            root_handlers.append("file")

        log_cfg["root"] = {"level": log_level, "handlers": root_handlers}
        logging.config.dictConfig(log_cfg)
    except Exception:
        logging.basicConfig(level=getattr(logging, str(log_level).upper(), logging.INFO))


def _collect_all_children(group_def: dict[str, Any], items: dict[str, Any]) -> set[str]:
    """Collect all child param names for a group definition."""
    children: set[str] = set()
    if "params" in group_def:
        children.update(group_def["params"])
    if "extra_params" in group_def:
        children.update(group_def["extra_params"])
    if "params_prefix" in group_def:
        prefix = group_def["params_prefix"]
        exclude: set[str] = set(group_def.get("exclude_from_children", []))
        children.update(k for k in items if k.startswith(prefix) and k not in exclude)
    return children


def filter_disabled_children(args: argparse.Namespace | dict[str, Any]) -> dict[str, Any]:
    """Remove child parameters that are irrelevant to the current config.

    Uses CONFIG_PARAMETER_GROUPS to determine parent-child relationships.
    Handles two cases:

    1. **Boolean toggles** (``enable_key``): when the toggle is False, all
       child params are removed except the toggle itself.
    2. **Enum selectors** (``selector_key`` + ``children_by_value``): only
       the child params relevant to the current selector value are kept;
       all other child params in the group are removed.
    """
    from src.constants import CONFIG_PARAMETER_GROUPS

    items: dict[str, Any] = dict(args.items() if isinstance(args, dict) else vars(args).items())

    for _group_name, group_def in CONFIG_PARAMETER_GROUPS.items():
        # --- Boolean toggle filtering ---
        enable_key = group_def.get("enable_key")
        if enable_key:
            enable_val = items.get(enable_key)
            if enable_val is False or str(enable_val).lower() == "false":
                children = _collect_all_children(group_def, items)
                children.discard(enable_key)
                for child in children:
                    items.pop(child, None)
                continue

        # --- Enum selector filtering ---
        selector_key = group_def.get("selector_key")
        children_by_value = group_def.get("children_by_value")
        if selector_key and children_by_value and selector_key in items:
            current_value = str(items[selector_key]).lower()
            relevant = set(children_by_value.get(current_value, []))
            all_enum_children: set[str] = set()
            for params_list in children_by_value.values():
                all_enum_children.update(params_list)
            irrelevant = all_enum_children - relevant
            for child in irrelevant:
                items.pop(child, None)

    return items


def log_args_config(
    args: Any,
    *,
    logger: logging.Logger | None = None,
    header: str | None = None,
    blank_line: bool = False,
) -> None:
    """Log configuration values from an args object or dict."""
    active_logger = logger or logging.getLogger(__name__)
    if header:
        active_logger.info(header)

    filtered = filter_disabled_children(args)

    for key, value in filtered.items():
        active_logger.info(f"{str(key):<20}  {value}")

    if blank_line:
        active_logger.info("")
