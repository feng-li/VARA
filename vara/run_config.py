"""Small helpers for script run configuration JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def parse_int_list(value: Any) -> list[int]:
    """Parse comma-separated or JSON-list integer options."""
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_float_list(value: Any) -> list[float]:
    """Parse comma-separated or JSON-list float options."""
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_str_list(value: Any) -> list[str]:
    """Parse comma-separated or JSON-list string options."""
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value).split(",") if x.strip()]


def json_config_value(value: Any) -> Any:
    """Convert common script values to JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def select_config_values(values: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    """Select known run-config values and normalize them for JSON output."""
    return {
        key: json_config_value(values[key])
        for key in keys
        if key in values
    }


def load_config_json_defaults(path: str | Path, keys: Iterable[str]) -> Dict[str, Any]:
    """Load argparse defaults from a saved run-config JSON."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    key_set = set(keys)
    return {key: config[key] for key in key_set if key in config}


def save_run_config(path: str | Path, config: Mapping[str, Any]) -> None:
    """Write a run-config JSON using stable formatting."""
    Path(path).write_text(
        json.dumps(dict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
