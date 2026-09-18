"""Configuration handling: layered YAML with dotted-key access."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .utils import find_by_designation, get_logger

log = get_logger("tttphot.config")

PACKAGE_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = PIPELINE_ROOT / "config" / "default.yaml"
# Merged automatically when present, after the defaults and before the target
# file. local.yaml is for machine-specific settings (paths); tns.yaml keeps
# credentials in their own gitignored file.
AUTO_CONFIGS = (PIPELINE_ROOT / "config" / "local.yaml",
                PIPELINE_ROOT / "config" / "tns.yaml")
# Keys never written out to config_used.yaml.
SECRET_KEYS = ("tns.api_key",)
INSTRUMENT_DIR = PIPELINE_ROOT / "config" / "instruments"
TARGET_DIR = PIPELINE_ROOT / "targets"


def deep_merge(base: Mapping, override: Mapping) -> dict:
    """Recursively merge *override* into *base* (neither is modified)."""
    out = copy.deepcopy(dict(base))
    for key, value in (override or {}).items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path) -> dict:
    path = Path(path)
    if not path.exists():
        if path == DEFAULT_CONFIG:
            raise FileNotFoundError(
                f"cannot find {path}.\n"
                "tttphot reads its configuration from the 'config' directory next to the\n"
                "package, so install it in place with 'pip install -e .' from the pipeline\n"
                "directory, or run it as 'python -m tttphot.cli' from there.")
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


class Config:
    """A merged configuration tree with ``cfg.get('a.b.c', default)`` access."""

    def __init__(self, data: Mapping):
        self._data = dict(data)
        self._warned_instruments: set = set()

    # -- construction -------------------------------------------------------
    @classmethod
    def load(
        cls,
        user_config: Optional[Path] = None,
        target: Optional[str] = None,
        overrides: Optional[Mapping] = None,
    ) -> "Config":
        # Layers, each overriding the one before: package defaults, any
        # auto-loaded local files, the target's own file, an explicit --config,
        # then command-line flags.
        data = load_yaml(DEFAULT_CONFIG)
        merged = [DEFAULT_CONFIG.name]

        for auto in AUTO_CONFIGS:
            if auto.exists():
                data = deep_merge(data, load_yaml(auto))
                merged.append(auto.name)

        if target:
            # SN2026abwp finds targets/AT2026abwp.yaml and the other way round:
            # TNS renames an AT to SN on classification.
            candidate = find_by_designation(TARGET_DIR, target)
            if candidate is not None:
                data = deep_merge(data, load_yaml(candidate))
                merged.append(f"targets/{candidate.name}")

        if user_config:
            data = deep_merge(data, load_yaml(user_config))
            merged.append(str(user_config))

        if overrides:
            data = deep_merge(data, overrides)

        data.setdefault("target", {})
        if target:
            data["target"].setdefault("name", target)

        data["_config_files"] = merged
        log.info("configuration: %s", " + ".join(merged))

        instrument = data.get("instrument")
        if instrument:
            inst_path = INSTRUMENT_DIR / f"{instrument}.yaml"
            data["instrument_config"] = load_yaml(inst_path)

        # One archive can hold frames from more than one camera. Load every
        # instrument named in the map so a frame can pick its own by header.
        configs = {}
        for header_value, name in (data.get("instruments") or {}).items():
            try:
                configs[str(header_value).strip().upper()] = load_yaml(INSTRUMENT_DIR / f"{name}.yaml")
            except FileNotFoundError:
                log.warning("instruments: %r maps to %s.yaml, which does not exist",
                            header_value, name)
        data["instrument_configs"] = configs

        # image.rebin overrides the per-instrument rebinning everywhere, so the
        # indexing (which sees only the instrument files) agrees with loading.
        rebin = (data.get("image") or {}).get("rebin")
        if rebin is not None:
            for inst in [data.get("instrument_config")] + list(configs.values()):
                if isinstance(inst, dict):
                    inst.setdefault("defaults", {})["rebin"] = rebin

        return cls(data)

    # -- access -------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted, None)
        if value is None:
            raise KeyError(f"required configuration key missing: {dotted}")
        return value

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def as_dict(self) -> dict:
        return copy.deepcopy(self._data)

    # -- instruments --------------------------------------------------------
    def instrument_for(self, header) -> dict:
        """The instrument definition matching a frame's INSTRUME header.

        Falls back to the configured default instrument, so an archive with a
        single camera needs no ``instruments`` map at all, and an unrecognised
        camera still runs -- loudly.
        """
        default = self.get("instrument_config", {}) or {}
        configs = self.get("instrument_configs", {}) or {}
        if not configs or header is None:
            return default
        name = str(header.get("INSTRUME", "") or "").strip().upper()
        if not name:
            return default
        found = configs.get(name)
        if found is not None:
            return found
        if name not in self._warned_instruments:
            self._warned_instruments.add(name)
            log.warning("instrument %r is not in the instruments map; "
                        "using %s. Add it to config/default.yaml if its gain, "
                        "pixel scale or saturation differ.",
                        name, default.get("name", self.get("instrument", "the default")))
        return default

    # -- resolved paths -----------------------------------------------------
    def resolve_path(self, value) -> Optional[Path]:
        """Expand a user-supplied path.

        ``~`` is expanded and a relative path is taken relative to the pipeline
        directory, not to wherever the command happened to be run from.
        """
        if value is None:
            return None
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (PIPELINE_ROOT / path)

    def _resolve(self, dotted: str, fallback: str) -> Path:
        value = self.get(dotted) or fallback
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (PIPELINE_ROOT / path).resolve()
        return path

    @property
    def data_root(self) -> Path:
        value = self.get("paths.data_root")
        if not value:
            raise KeyError("paths.data_root must be set (directory holding the FITS images)")
        return Path(value).expanduser()

    @property
    def work_root(self) -> Path:
        return self._resolve("paths.work_root", "runs")

    @property
    def cache_root(self) -> Path:
        return self._resolve("paths.cache_root", "cache")

    def dump(self, path) -> None:
        """Write the merged configuration, with any credentials redacted."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = copy.deepcopy(self._data)
        for dotted in SECRET_KEYS:
            parts = dotted.split(".")
            node = data
            for part in parts[:-1]:
                node = node.get(part) if isinstance(node, Mapping) else None
                if node is None:
                    break
            if isinstance(node, dict) and node.get(parts[-1]):
                node[parts[-1]] = "<redacted>"
        with open(path, "w") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
