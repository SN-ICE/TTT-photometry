"""Resolve a transient name to coordinates, and record them in the target file.

The Transient Name Server is the authority for a supernova's position, and it
is refined as better astrometry arrives, so looking the name up beats typing a
coordinate once and forgetting about it. TNS needs a bot API key; where none is
configured this falls back to SIMBAD, which is credential-free but coarser and
lags TNS for very recent objects.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from astropy.coordinates import SkyCoord
import astropy.units as u

from .utils import get_logger

log = get_logger("tttphot.nameresolve")

TNS_OBJECT_URL = "https://www.wis-tns.org/api/get/object"


@dataclass
class Resolution:
    coord: SkyCoord
    source: str                      # 'tns' | 'simbad'
    name: str = ""
    meta: dict = field(default_factory=dict)

    def describe(self) -> str:
        bits = [f"{self.source}: {self.coord.to_string('hmsdms', precision=3)}"]
        for key in ("object_type", "redshift", "hostname", "discoverydate"):
            value = self.meta.get(key)
            if value not in (None, "", "null"):
                bits.append(f"{key}={value}")
        return "  ".join(str(b) for b in bits)


def tns_objname(name: str) -> str:
    """'SN2026abgb', 'SN 2026abgb', 'AT2026abgb' -> '2026abgb'."""
    cleaned = str(name).strip()
    cleaned = re.sub(r"^(SN|AT|SNe)\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _tns_credentials(cfg) -> Optional[tuple]:
    api_key = (cfg.get("tns.api_key") if cfg else None) or os.environ.get("TNS_API_KEY")
    bot_id = (cfg.get("tns.bot_id") if cfg else None) or os.environ.get("TNS_BOT_ID")
    bot_name = (cfg.get("tns.bot_name") if cfg else None) or os.environ.get("TNS_BOT_NAME")
    if not api_key or not bot_id or not bot_name:
        return None
    return str(api_key), str(bot_id), str(bot_name)


def resolve_tns(name: str, cfg=None, timeout: float = 60.0) -> Optional[Resolution]:
    """Look the object up on the Transient Name Server."""
    import requests

    credentials = _tns_credentials(cfg)
    if credentials is None:
        log.info("no TNS credentials configured (tns.api_key / tns.bot_id / tns.bot_name, "
                 "or the TNS_API_KEY / TNS_BOT_ID / TNS_BOT_NAME environment variables)")
        return None
    api_key, bot_id, bot_name = credentials

    objname = tns_objname(name)
    marker = 'tns_marker{"tns_id": "%s", "type": "bot", "name": "%s"}' % (bot_id, bot_name)
    payload = {"objname": objname, "photometry": "0", "spectra": "0"}

    try:
        response = requests.post(
            TNS_OBJECT_URL,
            headers={"User-Agent": marker},
            data={"api_key": api_key, "data": json.dumps(payload)},
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        log.warning("TNS query failed: %s", exc)
        return None

    reply = (body or {}).get("data", {})
    if isinstance(reply, dict) and "reply" in reply:
        reply = reply["reply"]
    if not isinstance(reply, dict) or not reply:
        log.warning("TNS returned nothing for %r", objname)
        return None
    if "radeg" not in reply or reply.get("radeg") is None:
        log.warning("TNS reply for %r carries no coordinates (%s)",
                    objname, str(reply)[:200])
        return None

    try:
        coord = SkyCoord(float(reply["radeg"]) * u.deg, float(reply["decdeg"]) * u.deg)
    except Exception as exc:
        log.warning("could not read the TNS coordinates: %s", exc)
        return None

    object_type = reply.get("object_type") or {}
    meta = {
        "objname": reply.get("objname"),
        "prefix": reply.get("name_prefix"),
        "object_type": object_type.get("name") if isinstance(object_type, dict) else object_type,
        "redshift": reply.get("redshift"),
        "hostname": reply.get("hostname"),
        "host_redshift": reply.get("host_redshift"),
        "discoverydate": reply.get("discoverydate"),
        "discmagfilter": (reply.get("discmagfilter") or {}).get("name")
        if isinstance(reply.get("discmagfilter"), dict) else None,
    }
    log.info("TNS: %s%s at %s", meta.get("prefix") or "", meta.get("objname") or objname,
             coord.to_string("hmsdms", precision=3))
    return Resolution(coord=coord, source="tns", name=str(meta.get("objname") or objname),
                      meta={k: v for k, v in meta.items() if v not in (None, "", "null")})


def resolve_simbad(name: str, timeout: float = 60.0) -> Optional[Resolution]:
    """Fall back to SIMBAD. No credentials, but coarser and slower to update."""
    from astroquery.simbad import Simbad

    candidates = [str(name).strip()]
    stripped = tns_objname(name)
    if stripped != candidates[0]:
        candidates.append(f"SN {stripped}")
    candidates.append(f"SN{stripped}")

    errors = []
    for candidate in candidates:
        try:
            table = Simbad.query_object(candidate)
        except Exception as exc:
            errors.append(str(exc))
            log.debug("SIMBAD query for %r failed: %s", candidate, exc)
            continue
        if table is None or len(table) == 0:
            continue
        row = table[0]
        columns = {c.lower(): c for c in table.colnames}
        try:
            if "ra" in columns and "dec" in columns:
                ra_value, dec_value = row[columns["ra"]], row[columns["dec"]]
                # astroquery >= 0.4.8 returns degrees; older versions sexagesimal
                try:
                    coord = SkyCoord(float(ra_value) * u.deg, float(dec_value) * u.deg)
                except (TypeError, ValueError):
                    coord = SkyCoord(str(ra_value), str(dec_value), unit=(u.hourangle, u.deg))
            else:
                continue
        except Exception as exc:
            log.debug("could not read SIMBAD coordinates: %s", exc)
            continue
        log.info("SIMBAD: %s at %s", candidate, coord.to_string("hmsdms", precision=3))
        return Resolution(coord=coord, source="simbad", name=candidate)

    if errors:
        log.warning("SIMBAD could not be reached (%s)", errors[-1][:160])
    else:
        log.info("SIMBAD has no entry for %r", name)
    return None


def resolve(name: str, cfg=None, sources=None) -> Optional[Resolution]:
    """Try each configured source in turn."""
    if sources is None:
        sources = (cfg.get("target.resolve_sources") if cfg else None) or ["tns", "simbad"]
    for source in sources:
        try:
            if source == "tns":
                result = resolve_tns(name, cfg)
            elif source == "simbad":
                result = resolve_simbad(name)
            else:
                log.warning("unknown name resolver %r", source)
                continue
        except Exception as exc:
            log.warning("%s resolver failed: %s", source, exc)
            continue
        if result is not None:
            return result
    return None


# --------------------------------------------------------------------------
_TARGET_TEMPLATE = """# {target} -- written by 'tttphot resolve'.
target:
  name: {target}
  # {provenance}
  ra: {ra:.7f}
  dec: {dec:.7f}
"""


def target_file(target: str) -> Path:
    from .config import TARGET_DIR
    for candidate in (TARGET_DIR / f"{target}.yaml", TARGET_DIR / f"{target}.yml"):
        if candidate.exists():
            return candidate
    return TARGET_DIR / f"{target}.yaml"


def update_target_file(target: str, resolution: Resolution) -> tuple:
    """Write the resolved coordinates into targets/<target>.yaml.

    Edits only the ra/dec lines, line by line, so every comment and every other
    setting in the file survives -- a YAML round-trip would throw the comments
    away.
    """
    path = target_file(target)
    provenance = (f"position resolved from {resolution.source.upper()} on "
                  f"{date.today().isoformat()}")
    extra = ", ".join(f"{k}={v}" for k, v in resolution.meta.items()
                      if k in ("object_type", "redshift", "hostname"))
    if extra:
        provenance += f" ({extra})"

    ra_line = f"  ra: {resolution.coord.ra.deg:.7f}"
    dec_line = f"  dec: {resolution.coord.dec.deg:.7f}"

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_TARGET_TEMPLATE.format(
            target=target, provenance=provenance,
            ra=resolution.coord.ra.deg, dec=resolution.coord.dec.deg))
        log.info("created %s", path)
        return path, None

    lines = path.read_text().splitlines()
    out, previous = [], None
    in_target = False
    indent = "  "
    wrote_ra = wrote_dec = False

    for line in lines:
        if re.match(r"^target\s*:", line):
            in_target = True
            out.append(line)
            continue
        if in_target and line.strip() and not line.startswith((" ", "\t")):
            # left the target block without having seen ra/dec
            if not wrote_ra:
                out.append(f"{indent}# {provenance}")
                out.append(ra_line)
                out.append(dec_line)
                wrote_ra = wrote_dec = True
            in_target = False

        if in_target:
            match = re.match(r"^(\s+)(ra|dec)\s*:\s*(.*)$", line)
            if match:
                indent = match.group(1)
                key = match.group(2)
                if key == "ra":
                    previous = match.group(3).strip()
                    if out and out[-1].strip().startswith("# position resolved from"):
                        out.pop()
                    out.append(f"{indent}# {provenance}")
                    out.append(f"{indent}ra: {resolution.coord.ra.deg:.7f}")
                    wrote_ra = True
                else:
                    out.append(f"{indent}dec: {resolution.coord.dec.deg:.7f}")
                    wrote_dec = True
                continue
        out.append(line)

    if in_target and not wrote_ra:
        out.append(f"{indent}# {provenance}")
        out.append(f"{indent}ra: {resolution.coord.ra.deg:.7f}")
        out.append(f"{indent}dec: {resolution.coord.dec.deg:.7f}")

    path.write_text("\n".join(out) + "\n")
    log.info("updated %s", path)
    return path, previous
