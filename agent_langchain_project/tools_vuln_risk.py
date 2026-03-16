"""
Vulnerability scan tool.

This version optimizes OSV lookups using querybatch (one HTTP request for many deps),
with a fallback to per-package query if batch fails.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from .tools_osv import osv_query_batch, osv_query_package

logger = logging.getLogger("agent.vuln_risk")


@dataclass
class AffectedFinding:
    package: str
    version: str
    matched: bool
    reason: str


@dataclass
class RiskCard:
    query: str
    ecosystem: str
    summary: str
    severity: str
    vulns: List[Dict[str, Any]]
    affected_findings: List[AffectedFinding]
    evidence: List[Dict[str, Any]]
    notes: List[str]


def _normalize_pypi_name(name: str) -> str:
    # PEP 503 normalization (roughly): lower + replace underscores with hyphens
    return (name or "").strip().lower().replace("_", "-")


def _parse_requirements_file(path: str) -> List[Tuple[str, str]]:
    """
    Parse only strict pins: name==version

    Encoding handling (Windows-friendly):
    - try utf-8-sig
    - then utf-16 (handles BOM like FF FE / FE FF)
    - then gb18030 as fallback
    """
    deps: List[Tuple[str, str]] = []

    encodings_to_try = ["utf-8-sig", "utf-16", "gb18030"]
    last_err: Optional[Exception] = None
    content: Optional[str] = None

    for enc in encodings_to_try:
        try:
            with open(path, "r", encoding=enc) as f:
                content = f.read()
            last_err = None
            break
        except Exception as e:
            last_err = e
            continue

    if content is None:
        raise last_err or UnicodeDecodeError("unknown", b"", 0, 1, "cannot decode requirements file")

    for raw_line in content.splitlines():
        s = raw_line.strip()
        if not s or s.startswith("#"):
            continue
        if "==" not in s:
            continue
        name, ver = s.split("==", 1)
        name = name.strip()
        ver = ver.strip()
        if name and ver:
            deps.append((name, ver))

    return deps


def scan_vuln_risk_tool(arg_str: str) -> str:
    """
    Arg examples:
      requirements:/abs/path/requirements.txt
      requirements:/abs/path/requirements.txt --no-net
      requirements:/abs/path/requirements.txt --timeout=10
    """
    arg_str = (arg_str or "").strip()
    no_net = "--no-net" in arg_str

    # parse timeout
    timeout = 10
    for token in arg_str.split():
        if token.startswith("--timeout="):
            try:
                timeout = int(token.split("=", 1)[1])
            except Exception:
                pass

    if not arg_str.startswith("requirements:"):
        return json.dumps(
            {
                "error": "unsupported query; expected 'requirements:<path>'",
                "query": arg_str,
            },
            ensure_ascii=False,
            indent=2,
        )

    req_path = arg_str[len("requirements:") :].strip()
    # strip flags from path if user wrote "requirements:path --no-net"
    # keep everything before first " --"
    if " --" in req_path:
        req_path = req_path.split(" --", 1)[0].strip()

    notes: List[str] = []
    notes.append(f"[debug] cwd={os.getcwd()}")
    notes.append(f"[debug] requirements_path={req_path}")

    if not os.path.exists(req_path):
        return json.dumps(
            {
                "error": f"requirements file not found: {req_path}",
                "query": arg_str,
            },
            ensure_ascii=False,
            indent=2,
        )

    deps = _parse_requirements_file(req_path)
    notes.append(f"[debug] first_lines_repr={deps[:5]!r}")

    card = RiskCard(
        query=f"requirements:{req_path}" + (" --no-net" if no_net else ""),
        ecosystem="PyPI",
        summary=f"扫描依赖文件：{req_path}（解析到 {len(deps)} 条 name==version）",
        severity="",
        vulns=[],
        affected_findings=[],
        evidence=[],
        notes=notes,
    )

    # offline mode: do not query OSV
    if no_net:
        notes.append("[debug] --no-net enabled: skipped OSV queries")
        for name, ver in deps:
            card.affected_findings.append(
                AffectedFinding(package=name, version=ver, matched=False, reason="未联网查询（--no-net）")
            )
        return json.dumps(
            {
                **asdict(card),
                "affected_findings": [asdict(x) for x in card.affected_findings],
            },
            ensure_ascii=False,
            indent=2,
        )

    # online mode: batch query
    norm_map: List[Tuple[str, str, str]] = []  # (orig, ver, norm)
    batch_deps: List[Tuple[str, str]] = []
    for name, ver in deps:
        norm = _normalize_pypi_name(name)
        norm_map.append((name, ver, norm))
        batch_deps.append((norm, ver))

    notes.append(f"[debug] osv_queries={len(batch_deps)} source=requirements timeout={timeout}s")

    batch_result: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    try:
        import time
        t0 = time.time()
        batch_result = osv_query_batch(card.ecosystem, batch_deps, timeout=timeout)
        dt = time.time() - t0
        notes.append(f"[perf] osv_batch_elapsed_sec={dt:.3f}")
        notes.append("[debug] osv_query_mode=batch")
    except Exception as e:
        # fallback
        notes.append(f"[debug] osv_query_mode=fallback_single error={e}")
        logger.warning("osv.batch_failed_fallback_to_single", extra={"error": str(e)})

    for orig, ver, norm in norm_map:
        vulns: List[Dict[str, Any]] = []

        if batch_result:
            vulns = batch_result.get((card.ecosystem, norm, ver), [])
        else:
            vulns = osv_query_package(card.ecosystem, norm, ver, timeout=timeout)

        if not vulns:
            card.affected_findings.append(
                AffectedFinding(
                    package=orig,
                    version=ver,
                    matched=False,
                    reason=f"OSV vulns=0 (query_name={norm})",
                )
            )
            continue

        card.affected_findings.append(
            AffectedFinding(
                package=orig,
                version=ver,
                matched=True,
                reason=f"命中 {len(vulns)} 个漏洞 (query_name={norm})",
            )
        )

        # Keep a slim vulnerability list in card.vulns (avoid huge payload)
        for v in vulns:
            card.vulns.append(
                {
                    "id": v.get("id"),
                    "summary": v.get("summary"),
                    "details": (v.get("details") or "")[:200],
                    "aliases": v.get("aliases") or [],
                    "severity": v.get("severity") or [],
                    "references": v.get("references") or [],
                    "package": orig,
                    "version": ver,
                    "query_name": norm,
                }
            )

    # severity simple derivation
    if card.vulns:
        card.severity = "HAS_VULNS"
    else:
        card.severity = "UNKNOWN"
        notes.append("说明：OSV.dev 未对当前依赖版本返回漏洞，不代表绝对安全；建议结合更多数据源与运行环境评估。")

    return json.dumps(
        {
            **asdict(card),
            "affected_findings": [asdict(x) for x in card.affected_findings],
        },
        ensure_ascii=False,
        indent=2,
    )