"""
OSV API helpers.

Key optimizations:
- Use /v1/querybatch to query multiple packages in one HTTP request
- Keep an in-memory cache for (ecosystem, name, version) -> vulns
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("agent.osv")

OSV_API_BASE = "https://api.osv.dev"
OSV_QUERY_URL = f"{OSV_API_BASE}/v1/query"
OSV_QUERYBATCH_URL = f"{OSV_API_BASE}/v1/querybatch"

# (ecosystem, normalized_name, version) -> vulns[]
_OSV_CACHE: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}


def _http_post_json(url: str, payload: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def osv_query_package(
    ecosystem: str,
    name: str,
    version: str,
    timeout: int = 10,
) -> List[Dict[str, Any]]:
    """
    Query OSV for a single (ecosystem, name, version).
    Returns vulns[] (possibly empty).
    """
    eco = ecosystem or "PyPI"
    key = (eco, name, version)
    if key in _OSV_CACHE:
        return _OSV_CACHE[key]

    payload = {
        "package": {"ecosystem": eco, "name": name},
        "version": version,
    }

    try:
        data = _http_post_json(OSV_QUERY_URL, payload, timeout=timeout)
        vulns = data.get("vulns") or []
        _OSV_CACHE[key] = vulns
        return vulns
    except urllib.error.URLError as e:
        logger.warning("osv_query_package.network_error", extra={"ecosystem": eco, "name": name, "version": version, "error": str(e)})
        return []
    except Exception as e:
        logger.exception("osv_query_package.unexpected_error", extra={"ecosystem": eco, "name": name, "version": version, "error": str(e)})
        return []


def osv_query_batch(
    ecosystem: str,
    deps: List[Tuple[str, str]],
    timeout: int = 10,
) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    """
    Batch query OSV for multiple (name, version) in one request.

    Input:
      deps: [(name, version), ...]
    Output:
      {(ecosystem, name, version): vulns_list, ...}

    Notes:
    - If some keys are already cached, they won't be included in the HTTP call.
    - If OSV batch call fails, raises exception to allow caller fallback strategy.
    """
    eco = ecosystem or "PyPI"

    queries: List[Dict[str, Any]] = []
    order_keys: List[Tuple[str, str, str]] = []

    for name, version in deps:
        k = (eco, name, version)
        if k in _OSV_CACHE:
            continue
        queries.append({"package": {"ecosystem": eco, "name": name}, "version": version})
        order_keys.append(k)

    # everything cached
    if not queries:
        return {(eco, n, v): _OSV_CACHE.get((eco, n, v), []) for (n, v) in deps}

    payload = {"queries": queries}

    data = _http_post_json(OSV_QUERYBATCH_URL, payload, timeout=timeout)
    results = data.get("results") or []

    # results order matches queries order
    for k, r in zip(order_keys, results):
        vulns = (r or {}).get("vulns") or []
        _OSV_CACHE[k] = vulns

    return {(eco, n, v): _OSV_CACHE.get((eco, n, v), []) for (n, v) in deps}


def osv_cache_clear() -> None:
    _OSV_CACHE.clear()