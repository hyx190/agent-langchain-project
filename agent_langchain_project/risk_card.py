from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List


@dataclass
class Evidence:
    url: str
    snippet: str = ""
    source: str = "OSV"


@dataclass
class AffectedFinding:
    package: str
    version: str
    matched: bool
    reason: str


@dataclass
class RiskCard:
    query: str
    ecosystem: str = ""
    summary: str = ""
    severity: str = ""  # LOW/MEDIUM/HIGH/CRITICAL/UNKNOWN
    vulns: List[Dict[str, Any]] = field(default_factory=list)  # simplified vuln list
    affected_findings: List[AffectedFinding] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def to_text(self, max_vulns: int = 6) -> str:
        lines: List[str] = []
        lines.append(f"[RiskCard] query={self.query} ecosystem={self.ecosystem}")
        if self.summary:
            lines.append(f"summary: {self.summary}")
        lines.append(f"severity: {self.severity or 'UNKNOWN'}")

        if self.affected_findings:
            lines.append("affected_findings:")
            for f in self.affected_findings[:30]:
                flag = "HIT" if f.matched else "OK"
                lines.append(f"  - {flag} {f.package}=={f.version} ({f.reason})")

        if self.vulns:
            lines.append("vulns:")
            for v in self.vulns[:max_vulns]:
                vid = v.get("id", "")
                summ = v.get("summary", "")
                lines.append(f"  - {vid}: {summ}")

        if self.evidence:
            lines.append("evidence:")
            for e in self.evidence[:8]:
                snip = (e.snippet or "").strip().replace("\n", " ")
                if len(snip) > 160:
                    snip = snip[:160] + "..."
                lines.append(f"  - {e.source}: {e.url}")
                if snip:
                    lines.append(f"    snippet: {snip}")

        if self.notes:
            lines.append("notes:")
            for n in self.notes[:10]:
                lines.append(f"  - {n}")

        return "\n".join(lines)