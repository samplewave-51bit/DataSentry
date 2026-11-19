"""
DataSentry Agents - Enterprise Safety Guardrails

Implements defense-in-depth guardrail layer between autonomous agents and enterprise tools:
1. Least-Privilege Role-Based Access Control (RBAC) via YAML policy
2. Risk Tiering & Mandatory Human-in-the-Loop (HITL) gates for high-risk tools
3. Infinite-Loop Breaker & Step Limiter
4. Append-Only JSONL Audit Logger
5. Prompt Injection Sanitizer for untrusted data values
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import yaml
from pydantic import BaseModel, Field


class RiskTier(str, Enum):
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    FORBIDDEN = "forbidden"


class DecisionStatus(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REQUIRE_HUMAN_APPROVAL = "REQUIRE_HUMAN_APPROVAL"


class GuardrailDecision(BaseModel):
    allowed: bool
    status: DecisionStatus
    tool_name: str
    risk_tier: RiskTier
    reason: str
    requires_human_approval: bool = False
    step_count: int = 1
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SanitizationResult(BaseModel):
    is_safe: bool
    detected_threats: List[str] = Field(default_factory=list)
    sanitized_text: str


class InputSanitizer:
    """Detects and neutralizes prompt injection embedded in data cells or payloads."""

    INJECTION_PATTERNS = [
        (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", "DIRECT_PROMPT_INJECTION"),
        (r"system\s*override", "SYSTEM_OVERRIDE"),
        (r"developer\s+mode", "JAILBREAK_ATTEMPT"),
        (r"you\s+are\s+now\s+(in\s+)?(DAN|unrestricted)", "JAILBREAK_ATTEMPT"),
        (r"drop\s+table", "SQL_INJECTION_DESTRUCTIVE"),
        (r"delete\s+from\s+[a-zA-Z0-9_]+", "SQL_INJECTION_DESTRUCTIVE"),
        (r"delete\s+(the\s+)?dataset", "DATASET_DELETION_ATTEMPT"),
        (r"rm\s+-rf", "COMMAND_INJECTION_DESTRUCTIVE"),
        (r"truncate\s+table", "SQL_INJECTION_DESTRUCTIVE"),
        (r"grant\s+admin|escalate\s+privileges", "PRIVILEGE_ESCALATION"),
        (r"bypass\s+(all\s+)?guardrails?", "GUARDRAIL_BYPASS_ATTEMPT"),
        (r"</system>|<\|im_start\|>|\[SYSTEM\]", "DELIMITER_INJECTION"),
    ]

    @classmethod
    def sanitize(cls, text: str) -> SanitizationResult:
        detected: List[str] = []
        sanitized = text

        for pattern, threat_type in cls.INJECTION_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                detected.append(threat_type)
                # Redact the malicious phrase
                sanitized = re.sub(pattern, "[REDACTED_ADVERSARIAL_INJECTION]", sanitized, flags=re.IGNORECASE)

        is_safe = len(detected) == 0
        return SanitizationResult(
            is_safe=is_safe,
            detected_threats=detected,
            sanitized_text=sanitized,
        )


class AuditLogger:
    """Thread-safe append-only JSONL logger for complete operational compliance."""

    def __init__(self, log_path: str = "logs/audit_log.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def log(self, entry: Dict[str, Any]):
        entry_with_time = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **entry,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry_with_time) + "\n")


class SecurityPolicy:
    """Parses and manages policy rules from YAML."""

    def __init__(self, policy_path: Optional[str] = None):
        self.policy_path = policy_path or os.getenv("DATASENTRY_POLICY_PATH", "policies/security_policy.yaml")
        self.raw_config = self._load()

        settings = self.raw_config.get("settings", {})
        self.dry_run_default: bool = settings.get("dry_run_default", True)
        self.max_execution_steps: int = settings.get("max_execution_steps", 10)
        self.rate_limit_per_minute: int = settings.get("rate_limit_per_minute", 20)
        self.budget_cap_usd: float = settings.get("budget_cap_usd", 1.00)
        self.audit_log_path: str = settings.get("audit_log_path", "logs/audit_log.jsonl")

        self.risk_tiers: Dict[str, Any] = self.raw_config.get("risk_tiers", {})
        self.roles: Dict[str, Any] = self.raw_config.get("roles", {})
        self.forbidden_tools: Set[str] = set(self.raw_config.get("forbidden_tools", []))

    @classmethod
    def from_yaml(cls, path: str) -> SecurityPolicy:
        """Constructs SecurityPolicy from a YAML configuration file."""
        return cls(policy_path=path)

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.policy_path):
            with open(self.policy_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        # Sensible defaults if policy file missing
        return {
            "settings": {"dry_run_default": True, "max_execution_steps": 10, "budget_cap_usd": 1.0},
            "risk_tiers": {
                "low_risk": {"tools": ["pass_batch", "request_schema_fix", "log_audit_event"]},
                "medium_risk": {"tools": ["quarantine_batch"]},
                "high_risk": {"tools": ["trigger_retraining", "delete_dataset", "overwrite_reference"]},
            },
            "roles": {
                "remediation_agent": {
                    "allowed_tools": ["pass_batch", "quarantine_batch", "request_schema_fix", "trigger_retraining"]
                }
            },
            "forbidden_tools": ["delete_dataset", "drop_database"],
        }

    def get_tool_risk_tier(self, tool_name: str) -> RiskTier:
        if tool_name in self.forbidden_tools:
            return RiskTier.FORBIDDEN

        for tier_name, tier_info in self.risk_tiers.items():
            if tool_name in tier_info.get("tools", []):
                if tier_name == "high_risk":
                    return RiskTier.HIGH_RISK
                if tier_name == "medium_risk":
                    return RiskTier.MEDIUM_RISK
                return RiskTier.LOW_RISK

        # Unknown tools default to HIGH_RISK for fail-safe security
        return RiskTier.HIGH_RISK


class GuardrailEngine:
    """Enforces enterprise safety guardrails on all agent actions."""

    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self.logger = AuditLogger(log_path=self.policy.audit_log_path)
        self.sanitizer = InputSanitizer()

    def validate_action(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        agent_role: str = "remediation_agent",
        human_approved: bool = False,
        step_count: int = 1,
    ) -> GuardrailDecision:
        """Evaluates tool invocation against all active security constraints."""

        # 1. Step Budget / Loop Breaker
        if step_count > self.policy.max_execution_steps:
            dec = GuardrailDecision(
                allowed=False,
                status=DecisionStatus.BLOCK,
                tool_name=tool_name,
                risk_tier=RiskTier.HIGH_RISK,
                reason=f"BLOCKED: Maximum step budget exceeded ({step_count} > {self.policy.max_execution_steps}). Loop breaker engaged.",
                step_count=step_count,
            )
            self._log_decision(dec, arguments)
            return dec

        # 2. Forbidden Tools
        risk_tier = self.policy.get_tool_risk_tier(tool_name)
        if risk_tier == RiskTier.FORBIDDEN:
            dec = GuardrailDecision(
                allowed=False,
                status=DecisionStatus.BLOCK,
                tool_name=tool_name,
                risk_tier=RiskTier.FORBIDDEN,
                reason=f"BLOCKED: Tool '{tool_name}' is permanently forbidden by security policy.",
                step_count=step_count,
            )
            self._log_decision(dec, arguments)
            return dec

        # 3. Role-Based Access Control (RBAC)
        allowed_tools = self.policy.roles.get(agent_role, {}).get("allowed_tools", [])
        if tool_name not in allowed_tools:
            dec = GuardrailDecision(
                allowed=False,
                status=DecisionStatus.BLOCK,
                tool_name=tool_name,
                risk_tier=risk_tier,
                reason=f"BLOCKED: Role '{agent_role}' does not possess permission for tool '{tool_name}'.",
                step_count=step_count,
            )
            self._log_decision(dec, arguments)
            return dec

        # 4. Input Sanitization against Prompt Injection in tool args
        for k, v in arguments.items():
            if isinstance(v, str):
                san_res = self.sanitizer.sanitize(v)
                if not san_res.is_safe:
                    dec = GuardrailDecision(
                        allowed=False,
                        status=DecisionStatus.BLOCK,
                        tool_name=tool_name,
                        risk_tier=risk_tier,
                        reason=f"BLOCKED: Prompt injection attack detected in argument '{k}' ({', '.join(san_res.detected_threats)}).",
                        step_count=step_count,
                    )
                    self._log_decision(dec, arguments)
                    return dec

        # 5. Risk Tier & Human Approval Gate (HITL)
        if risk_tier == RiskTier.HIGH_RISK:
            if not human_approved:
                dec = GuardrailDecision(
                    allowed=False,
                    status=DecisionStatus.REQUIRE_HUMAN_APPROVAL,
                    tool_name=tool_name,
                    risk_tier=RiskTier.HIGH_RISK,
                    reason=f"HOLD: High-risk action '{tool_name}' requires explicit human approval token.",
                    requires_human_approval=True,
                    step_count=step_count,
                )
                self._log_decision(dec, arguments)
                return dec

        # All checks passed
        dec = GuardrailDecision(
            allowed=True,
            status=DecisionStatus.ALLOW,
            tool_name=tool_name,
            risk_tier=risk_tier,
            reason=f"Action '{tool_name}' authorized under role '{agent_role}'.",
            step_count=step_count,
        )
        self._log_decision(dec, arguments)
        return dec

    def _log_decision(self, dec: GuardrailDecision, args: Dict[str, Any]):
        self.logger.log({
            "action": "GUARDRAIL_EVALUATION",
            "tool_name": dec.tool_name,
            "status": dec.status.value,
            "allowed": dec.allowed,
            "risk_tier": dec.risk_tier.value,
            "reason": dec.reason,
            "step_count": dec.step_count,
            "arguments_sample": {k: str(v)[:80] for k, v in args.items()},
        })
