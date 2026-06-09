"""
CollectionSelector — map intent to Chroma collection whitelist.

Rule-based, deterministic, no LLM involved.

Each intent maps to one or more business-domain collections.
Only collections that exist in the Chroma store are returned.
If a collection doesn't exist yet, it is silently skipped.

Future: LLM may suggest additional collections within the whitelist,
but the backend will always enforce the whitelist boundary.
"""
from __future__ import annotations

from typing import Any

# ── Intent → allowed collections whitelist ──────────────────────────

INTENT_COLLECTION_MAP: dict[str, list[str]] = {
    "advisory": [
        "advisory_knowledge",
        "risk_knowledge",
        "education_knowledge",
        "compliance_knowledge",
    ],
    "financial_report": [
        "advisory_knowledge",
        "risk_knowledge",
        "compliance_knowledge",
    ],
    "risk_control": [
        "risk_knowledge",
        "compliance_knowledge",
        "advisory_knowledge",
    ],
    "compliance": [
        "compliance_knowledge",
        "risk_knowledge",
    ],
    "education": [
        "education_knowledge",
        "risk_knowledge",
        "compliance_knowledge",
    ],
}

# ── Fallback collection (used when intent is unknown) ───────────────

FALLBACK_COLLECTION = "education_knowledge"


# ── Collection metadata (for documentation and dynamic listing) ────

COLLECTION_META: dict[str, dict] = {
    "advisory_knowledge": {
        "display_name": "投顾知识库",
        "domain": "investment_advisory",
        "description": "资产配置原则、投资组合理论、定投策略等投顾通用知识",
        "agent_scope": ["investment_advisor"],
    },
    "compliance_knowledge": {
        "display_name": "合规知识库",
        "domain": "compliance",
        "description": "金融监管法规、合规要求、信息披露规范",
        "agent_scope": ["compliance", "investment_advisor"],
    },
    "education_knowledge": {
        "display_name": "金融科普知识库",
        "domain": "financial_education",
        "description": "基金、股票、债券等金融产品基础概念与入门知识",
        "agent_scope": ["education"],
    },
    "risk_knowledge": {
        "display_name": "风控知识库",
        "domain": "risk_control",
        "description": "风险模型、风险评估方法论、风险管理框架",
        "agent_scope": ["risk_control"],
    },
    "financial_report_knowledge": {
        "display_name": "财报分析知识库",
        "domain": "financial_report",
        "description": "财务报表分析、估值方法、会计准则解读",
        "agent_scope": ["financial_report"],
    },
    "market_knowledge": {
        "display_name": "市场知识库",
        "domain": "market",
        "description": "宏观经济指标、行业分析框架、市场周期理论",
        "agent_scope": ["investment_advisor", "education"],
    },
}


class CollectionSelector:
    """Select which Chroma collections to query for a given intent.

    Usage:
        selector = CollectionSelector()
        names = selector.get_collections("advisory")
        # → ["advisory_knowledge", "education_knowledge", "compliance_knowledge"]

        # With a live ChromaStore to filter by existence:
        names = selector.get_collections("advisory", chroma_store=store)
        # → only collections that actually exist in Chroma
    """

    def get_collections(
        self,
        intent: str,
        chroma_store: Any = None,
    ) -> list[str]:
        """Return allowed collection names for the given intent.

        Args:
            intent: One of the 5 intent labels (advisory, compliance, etc.).
            chroma_store: Optional ChromaStore instance. If provided, only
                          collections that exist in the store are returned.

        Returns:
            List of collection name strings, ordered by priority.
        """
        allowed = INTENT_COLLECTION_MAP.get(intent, [FALLBACK_COLLECTION])

        if chroma_store is None:
            return list(allowed)

        # Filter to only existing collections
        return [name for name in allowed if chroma_store.collection_exists(name)]

    def get_collection_meta(self, name: str) -> dict:
        """Return metadata dict for a given collection, or empty dict."""
        return COLLECTION_META.get(name, {})

    def list_all_collections(self) -> list[dict]:
        """Return all known collections with their metadata."""
        result: list[dict] = []
        for name, meta in COLLECTION_META.items():
            result.append({"name": name, **meta})
        return result

    @staticmethod
    def is_collection_allowed(name: str, intent: str) -> bool:
        """Check if a specific collection is in the whitelist for an intent."""
        allowed = INTENT_COLLECTION_MAP.get(intent, [FALLBACK_COLLECTION])
        return name in allowed

    # ── Multi-intent collection merging ─────────────────────────────

    def get_collections_for_intents(
        self,
        primary_intent: str,
        secondary_intents: list[str] | None = None,
        chroma_store: Any = None,
    ) -> list[str]:
        """Return merged collection list for primary + secondary intents.

        Order: primary's collections first, then secondary intents'
        collections appended in order.  Duplicates are removed while
        preserving first-occurrence order.

        Args:
            primary_intent: The primary intent label.
            secondary_intents: Additional intents whose collections should
                               be merged in.  May be None or empty.
            chroma_store: Optional ChromaStore.  If provided, only
                          collections that exist in the store are returned.

        Returns:
            Deduplicated, ordered list of collection name strings.
        """
        selected = self.get_collections(primary_intent, chroma_store=None)
        seen: set[str] = set(selected)

        for secondary in (secondary_intents or []):
            extras = self.get_collections(secondary, chroma_store=None)
            for coll in extras:
                if coll not in seen:
                    selected.append(coll)
                    seen.add(coll)

        # Filter by store existence if requested
        if chroma_store is not None:
            selected = [
                c for c in selected
                if chroma_store.collection_exists(c)
            ]

        return selected
