# API Reference

## Health

`GET /api/health`

Response:

```json
{
  "status": "ok",
  "service": "financial-agentic-rag"
}
```

## Consultation

`POST /api/consultations`

Request:

```json
{
  "question": "我月薪1万，风险厌恶，3年后买房，该如何配置资产？",
  "user_profile": {}
}
```

Response:

```json
{
  "intent": "advisory",
  "agent": "investment_advisor",
  "answer": "...",
  "risk_notice": "投资有风险，本回答仅作信息参考，不构成投资决策依据。",
  "sources": [
    {
      "title": "资产配置基础原则",
      "source_type": "investment_knowledge",
      "url": null,
      "confidence": 0.85
    }
  ],
  "warnings": []
}
```

### Pipeline

1. `IntentRouter` classifies the question → intent label.
2. `RagPlanner` builds a list of structured retrieval queries (semantic-only for MVP).
3. `KnowledgeRetriever` returns mock sources keyed by intent.
4. The matching `FinancialAgent` generates a placeholder answer with the sources.
5. `ComplianceGuard.review()` scans the answer against 20+ compliance rules and appends `warnings` entries for any violations (荐股、涨跌预测、收益承诺、替用户决策等). Investment-related answers always receive a `risk_notice`.

### Intent Labels

| Label | Meaning |
|-------|---------|
| `advisory` | 智能投顾 |
| `financial_report` | 财报分析 |
| `risk_control` | 风控审查 |
| `compliance` | 监管合规 |
| `education` | 金融科普 |

### Compliance Warnings

When the ComplianceGuard detects a violation, a warning string is appended to `warnings`. Warning messages are prefixed with `[违规]` for violation severity or `[风险]` for advisory warnings.
