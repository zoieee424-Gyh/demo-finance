# RAG Retrieval Quality Evaluation Report

**Generated:** 2026-06-06 17:57:05
**Python:** `D:\AI\soft\conda\envs\python3.11\python.exe`
**Top-K:** 5

## 1. Execution Environment

| Item | Value |
|------|-------|
| Timestamp | 2026-06-06 17:57:05 |
| Python | `D:\AI\soft\conda\envs\python3.11\python.exe` |
| Chroma collections | 4 |
| Chroma vectors | 44 |
| Eval cases | 40 |
| Top-K | 5 |
| Elapsed | 58.1s |

## 2. Overall Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Cases | 40 | — | — |
| Intent Accuracy | 95.00% | ≥ 0.90 | ✅ PASS |
| Collection Coverage | 100.00% | — | — |
| Top-1 Accuracy | 70.00% | ≥ 70% | ✅ PASS |
| Recall@5 | 92.50% | ≥ 85% | ✅ PASS |
| MRR | 0.7863 | — | — |

**Overall Verdict: ✅ PASS**

## 3. Metrics by Expected Intent

| Intent | Cases | Intent Acc | Coll Cov | Top-1 Acc | Recall@K | MRR |
|--------|-------|------------|----------|-----------|----------|-----|
| advisory | 13 | 92.31% | 100.00% | 76.92% | 92.31% | 0.8231 |
| compliance | 9 | 88.89% | 100.00% | 44.44% | 100.00% | 0.6574 |
| education | 9 | 100.00% | 100.00% | 88.89% | 88.89% | 0.8889 |
| risk_control | 9 | 100.00% | 100.00% | 66.67% | 88.89% | 0.7593 |

## 4. Metrics by Difficulty

| Difficulty | Cases | Top-1 Acc | Recall@K | MRR |
|------------|-------|-----------|----------|-----|
| easy | 13 | 84.62% | 100.00% | 0.9038 |
| hard | 11 | 54.55% | 90.91% | 0.6848 |
| medium | 16 | 68.75% | 87.50% | 0.7604 |

## 5. Top Failures

### 1. [CMP-003] AI给出的投资建议靠谱吗？还需要自己判断吗？

| Field | Value |
|-------|-------|
| Expected Intent | `compliance` |
| Actual Intent | `advisory` |
| Secondary Intents | `compliance` |
| Selected Collections | `advisory_knowledge`, `education_knowledge`, `compliance_knowledge` |
| Expected Titles | 投资建议合规边界, 风险提示标准表达 |
| Difficulty | medium |
| Failure Types | intent_error, top1_miss |

**Top search results:**

| Rank | Title | Score | Collection |
|------|-------|-------|------------|
| 1 | 投资顾问合规边界与风险揭示 | 0.6403 | compliance_knowledge |
| 2 | 投资顾问合规边界与风险揭示 | 0.6074 | compliance_knowledge |
| 3 | 投资建议合规边界 | 0.5972 | compliance_knowledge |

**Diagnosis:**

- ❌ Intent misrouted: expected `compliance`, got `advisory`.
- **Possible fix:** Add/boost keywords for `compliance` or check keyword overlap with `advisory`.
- ⚠️ Top-1 miss but recall hit. Ranking issue.
- **Possible fix:** Improve embedding quality or add reranker.

### 2. [XDM-002] 定投基金有风险吗？合规吗？

| Field | Value |
|-------|-------|
| Expected Intent | `advisory` |
| Actual Intent | `compliance` |
| Secondary Intents | `advisory`, `risk_control` |
| Selected Collections | `compliance_knowledge`, `advisory_knowledge`, `education_knowledge`, `risk_knowledge` |
| Expected Titles | 基金定投常见原则, 投资建议合规边界 |
| Difficulty | medium |
| Failure Types | intent_error |

**Top search results:**

| Rank | Title | Score | Collection |
|------|-------|-------|------------|
| 1 | 基金定投常见原则 | 0.6473 | advisory_knowledge |
| 2 | 基金定投常见原则 | 0.6242 | advisory_knowledge |
| 3 | 投资顾问合规边界与风险揭示 | 0.5966 | compliance_knowledge |

**Diagnosis:**

- ❌ Intent misrouted: expected `advisory`, got `compliance`.
- **Possible fix:** Add/boost keywords for `advisory` or check keyword overlap with `compliance`.

### 3. [ADV-006] 保守型投资者适合买什么？

| Field | Value |
|-------|-------|
| Expected Intent | `advisory` |
| Actual Intent | `advisory` |
| Secondary Intents | — |
| Selected Collections | `advisory_knowledge`, `education_knowledge`, `compliance_knowledge` |
| Expected Titles | 风险等级与资产类别匹配, 资产配置基础原则 |
| Difficulty | medium |
| Failure Types | top1_miss, recall_miss |

**Top search results:**

| Rank | Title | Score | Collection |
|------|-------|-------|------------|
| 1 | 资产配置模型说明 | 0.5571 | advisory_knowledge |
| 2 | 资产配置模型说明 | 0.5449 | advisory_knowledge |
| 3 | 长期投资与复利效应 | 0.5404 | advisory_knowledge |

**Diagnosis:**

- ❌ No expected title found in top-5.
- **Possible fix:** Add knowledge document covering this topic, or adjust collection selector to include more collections.

### 4. [RSK-007] 最大回撤是什么？对我的投资有什么意义？

| Field | Value |
|-------|-------|
| Expected Intent | `risk_control` |
| Actual Intent | `risk_control` |
| Secondary Intents | `education` |
| Selected Collections | `risk_knowledge`, `compliance_knowledge`, `education_knowledge` |
| Expected Titles | 权益类资产波动风险 |
| Difficulty | hard |
| Failure Types | top1_miss, recall_miss |

**Top search results:**

| Rank | Title | Score | Collection |
|------|-------|-------|------------|
| 1 | 投资顾问合规边界与风险揭示 | 0.5056 | compliance_knowledge |
| 2 | 投资顾问合规边界与风险揭示 | 0.4935 | compliance_knowledge |
| 3 | 持仓集中度风险 | 0.4874 | risk_knowledge |

**Diagnosis:**

- ❌ No expected title found in top-5.
- **Possible fix:** Add knowledge document covering this topic, or adjust collection selector to include more collections.

### 5. [XDM-003] 我想学理财，有什么风险需要注意？入门该看什么？

| Field | Value |
|-------|-------|
| Expected Intent | `education` |
| Actual Intent | `education` |
| Secondary Intents | `advisory`, `risk_control` |
| Selected Collections | `education_knowledge`, `advisory_knowledge`, `compliance_knowledge`, `risk_knowledge` |
| Expected Titles | 什么是基金, 持仓集中度风险 |
| Difficulty | medium |
| Failure Types | top1_miss, recall_miss |

**Top search results:**

| Rank | Title | Score | Collection |
|------|-------|-------|------------|
| 1 | 长期投资与复利效应 | 0.5605 | advisory_knowledge |
| 2 | 投资者风险画像与适当性规则 | 0.5513 | risk_knowledge |
| 3 | 市场风险与系统性风险 | 0.5502 | risk_knowledge |

**Diagnosis:**

- ❌ No expected title found in top-5.
- **Possible fix:** Add knowledge document covering this topic, or adjust collection selector to include more collections.

### 6. [ADV-008] 再平衡是什么意思？怎么做再平衡？

| Field | Value |
|-------|-------|
| Expected Intent | `advisory` |
| Actual Intent | `advisory` |
| Secondary Intents | `education` |
| Selected Collections | `advisory_knowledge`, `education_knowledge`, `compliance_knowledge` |
| Expected Titles | 资产配置基础原则 |
| Difficulty | hard |
| Failure Types | top1_miss |

**Top search results:**

| Rank | Title | Score | Collection |
|------|-------|-------|------------|
| 1 | 资产配置模型说明 | 0.7325 | advisory_knowledge |
| 2 | 资产配置模型说明 | 0.6582 | advisory_knowledge |
| 3 | 投资顾问合规边界与风险揭示 | 0.4976 | compliance_knowledge |

**Diagnosis:**

- ⚠️ Top-1 miss but recall hit. Ranking issue.
- **Possible fix:** Improve embedding quality or add reranker.

### 7. [CMP-002] 能给我推荐几只股票吗？

| Field | Value |
|-------|-------|
| Expected Intent | `compliance` |
| Actual Intent | `compliance` |
| Secondary Intents | — |
| Selected Collections | `compliance_knowledge` |
| Expected Titles | 投资建议合规边界 |
| Difficulty | easy |
| Failure Types | top1_miss |

**Top search results:**

| Rank | Title | Score | Collection |
|------|-------|-------|------------|
| 1 | 投资顾问合规边界与风险揭示 | 0.4849 | compliance_knowledge |
| 2 | 投资建议合规边界 | 0.4511 | compliance_knowledge |
| 3 | 投资顾问合规边界与风险揭示 | 0.4416 | compliance_knowledge |

**Diagnosis:**

- ⚠️ Top-1 miss but recall hit. Ranking issue.
- **Possible fix:** Improve embedding quality or add reranker.

### 8. [CMP-005] 买基金之前，销售机构需要了解我的哪些信息？

| Field | Value |
|-------|-------|
| Expected Intent | `compliance` |
| Actual Intent | `compliance` |
| Secondary Intents | — |
| Selected Collections | `compliance_knowledge` |
| Expected Titles | 投资者适当性管理规定概要 |
| Difficulty | medium |
| Failure Types | top1_miss |

**Top search results:**

| Rank | Title | Score | Collection |
|------|-------|-------|------------|
| 1 | 投资顾问合规边界与风险揭示 | 0.5954 | compliance_knowledge |
| 2 | 投资顾问合规边界与风险揭示 | 0.5587 | compliance_knowledge |
| 3 | 投资者适当性管理规定概要 | 0.5424 | compliance_knowledge |

**Diagnosis:**

- ⚠️ Top-1 miss but recall hit. Ranking issue.
- **Possible fix:** Improve embedding quality or add reranker.

### 9. [CMP-007] 收益保证年化8%的产品能买吗？

| Field | Value |
|-------|-------|
| Expected Intent | `compliance` |
| Actual Intent | `compliance` |
| Secondary Intents | — |
| Selected Collections | `compliance_knowledge` |
| Expected Titles | 投资建议合规边界 |
| Difficulty | easy |
| Failure Types | top1_miss |

**Top search results:**

| Rank | Title | Score | Collection |
|------|-------|-------|------------|
| 1 | 投资顾问合规边界与风险揭示 | 0.6394 | compliance_knowledge |
| 2 | 投资顾问合规边界与风险揭示 | 0.5410 | compliance_knowledge |
| 3 | 投资顾问合规边界与风险揭示 | 0.5188 | compliance_knowledge |

**Diagnosis:**

- ⚠️ Top-1 miss but recall hit. Ranking issue.
- **Possible fix:** Improve embedding quality or add reranker.

### 10. [RSK-003] 股票市场大跌怎么办？

| Field | Value |
|-------|-------|
| Expected Intent | `risk_control` |
| Actual Intent | `risk_control` |
| Secondary Intents | — |
| Selected Collections | `risk_knowledge`, `compliance_knowledge` |
| Expected Titles | 权益类资产波动风险, 市场风险与系统性风险 |
| Difficulty | medium |
| Failure Types | top1_miss |

**Top search results:**

| Rank | Title | Score | Collection |
|------|-------|-------|------------|
| 1 | 投资者保护制度框架 | 0.4263 | compliance_knowledge |
| 2 | 市场风险与系统性风险 | 0.4050 | risk_knowledge |
| 3 | 投资顾问合规边界与风险揭示 | 0.3788 | compliance_knowledge |

**Diagnosis:**

- ⚠️ Top-1 miss but recall hit. Ranking issue.
- **Possible fix:** Improve embedding quality or add reranker.

### 11. [XDM-001] 保守型投资者能买债券基金吗？风险大吗？

| Field | Value |
|-------|-------|
| Expected Intent | `advisory` |
| Actual Intent | `advisory` |
| Secondary Intents | `risk_control` |
| Selected Collections | `advisory_knowledge`, `education_knowledge`, `compliance_knowledge`, `risk_knowledge` |
| Expected Titles | 风险等级与资产类别匹配, 什么是债券 |
| Difficulty | hard |
| Failure Types | top1_miss |

**Top search results:**

| Rank | Title | Score | Collection |
|------|-------|-------|------------|
| 1 | 信用风险基础知识 | 0.5452 | risk_knowledge |
| 2 | 什么是债券 | 0.5366 | education_knowledge |
| 3 | 信用风险基础知识 | 0.5231 | risk_knowledge |

**Diagnosis:**

- ⚠️ Top-1 miss but recall hit. Ranking issue.
- **Possible fix:** Improve embedding quality or add reranker.

### 12. [XDM-004] 投资顾问给我推荐高收益产品，这合规吗？

| Field | Value |
|-------|-------|
| Expected Intent | `compliance` |
| Actual Intent | `compliance` |
| Secondary Intents | — |
| Selected Collections | `compliance_knowledge` |
| Expected Titles | 投资建议合规边界 |
| Difficulty | hard |
| Failure Types | top1_miss |

**Top search results:**

| Rank | Title | Score | Collection |
|------|-------|-------|------------|
| 1 | 投资顾问合规边界与风险揭示 | 0.6774 | compliance_knowledge |
| 2 | 投资建议合规边界 | 0.6757 | compliance_knowledge |
| 3 | 投资顾问合规边界与风险揭示 | 0.6391 | compliance_knowledge |

**Diagnosis:**

- ⚠️ Top-1 miss but recall hit. Ranking issue.
- **Possible fix:** Improve embedding quality or add reranker.

### 13. [XDM-006] 市场下跌时应该卖出止损吗？这算不算追涨杀跌？

| Field | Value |
|-------|-------|
| Expected Intent | `risk_control` |
| Actual Intent | `risk_control` |
| Secondary Intents | `advisory` |
| Selected Collections | `risk_knowledge`, `compliance_knowledge`, `advisory_knowledge`, `education_knowledge` |
| Expected Titles | 权益类资产波动风险, 长期投资与复利效应 |
| Difficulty | hard |
| Failure Types | top1_miss |

**Top search results:**

| Rank | Title | Score | Collection |
|------|-------|-------|------------|
| 1 | 基金定投常见原则 | 0.4521 | advisory_knowledge |
| 2 | 市场风险与系统性风险 | 0.4244 | risk_knowledge |
| 3 | 长期投资与复利效应 | 0.4237 | advisory_knowledge |

**Diagnosis:**

- ⚠️ Top-1 miss but recall hit. Ranking issue.
- **Possible fix:** Improve embedding quality or add reranker.

## 6. Low Score Warnings

Cases where top-1 result score < 0.45 (low confidence):

| ID | Query | Top-1 Title | Score | Hit? |
|----|-------|-------------|-------|------|
| ADV-005 | 我月薪一万，想买房，怎么理财？ | 资产配置基础原则 | 0.4166 | ✅ |
| EDU-005 | 我是小白，完全不懂投资，从哪开始学？ | 什么是基金 | 0.3940 | ✅ |
| EDU-007 | 市盈率市净率怎么看？ | 基金净值与估值基础 | 0.4264 | ✅ |
| RSK-003 | 股票市场大跌怎么办？ | 投资者保护制度框架 | 0.4263 | ❌ |

**4 cases** have low top-1 scores. These may indicate weak semantic matches or knowledge gaps.

## 7. Overall Assessment

**Verdict: ✅ PASS**

- Intent Accuracy **meets target** (95.00% ≥ 0.90).
- Top-1 Accuracy **meets target** (70.00% ≥ 70%).
- Recall@5 **meets target** (92.50% ≥ 85%).

## 8. Next-Step Recommendations

Priority-ordered recommendations based on evaluation results:

1. **Fix IntentRouter keywords** — 2 misrouted cases. Review keyword overlap for affected intent pairs: advisory→compliance, compliance→advisory.

2. **Fill knowledge gaps** — 3 cases with zero recall. Missing topics include: 保守型投资者适合买什么？, 最大回撤是什么？对我的投资有什么意义？, 我想学理财，有什么风险需要注意？入门该看什么？.

3. **Consider reranker or BM25 hybrid** — 9 cases have correct recall but wrong top-1. Ranking quality could be improved.

4. **Expand knowledge base** — current 31 vectors across 4 collections is relatively small. Add 2-3 documents per collection to improve coverage.

5. **Review hard case failures** — 5 hard cases failed. These may require specialized documents or hybrid retrieval.

---
*Report generated by `backend/scripts/evaluate_rag.py` at 2026-06-06 17:57:05*
