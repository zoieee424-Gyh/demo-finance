# ADR 0001: MVP Architecture

## Status

Accepted

## Decision

MVP 使用 FastAPI + Vue3 的前后端分离架构。后端先建立清晰模块边界：API、Agents、RAG、Services、Schemas、Core。LLM 使用云端 API，但具体 Provider 由用户在接入阶段提供。向量库采用 Chroma，先实现基础语义检索，后续预留 BM25 混合检索扩展点。

## Reasons

- 企划中推荐技术栈包含 FastAPI、Vue3、LangChain、Chroma/Milvus；用户已明确向量库采用 Chroma。
- MVP 需要尽快形成可运行演示链路。
- 金融合规规则必须在核心流程中内置，而不是后期补丁。

## Consequences

- 首版更重视结构和可演示闭环，不急于接入真实行情或生产级向量库。
- 后续新增数据源、BM25 混合检索、LLM Provider 时必须走适配层。
