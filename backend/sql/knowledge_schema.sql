-- ============================================================================
-- Knowledge Base Metadata Schema (MySQL 8 compatible)
--
-- Design:
--   MySQL stores structured metadata about collections, documents, and chunks.
--   Chroma stores the actual vector embeddings and text content.
--   The two are linked via collection_name + chroma_id.
--
-- This file is the authoritative DDL — run it once to initialize the metadata
-- database. Tables use InnoDB, utf8mb4, and standard audit columns.
-- ============================================================================

CREATE DATABASE IF NOT EXISTS fin_agent_knowledge
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE fin_agent_knowledge;

-- ============================================================================
-- knowledge_collections
-- ============================================================================

CREATE TABLE IF NOT EXISTS knowledge_collections (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(128)  NOT NULL UNIQUE
        COMMENT 'Chroma collection name, e.g. advisory_knowledge',
    display_name    VARCHAR(256)  NOT NULL
        COMMENT 'Human-readable Chinese name',
    domain          VARCHAR(64)   NOT NULL
        COMMENT 'Business domain, e.g. advisory / compliance / risk',
    description     TEXT
        COMMENT 'What this collection covers',
    agent_scope     JSON
        COMMENT 'List of agent names that may query this collection',
    retrieval_policy VARCHAR(64)  NOT NULL DEFAULT 'semantic'
        COMMENT 'Retrieval strategy: semantic (current), hybrid (future)',
    enabled         TINYINT(1)    NOT NULL DEFAULT 1,
    created_at      DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_collection_domain (domain),
    INDEX idx_collection_enabled (enabled)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Business-domain collections registered in Chroma';


-- ============================================================================
-- knowledge_documents
-- ============================================================================

CREATE TABLE IF NOT EXISTS knowledge_documents (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    collection_name VARCHAR(128)  NOT NULL
        COMMENT 'FK to knowledge_collections.name',
    title           VARCHAR(512)  NOT NULL
        COMMENT 'Document title',
    source_type     VARCHAR(64)   NOT NULL DEFAULT 'knowledge_base'
        COMMENT 'e.g. investment_knowledge, regulations, financial_education',
    authority       VARCHAR(128)
        COMMENT 'Authority level: high / medium / low',
    url             VARCHAR(2048)
        COMMENT 'Original URL (if applicable)',
    file_path       VARCHAR(1024)
        COMMENT 'Local file path of the ingested document',
    version         VARCHAR(32)   NOT NULL DEFAULT '1.0',
    status          VARCHAR(32)   NOT NULL DEFAULT 'active'
        COMMENT 'active / archived / deprecated',
    tags            JSON
        COMMENT 'Flexible tags for filtering',
    summary         TEXT
        COMMENT 'Brief document summary (optional, may be LLM-generated later)',
    chunk_count     INT UNSIGNED  NOT NULL DEFAULT 0
        COMMENT 'Number of chunks generated from this document',
    created_at      DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_doc_collection (collection_name),
    INDEX idx_doc_source_type (source_type),
    INDEX idx_doc_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Ingested documents registered in Chroma';


-- ============================================================================
-- knowledge_chunks
-- ============================================================================

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    document_id     BIGINT UNSIGNED NOT NULL
        COMMENT 'FK to knowledge_documents.id',
    collection_name VARCHAR(128)  NOT NULL
        COMMENT 'Denormalized for fast lookup',
    chunk_index     INT UNSIGNED  NOT NULL DEFAULT 0
        COMMENT '0-based chunk position within the document',
    chroma_id       VARCHAR(512)  NOT NULL
        COMMENT 'Chroma document id, format: {collection}_{content_hash}',
    title           VARCHAR(512)
        COMMENT 'Inherited from parent document',
    content_hash    VARCHAR(64)   NOT NULL
        COMMENT 'SHA-256 prefix for deduplication',
    token_count     INT UNSIGNED  NOT NULL DEFAULT 0
        COMMENT 'Character count of chunk content',
    metadata_json   JSON
        COMMENT 'Full chunk metadata (Chroma-compatible subset)',
    created_at      DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY uk_chroma_id (chroma_id),
    INDEX idx_chunk_collection (collection_name),
    INDEX idx_chunk_document (document_id),
    INDEX idx_chunk_hash (content_hash),

    CONSTRAINT fk_chunk_document
        FOREIGN KEY (document_id) REFERENCES knowledge_documents(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Individual chunks stored as vectors in Chroma';


-- ============================================================================
-- Seed data: register the 4 MVP collections
-- ============================================================================

INSERT INTO knowledge_collections (name, display_name, domain, description, agent_scope) VALUES
('advisory_knowledge',  '投顾知识库',   'advisory',            '资产配置、投资组合、定投策略等投顾通用知识',           '["investment_advisor"]'),
('compliance_knowledge', '合规知识库',  'compliance',          '金融监管法规、合规红线、信息披露规范',                  '["compliance", "investment_advisor"]'),
('education_knowledge',  '金融科普知识库', 'financial_education', '基金/股票/债券等基础概念、金融术语、入门教程',        '["education"]'),
('risk_knowledge',       '风控知识库',   'risk_control',        '风险模型、风险评估方法、持仓风险与流动性管理',          '["risk_control"]')
ON DUPLICATE KEY UPDATE display_name = VALUES(display_name);
