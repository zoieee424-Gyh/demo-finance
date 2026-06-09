"""
KnowledgeMetadataStore — MySQL-backed metadata for the knowledge base.

Design:
  - Uses pymysql (no ORM) for lightweight, auditable SQL.
  - MySQL stores collection/document/chunk metadata only.
  - Chroma stores vector embeddings and full text content.
  - Linked via collection_name + chroma_id.

Security:
  - All configuration comes from environment variables; passwords are never
    hard-coded.
  - Default mysql_enabled=False: tests and unconfigured environments never
    attempt a real connection.
"""
from __future__ import annotations

import json
from typing import Any

import pymysql


class KnowledgeMetadataStore:
    """MySQL-backed structured metadata store for the RAG knowledge base.

    Usage::

        store = KnowledgeMetadataStore()
        if store.is_configured():
            store.initialize_schema()
            doc_id = store.upsert_document(...)
            store.replace_chunks(doc_id, ...)
    """

    def __init__(self) -> None:
        from app.core.config import settings

        self._enabled = settings.mysql_enabled
        self._host = settings.mysql_host
        self._port = settings.mysql_port
        self._user = settings.mysql_user
        self._password = settings.mysql_password
        self._database = settings.mysql_database
        self._conn: Any = None

    # ── Connection management ────────────────────────────────────────

    def is_configured(self) -> bool:
        """Return True if MySQL is enabled AND a password is provided."""
        return bool(self._enabled and self._password)

    def _get_connection(self) -> Any:
        """Return a live pymysql connection, creating one lazily."""
        if self._conn is None or not getattr(self._conn, "open", False):
            self._conn = pymysql.connect(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password,
                database=self._database,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=False,
            )
        return self._conn

    def close(self) -> None:
        """Close the underlying connection if open."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> "KnowledgeMetadataStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Schema initialisation ────────────────────────────────────────

    def initialize_schema(self, sql_path: str | None = None) -> None:
        """Execute the knowledge schema DDL and seed collections.

        Connects without specifying a database first so that the
        ``CREATE DATABASE IF NOT EXISTS`` statement in the schema
        file can succeed. After the database is created the connection
        switches to it.

        Args:
            sql_path: Path to knowledge_schema.sql. If None, uses the
                      default path relative to this project.
        """
        if sql_path is None:
            from pathlib import Path
            sql_path = str(
                Path(__file__).resolve().parent.parent.parent
                / "sql" / "knowledge_schema.sql"
            )

        with open(sql_path, "r", encoding="utf-8") as f:
            sql = f.read()

        # First, ensure the database exists by connecting without one
        self._ensure_database()

        conn = self._get_connection()
        # Split on semicolons, skip empty / comment-only blocks
        statements = _split_sql_statements(sql)
        with conn.cursor() as cur:
            for stmt in statements:
                if not stmt.strip():
                    continue
                # Skip the CREATE DATABASE / USE statements — already handled
                upper = stmt.strip().upper()
                if upper.startswith("CREATE DATABASE") or upper.startswith("USE "):
                    continue
                try:
                    cur.execute(stmt)
                except pymysql.MySQLError as e:
                    # Duplicate key / already-exists errors are safe to ignore
                    code = getattr(e, "args", [None])[0] if e.args else None
                    if code == 1050:   # Table already exists
                        pass
                    elif code == 1062:  # Duplicate entry (seed data)
                        pass
                    elif code == 1007:  # Database already exists
                        pass
                    else:
                        raise
        conn.commit()

    def _ensure_database(self) -> None:
        """Create the target database if it does not exist."""
        conn = pymysql.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            charset="utf8mb4",
            autocommit=True,
        )
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{self._database}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.close()

    # ── Collection CRUD ──────────────────────────────────────────────

    def upsert_collection(
        self,
        name: str,
        display_name: str,
        domain: str,
        description: str = "",
        agent_scope: list[str] | None = None,
        retrieval_policy: str = "semantic",
        enabled: bool = True,
    ) -> None:
        """Insert or update a collection record."""
        agent_json = json.dumps(agent_scope or [], ensure_ascii=False)
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO knowledge_collections
                   (name, display_name, domain, description, agent_scope,
                    retrieval_policy, enabled)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                     display_name = VALUES(display_name),
                     domain = VALUES(domain),
                     description = VALUES(description),
                     agent_scope = VALUES(agent_scope),
                     retrieval_policy = VALUES(retrieval_policy),
                     enabled = VALUES(enabled)""",
                (name, display_name, domain, description, agent_json,
                 retrieval_policy, int(enabled)),
            )
        conn.commit()

    # ── Document CRUD ────────────────────────────────────────────────

    def upsert_document(
        self,
        collection_name: str,
        title: str,
        source_type: str = "knowledge_base",
        authority: str | None = None,
        url: str | None = None,
        file_path: str | None = None,
        version: str = "1.0",
        status: str = "active",
        tags: list[str] | None = None,
        summary: str | None = None,
        chunk_count: int = 0,
    ) -> int:
        """Insert or update a document record. Returns the document id.

        Upsert key: (collection_name, file_path) when file_path is set;
        otherwise (collection_name, title).
        """
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        conn = self._get_connection()
        with conn.cursor() as cur:
            # Try the ON DUPLICATE KEY path — but the table doesn't have
            # a unique key on (collection_name, file_path), so we do a
            # manual upsert via SELECT + INSERT/UPDATE.
            existing = self._find_document(cur, collection_name, title, file_path)

            if existing:
                doc_id = existing["id"]
                cur.execute(
                    """UPDATE knowledge_documents
                       SET source_type=%s, authority=%s, url=%s, file_path=%s,
                           version=%s, status=%s, tags=%s, summary=%s,
                           chunk_count=%s
                       WHERE id=%s""",
                    (source_type, authority, url, file_path, version, status,
                     tags_json, summary, chunk_count, doc_id),
                )
            else:
                cur.execute(
                    """INSERT INTO knowledge_documents
                       (collection_name, title, source_type, authority, url,
                        file_path, version, status, tags, summary, chunk_count)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (collection_name, title, source_type, authority, url,
                     file_path, version, status, tags_json, summary, chunk_count),
                )
                doc_id = cur.lastrowid
        conn.commit()
        return doc_id

    def upsert_document_with_chunks(
        self,
        collection_name: str,
        title: str,
        chunks_with_chroma_ids: list[dict],
        source_type: str = "knowledge_base",
        authority: str | None = None,
        url: str | None = None,
        file_path: str | None = None,
        version: str = "1.0",
        status: str = "active",
        tags: list[str] | None = None,
        summary: str | None = None,
        chunk_count: int = 0,
    ) -> int:
        """Upsert a document and replace its chunks in one transaction."""
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                existing = self._find_document(cur, collection_name, title, file_path)

                if existing:
                    doc_id = existing["id"]
                    cur.execute(
                        """UPDATE knowledge_documents
                           SET source_type=%s, authority=%s, url=%s, file_path=%s,
                               version=%s, status=%s, tags=%s, summary=%s,
                               chunk_count=%s
                           WHERE id=%s""",
                        (source_type, authority, url, file_path, version, status,
                         tags_json, summary, chunk_count, doc_id),
                    )
                else:
                    cur.execute(
                        """INSERT INTO knowledge_documents
                           (collection_name, title, source_type, authority, url,
                            file_path, version, status, tags, summary, chunk_count)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (collection_name, title, source_type, authority, url,
                         file_path, version, status, tags_json, summary, chunk_count),
                    )
                    doc_id = cur.lastrowid

                self._replace_chunks_with_cursor(
                    cur, doc_id, collection_name, chunks_with_chroma_ids
                )

            conn.commit()
            return doc_id
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def _find_document(
        cur: Any, collection_name: str, title: str, file_path: str | None,
    ) -> dict | None:
        """Return an existing document row or None."""
        if file_path:
            cur.execute(
                """SELECT id FROM knowledge_documents
                   WHERE collection_name=%s AND file_path=%s LIMIT 1""",
                (collection_name, file_path),
            )
        else:
            cur.execute(
                """SELECT id FROM knowledge_documents
                   WHERE collection_name=%s AND title=%s LIMIT 1""",
                (collection_name, title),
            )
        return cur.fetchone()

    # ── Chunk CRUD ───────────────────────────────────────────────────

    def replace_chunks(
        self,
        document_id: int,
        collection_name: str,
        chunks_with_chroma_ids: list[dict],
    ) -> int:
        """Replace all chunks for a document atomically.

        Args:
            document_id: FK to knowledge_documents.id.
            collection_name: Denormalised collection name.
            chunks_with_chroma_ids: List of dicts, each containing:
                - chroma_id (str, required)
                - chunk_index (int)
                - title (str)
                - content_hash (str)
                - token_count (int)
                - metadata_json (dict)

        Returns:
            Number of chunks inserted.
        """
        if not chunks_with_chroma_ids:
            return 0

        conn = self._get_connection()
        with conn.cursor() as cur:
            inserted = self._replace_chunks_with_cursor(
                cur, document_id, collection_name, chunks_with_chroma_ids
            )
        conn.commit()
        return inserted

    @staticmethod
    def _replace_chunks_with_cursor(
        cur: Any,
        document_id: int,
        collection_name: str,
        chunks_with_chroma_ids: list[dict],
    ) -> int:
        """Replace chunks using an existing cursor/transaction."""
        cur.execute("DELETE FROM knowledge_chunks WHERE document_id=%s", (document_id,))

        inserted = 0
        for chunk in chunks_with_chroma_ids:
            meta_json = json.dumps(
                chunk.get("metadata_json", {}), ensure_ascii=False
            )
            try:
                cur.execute(
                    """INSERT INTO knowledge_chunks
                       (document_id, collection_name, chunk_index, chroma_id,
                        title, content_hash, token_count, metadata_json)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        document_id,
                        collection_name,
                        chunk.get("chunk_index", 0),
                        chunk["chroma_id"],
                        chunk.get("title", ""),
                        chunk.get("content_hash", ""),
                        chunk.get("token_count", 0),
                        meta_json,
                    ),
                )
                inserted += 1
            except pymysql.IntegrityError:
                # Duplicate chroma_id — skip (should not happen after DELETE)
                pass
        return inserted

    # ── Read helpers ─────────────────────────────────────────────────

    def get_collection_counts(self) -> list[dict]:
        """Return per-collection document / chunk counts."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("""SELECT name, display_name, domain, enabled
                           FROM knowledge_collections ORDER BY name""")
            return cur.fetchall()

    def get_document_counts(self) -> list[dict]:
        """Return per-collection document count."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT collection_name, COUNT(*) AS doc_count
                   FROM knowledge_documents WHERE status='active'
                   GROUP BY collection_name ORDER BY collection_name"""
            )
            return cur.fetchall()

    def get_chunk_counts_by_collection(self) -> dict[str, int]:
        """Return {collection_name: chunk_count} for all collections."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT collection_name, COUNT(*) AS cnt
                   FROM knowledge_chunks GROUP BY collection_name"""
            )
            rows = cur.fetchall()
        return {r["collection_name"]: r["cnt"] for r in rows}

    def get_total_counts(self) -> dict[str, int]:
        """Return total document and chunk counts."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM knowledge_documents WHERE status='active'")
            doc_total = cur.fetchone()["cnt"]
            cur.execute("SELECT COUNT(*) AS cnt FROM knowledge_chunks")
            chunk_total = cur.fetchone()["cnt"]
            cur.execute("SELECT COUNT(*) AS cnt FROM knowledge_collections WHERE enabled=1")
            coll_total = cur.fetchone()["cnt"]
        return {
            "collections": coll_total,
            "documents": doc_total,
            "chunks": chunk_total,
        }


# ── Helpers ─────────────────────────────────────────────────────────────


def _split_sql_statements(sql: str) -> list[str]:
    """Split a SQL script into individual statements.

    Skips pure-comment lines and empty statements.
    """
    statements: list[str] = []
    current: list[str] = []

    for line in sql.split("\n"):
        stripped = line.strip()
        # Skip comment-only lines
        if stripped.startswith("--") or stripped.startswith("#"):
            continue
        if not stripped:
            continue
        current.append(line)
        if stripped.endswith(";"):
            stmt = "\n".join(current).rstrip(";")
            if stmt.strip():
                statements.append(stmt)
            current = []

    # Catch trailing statement without semicolon
    if current:
        stmt = "\n".join(current).strip()
        if stmt:
            statements.append(stmt)

    return statements
