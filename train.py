"""
train.py — Schema Knowledge Seeder for Vanna 2.0

Connects to PostgreSQL, extracts full schema (tables, columns, PKs, FKs,
constraints), and seeds it into ChromaDB as text memories using
nomic-embed-text embeddings via Ollama.

At query time, the Vanna agent's DefaultLlmContextEnhancer searches these
memories by vector similarity and injects the top matches into the LLM's
system prompt — giving it the schema context it needs to generate accurate SQL.

Usage:
    python train.py                  # Seed schema (append to existing)
    python train.py --reset          # Clear existing memories and re-seed
    python train.py --add-examples   # Also seed example question-SQL pairs
    python train.py --reset --add-examples   # Full clean re-seed
"""

import argparse
import json
import uuid
import sys
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import RealDictCursor
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


# ═══════════════════════════════════════════════════════════════════
#  Configuration — update these to match your environment
# ═══════════════════════════════════════════════════════════════════

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OLLAMA_HOST = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "vanna_memory")
DB_SCHEMA = os.environ.get("DB_SCHEMA", "public")

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 5432)),
    "dbname": os.environ.get("DB_NAME", "dvdrental"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", ""),
}


# ═══════════════════════════════════════════════════════════════════
#  ChromaDB Initialization
# ═══════════════════════════════════════════════════════════════════

def get_embedding_function():
    """Create the Ollama embedding function for nomic-embed-text."""
    return OllamaEmbeddingFunction(
        url=OLLAMA_HOST,
        model_name=EMBED_MODEL,
    )


def get_collection(reset=False):
    """Get or create the ChromaDB collection with the correct embedding function."""
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    ef = get_embedding_function()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("  ✓ Cleared existing collection")
        except Exception:
            print("  ⓘ No existing collection to clear")

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )


# ═══════════════════════════════════════════════════════════════════
#  Schema Fetching from PostgreSQL
# ═══════════════════════════════════════════════════════════════════

def fetch_schema():
    """
    Fetch complete schema from PostgreSQL:
    tables, columns (with types), primary keys, foreign keys,
    unique constraints, and row counts.
    
    Uses DB_SCHEMA to target a specific PostgreSQL schema (default: 'public').
    """
    conn = psycopg2.connect(**DB_CONFIG)
    schemas = tuple(s.strip() for s in DB_SCHEMA.split(','))
    cur = conn.cursor(cursor_factory=RealDictCursor)
    schema = {}

    # All user-defined base tables in the target schema
    cur.execute("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema IN %s
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """, (schemas,))
    tables = [(row["table_schema"], row["table_name"]) for row in cur.fetchall()]

    for schema_name, table in tables:
        # ── Columns ──
        cur.execute("""
            SELECT column_name, data_type, is_nullable, column_default,
                   character_maximum_length, numeric_precision, numeric_scale,
                   udt_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """, (schema_name, table))
        columns = cur.fetchall()

        # ── Primary Keys ──
        cur.execute("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.table_schema = %s
              AND tc.table_name = %s
              AND tc.constraint_type = 'PRIMARY KEY'
        """, (schema_name, table))
        pks = [row["column_name"] for row in cur.fetchall()]

        # ── Foreign Keys ──
        cur.execute("""
            SELECT
                kcu.column_name,
                ccu.table_name  AS referenced_table,
                ccu.column_name AS referenced_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON tc.constraint_name = ccu.constraint_name
             AND tc.table_schema = ccu.table_schema
            WHERE tc.table_schema = %s
              AND tc.table_name = %s
              AND tc.constraint_type = 'FOREIGN KEY'
        """, (schema_name, table))
        fks = cur.fetchall()

        # ── Unique Constraints ──
        cur.execute("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.table_schema = %s
              AND tc.table_name = %s
              AND tc.constraint_type = 'UNIQUE'
        """, (schema_name, table))
        uniques = [row["column_name"] for row in cur.fetchall()]

        # ── Row Count (Fast Estimate via pg_class) ──
        cur.execute("""
            SELECT reltuples::bigint AS cnt
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = %s AND n.nspname = %s
        """, (table, schema_name))
        result = cur.fetchone()
        row_count = result["cnt"] if result and result["cnt"] >= 0 else 0

        schema[table] = {
            "columns": columns,
            "primary_keys": pks,
            "foreign_keys": fks,
            "unique_constraints": uniques,
            "row_count": row_count,
        }

    cur.close()
    conn.close()
    return schema


# ═══════════════════════════════════════════════════════════════════
#  Memory Builders — convert schema into searchable text descriptions
# ═══════════════════════════════════════════════════════════════════

def build_table_memory(table_name, info):
    """
    Build a rich natural-language description of a table.
    Designed to match well against user questions like
    "show me customer emails" or "how many films are there".
    """
    lines = [f"Table: {table_name} ({info['row_count']} rows)"]
    lines.append("Columns:")

    for col in info["columns"]:
        dtype = col["data_type"]
        if col["character_maximum_length"]:
            dtype += f"({col['character_maximum_length']})"
        elif col["numeric_precision"] and col["data_type"] == "numeric":
            scale = col["numeric_scale"] or 0
            dtype += f"({col['numeric_precision']},{scale})"

        parts = [f"  - {col['column_name']} {dtype}"]
        if col["is_nullable"] == "NO":
            parts.append("NOT NULL")
        if col["column_name"] in info["primary_keys"]:
            parts.append("PRIMARY KEY")
        if col["column_name"] in info["unique_constraints"]:
            parts.append("UNIQUE")
        if col["column_default"]:
            parts.append(f"DEFAULT {col['column_default']}")
        lines.append(", ".join(parts))

    if info["foreign_keys"]:
        lines.append("Foreign Keys:")
        for fk in info["foreign_keys"]:
            lines.append(
                f"  - {table_name}.{fk['column_name']} → "
                f"{fk['referenced_table']}.{fk['referenced_column']}"
            )

    return "\n".join(lines)


def build_ddl_memory(table_name, info):
    """Build a CREATE TABLE DDL statement (useful for LLMs that think in SQL)."""
    col_defs = []
    for col in info["columns"]:
        dtype = col["udt_name"]
        if col["character_maximum_length"]:
            dtype = f"varchar({col['character_maximum_length']})"

        parts = [f"    {col['column_name']} {dtype}"]
        if col["is_nullable"] == "NO":
            parts.append("NOT NULL")
        if col["column_default"]:
            parts.append(f"DEFAULT {col['column_default']}")
        col_defs.append(" ".join(parts))

    if info["primary_keys"]:
        col_defs.append(f"    PRIMARY KEY ({', '.join(info['primary_keys'])})")

    for fk in info["foreign_keys"]:
        col_defs.append(
            f"    FOREIGN KEY ({fk['column_name']}) "
            f"REFERENCES {fk['referenced_table']}({fk['referenced_column']})"
        )

    body = ",\n".join(col_defs)
    return f"CREATE TABLE {table_name} (\n{body}\n);"


def build_overview_memory(schema):
    """Build a high-level database overview listing all tables and row counts.
    
    Column details are intentionally omitted here to keep the overview compact
    and within embedding model context limits. Full column information is
    already stored separately via per-table description and DDL memories.
    """
    db_name = DB_CONFIG["dbname"]
    lines = [f"Database: {db_name}"]
    lines.append(f"Total tables: {len(schema)}")
    lines.append(f"Tables: {', '.join(sorted(schema.keys()))}")
    lines.append("")
    for table_name in sorted(schema.keys()):
        info = schema[table_name]
        col_count = len(info["columns"])
        fk_count = len(info["foreign_keys"])
        lines.append(
            f"  {table_name}: {info['row_count']} rows, {col_count} columns, {fk_count} foreign keys"
        )
    return "\n".join(lines)


def build_relationships_memory(schema, chunk_size=15):
    """Build foreign-key relationship maps, chunked to stay within embedding context limits.
    
    Returns a list of text chunks (each covering up to `chunk_size` tables)
    instead of a single string, so large schemas don't overflow the
    embedding model's context window.
    """
    sorted_tables = sorted(schema.keys())
    chunks = []

    for i in range(0, len(sorted_tables), chunk_size):
        batch = sorted_tables[i:i + chunk_size]
        part_num = (i // chunk_size) + 1
        total_parts = (len(sorted_tables) + chunk_size - 1) // chunk_size
        lines = [f"Database Relationships and Join Paths (Part {part_num}/{total_parts}):"]
        lines.append("")

        for table_name in batch:
            for fk in schema[table_name]["foreign_keys"]:
                lines.append(
                    f"JOIN {table_name} WITH {fk['referenced_table']}: "
                    f"{table_name}.{fk['column_name']} = "
                    f"{fk['referenced_table']}.{fk['referenced_column']}"
                )

        chunk_text = "\n".join(lines)
        # Only add if there are actual relationships in this batch
        if len(lines) > 2:
            chunks.append(chunk_text)

    # If no relationships exist at all, return a single descriptive chunk
    if not chunks:
        chunks.append("Database Relationships and Join Paths: No foreign key relationships found.")

    return chunks


# ═══════════════════════════════════════════════════════════════════
#  ChromaDB Writers — save memories matching ChromaAgentMemory format
# ═══════════════════════════════════════════════════════════════════

def save_text_memory(collection, content):
    """
    Save a text memory to ChromaDB.
    Mimics the internal format of ChromaAgentMemory.save_text_memory()
    so the DefaultLlmContextEnhancer can find and retrieve it.
    """
    memory_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    collection.add(
        ids=[memory_id],
        documents=[content],
        metadatas=[{
            "content": content,
            "timestamp": timestamp,
            "is_text_memory": True,
        }],
    )
    return memory_id


def save_tool_memory(collection, question, sql):
    """
    Save a question→SQL pair as a tool memory.
    Mimics the internal format of ChromaAgentMemory.save_tool_usage()
    so the SearchSavedCorrectToolUsesTool can find and retrieve it.
    """
    memory_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    collection.add(
        ids=[memory_id],
        documents=[question],
        metadatas=[{
            "question": question,
            "tool_name": "run_sql",
            "args_json": json.dumps({"sql": sql}),
            "timestamp": timestamp,
            "success": True,
        }],
    )
    return memory_id


# ═══════════════════════════════════════════════════════════════════
#  Example Question-SQL Pairs (dvdrental-specific)
# ═══════════════════════════════════════════════════════════════════

EXAMPLE_QUERIES = [
    (
        "How many customers are there?",
        "SELECT COUNT(*) AS total_customers FROM customer;",
    ),
    (
        "List all film titles and their rental rates",
        "SELECT title, rental_rate FROM film ORDER BY title;",
    ),
    (
        "Which customers have rented the most films?",
        "SELECT c.first_name, c.last_name, COUNT(r.rental_id) AS rental_count "
        "FROM customer c "
        "JOIN rental r ON c.customer_id = r.customer_id "
        "GROUP BY c.customer_id, c.first_name, c.last_name "
        "ORDER BY rental_count DESC LIMIT 10;",
    ),
    (
        "What are the top 10 most rented films?",
        "SELECT f.title, COUNT(r.rental_id) AS times_rented "
        "FROM film f "
        "JOIN inventory i ON f.film_id = i.film_id "
        "JOIN rental r ON i.inventory_id = r.inventory_id "
        "GROUP BY f.film_id, f.title "
        "ORDER BY times_rented DESC LIMIT 10;",
    ),
    (
        "Show total revenue by store",
        "SELECT s.store_id, SUM(p.amount) AS total_revenue "
        "FROM store s "
        "JOIN staff st ON s.store_id = st.store_id "
        "JOIN payment p ON st.staff_id = p.staff_id "
        "GROUP BY s.store_id ORDER BY total_revenue DESC;",
    ),
    (
        "List all film categories with the number of films in each",
        "SELECT c.name AS category, COUNT(fc.film_id) AS film_count "
        "FROM category c "
        "JOIN film_category fc ON c.category_id = fc.category_id "
        "GROUP BY c.category_id, c.name ORDER BY film_count DESC;",
    ),
    (
        "Find customers who have never rented a film",
        "SELECT c.first_name, c.last_name, c.email "
        "FROM customer c "
        "LEFT JOIN rental r ON c.customer_id = r.customer_id "
        "WHERE r.rental_id IS NULL;",
    ),
    (
        "What is the average rental duration by film category?",
        "SELECT c.name AS category, AVG(f.rental_duration) AS avg_duration "
        "FROM category c "
        "JOIN film_category fc ON c.category_id = fc.category_id "
        "JOIN film f ON fc.film_id = f.film_id "
        "GROUP BY c.category_id, c.name ORDER BY avg_duration DESC;",
    ),
    (
        "Show monthly revenue for 2005",
        "SELECT DATE_TRUNC('month', payment_date) AS month, SUM(amount) AS revenue "
        "FROM payment "
        "WHERE EXTRACT(YEAR FROM payment_date) = 2005 "
        "GROUP BY month ORDER BY month;",
    ),
    (
        "List actors who appeared in the most films",
        "SELECT a.first_name, a.last_name, COUNT(fa.film_id) AS film_count "
        "FROM actor a "
        "JOIN film_actor fa ON a.actor_id = fa.actor_id "
        "GROUP BY a.actor_id, a.first_name, a.last_name "
        "ORDER BY film_count DESC LIMIT 10;",
    ),
]


# ═══════════════════════════════════════════════════════════════════
#  Main Training Function
# ═══════════════════════════════════════════════════════════════════

def train(reset=False, add_examples=False):
    """Fetch schema from PostgreSQL and seed it into ChromaDB."""
    print("=" * 60)
    print("  Vanna 2.0 — Schema Knowledge Seeder")
    print("=" * 60)

    print(f"\n  📦 Embedding model : {EMBED_MODEL}")
    print(f"  🌐 Ollama host     : {OLLAMA_HOST}")
    print(f"  📂 ChromaDB path   : {CHROMA_DIR}")
    print(f"  📋 Collection      : {COLLECTION_NAME}")
    print(f"  🗄️  Database        : {DB_CONFIG['dbname']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")

    # ── Initialize ChromaDB ──
    print("\n--- Initializing ChromaDB ---")
    collection = get_collection(reset=reset)

    # ── Fetch Schema ──
    print("\n--- Fetching Schema from PostgreSQL ---")
    try:
        schema = fetch_schema()
    except Exception as e:
        print(f"  ✗ Failed to connect to database: {e}")
        sys.exit(1)
    print(f"  ✓ Found {len(schema)} tables")

    count = 0

    # ── Seed: Database Overview ──
    print("\n--- Seeding Database Overview ---")
    overview = build_overview_memory(schema)
    save_text_memory(collection, overview)
    count += 1
    print("  ✓ Overview")

    # ── Seed: Per-Table Schema + DDL ──
    print("\n--- Seeding Table Schemas ---")
    for table_name in sorted(schema.keys()):
        info = schema[table_name]

        # Natural-language description (matches user questions well)
        desc = build_table_memory(table_name, info)
        save_text_memory(collection, desc)
        count += 1

        # DDL statement (matches SQL-oriented thinking)
        ddl = build_ddl_memory(table_name, info)
        save_text_memory(collection, ddl)
        count += 1

        fk_str = f", {len(info['foreign_keys'])} FKs" if info["foreign_keys"] else ""
        print(
            f"  ✓ {table_name:30s} "
            f"({info['row_count']:>6} rows, {len(info['columns']):>2} cols{fk_str})"
        )

    # ── Seed: Relationships (chunked for large schemas) ──
    print("\n--- Seeding Relationships ---")
    relationship_chunks = build_relationships_memory(schema)
    for chunk in relationship_chunks:
        save_text_memory(collection, chunk)
        count += 1
    print(f"  ✓ Relationship / join-path map ({len(relationship_chunks)} chunk(s))")

    # ── Seed: Example Q→SQL Pairs (optional) ──
    if add_examples:
        print("\n--- Seeding Example Question→SQL Pairs ---")
        for question, sql in EXAMPLE_QUERIES:
            save_tool_memory(collection, question, sql)
            count += 1
            print(f"  ✓ {question[:55]}...")

    # ── Summary ──
    total_tool = len(EXAMPLE_QUERIES) if add_examples else 0

    print("\n" + "=" * 60)
    print("  ✅ Training complete!")
    print(f"  📊 Seeded {count} total memories:")
    print(f"     • {len(schema)} table descriptions")
    print(f"     • {len(schema)} DDL statements")
    print(f"     • 1 database overview")
    print(f"     • {len(relationship_chunks)} relationship map chunk(s)")
    if add_examples:
        print(f"     • {total_tool} example question→SQL pairs")
    print(f"\n  🔍 At query time, the top-5 matching memories")
    print(f"     will be injected into the LLM's system prompt.")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed PostgreSQL schema knowledge into Vanna 2.0 agent memory (ChromaDB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python train.py --reset              Clear old memories and re-seed schema
  python train.py --add-examples       Seed schema + example Q→SQL pairs
  python train.py --reset --add-examples  Full clean re-seed with examples

prerequisites:
  • Ollama must be running with nomic-embed-text pulled:
      ollama pull nomic-embed-text
  • PostgreSQL must be accessible with the credentials in this script
        """,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing ChromaDB collection before seeding (required when changing embedding models)",
    )
    parser.add_argument(
        "--add-examples",
        action="store_true",
        help="Also seed example question→SQL pairs for the dvdrental database",
    )
    args = parser.parse_args()

    train(reset=args.reset, add_examples=args.add_examples)
