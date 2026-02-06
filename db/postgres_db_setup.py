import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from utils.parsers.env_parser import EnvConfig
from psycopg2 import sql

def create_role_if_not_exists(conn, target_user, target_password, createdb=False):
    """
    Checks and creates a PostgreSQL role if it does not exist.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s;", (target_user,))
        if not cur.fetchone():
            print(f"Creating role/user '{target_user}' ...")
            query = sql.SQL(
                "CREATE USER {role} WITH PASSWORD %s {createdb}"
            ).format(
                role=sql.Identifier(target_user),
                createdb=sql.SQL("CREATEDB") if createdb else sql.SQL("")
            )
            cur.execute(query, [target_password])
            print(f"Role/user '{target_user}' created.")
        else:
            print(f"Role/user '{target_user}' already exists.")

class PostgresDbSetup:
    """
    Safe and Idempotent PostgreSQL Database and Schema Setup Utility

    - Ensures user/role exists (creates if missing)
    - Ensures database exists (creates if missing) and owned properly
    - Enables pgvector
    - Creates required tables for vector/document storage (if missing)
    - Never drops or overwrites existing tables/data
    """
    def __init__(self, environment=None):
        # Load or validate environment
        self.env = environment if environment is not None else EnvConfig()

        # Try to get 'admin'/superuser credentials for setup, fallback to 'postgres'
        self.admin_user = self.env.get('POSTGRES_ADMIN_USERNAME', 'codebase_analytics_pg')
        self.admin_password = self.env.get('POSTGRES_ADMIN_PASSWORD', 'codebase_analytics_pg')

        self.username = self.env.get('POSTGRES_USERNAME')
        self.password = self.env.get('POSTGRES_PASSWORD')
        self.database = self.env.get('POSTGRES_DATABASE')
        self.host = self.env.get('POSTGRES_HOST', 'localhost')
        self.port = self.env.get('POSTGRES_PORT', 5432)
        self.collection_table = self.env.get('POSTGRES_COLLECTION_TABLENAME')
        self.embedding_table = self.env.get('POSTGRES_EMBEDDING_TABLENAME')

    def create_db_if_not_exists(self, conn):
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (self.database,))
            if not cur.fetchone():
                print(f"Creating database '{self.database}' with owner '{self.username}' ...")
                cur.execute(
                    sql.SQL("CREATE DATABASE {} OWNER {};").format(
                        sql.Identifier(self.database),
                        sql.Identifier(self.username)))
                print(f"Database '{self.database}' created.")
            else:
                print(f"Database '{self.database}' already exists.")

    def run_schema_setup(self, dbconn):
        # Enable pgvector extension (must be superuser)
        with dbconn.cursor() as cur:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                print("pgvector extension enabled.")
            except psycopg2.errors.InsufficientPrivilege:
                print("WARNING: Not superuser, skipping CREATE EXTENSION vector. Please ensure it's installed by a superuser.")
                dbconn.rollback()
            except Exception as ex:
                print(f"Could not create extension 'vector': {ex}")
                dbconn.rollback()

        # Create tables if not exist
        with dbconn.cursor() as cur:
            print(f"Creating table {self.collection_table} (if not exists)...")
            cur.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    uuid UUID PRIMARY KEY,
                    name TEXT,
                    cmetadata JSONB
                );
            """).format(sql.Identifier(self.collection_table)))

            print(f"Creating table {self.embedding_table} (if not exists)...")
            cur.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    id UUID PRIMARY KEY,
                    collection_id UUID NOT NULL REFERENCES {}(uuid) ON DELETE CASCADE,
                    embedding VECTOR(1024),
                    document TEXT,
                    cmetadata JSONB,
                    source_file TEXT,
                    ingested_at TIMESTAMPTZ DEFAULT NOW()
                );
            """).format(
                sql.Identifier(self.embedding_table),
                sql.Identifier(self.collection_table)
            ))

            # Optional: Add indexes for faster queries (uncomment if needed)
            # cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS idx_embedding_vector ON {} USING ivfflat (embedding);").format(sql.Identifier(self.embedding_table)))
            # cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS idx_collection_id ON {} (collection_id);").format(sql.Identifier(self.embedding_table)))
            dbconn.commit()
            print("Vector schema/tables ready.")

    def run(self):
        # 1. Connect as admin to 'postgres' DB for role/database setup
        with psycopg2.connect(
            dbname='postgres',
            user=self.admin_user,
            password=self.admin_password,
            host=self.host,
            port=self.port
        ) as conn_admin:
            conn_admin.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            try:
                # Ensure role exists
                create_role_if_not_exists(
                    conn_admin,
                    target_user=self.username,
                    target_password=self.password,
                    createdb=True
                )
                # Ensure DB exists
                self.create_db_if_not_exists(conn_admin)
            except Exception as ex:
                print(f"Error during admin setup: {ex}")
                raise

        # 2. Connect to user database for schema/extension setup
        with psycopg2.connect(
            dbname=self.database,
            user=self.admin_user,
            password=self.admin_password,
            host=self.host,
            port=self.port
        ) as conn_user:
            try:
                self.run_schema_setup(conn_user)
            except Exception as ex:
                print(f"Error during schema setup: {ex}")
                raise
        print("All schema/database setup complete.")

if __name__ == "__main__":
    PostgresDbSetup().run()