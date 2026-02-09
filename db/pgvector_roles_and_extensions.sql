-- CARE — Codebase Analysis & Refactor Engine
-- Create application role/user for CARE vector DB if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_roles
        WHERE rolname = 'codebase_analytics_user'
    ) THEN
        CREATE ROLE codebase_analytics_user
            WITH LOGIN
                 NOSUPERUSER
                 NOCREATEDB
                 NOCREATEROLE
                 NOINHERIT
                 NOREPLICATION
                 PASSWORD 'postgres';
    END IF;
END
$$;