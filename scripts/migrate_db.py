"""
Database migration script
Applies the database schema to PostgreSQL
"""

import psycopg2
import os
import argparse
from pathlib import Path


def run_migration(connection_string: str, schema_file: str):
    """
    Run database migration from schema file

    Args:
        connection_string: PostgreSQL connection string
        schema_file: Path to SQL schema file
    """
    print(f"Connecting to database...")

    try:
        # Connect to database
        conn = psycopg2.connect(connection_string)
        conn.autocommit = True
        cursor = conn.cursor()

        # Read schema file
        print(f"Reading schema from {schema_file}...")
        with open(schema_file, 'r') as f:
            schema_sql = f.read()

        # Execute schema
        print("Executing schema...")
        cursor.execute(schema_sql)

        # Verify tables
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)

        tables = cursor.fetchall()
        print("\nCreated tables:")
        for table in tables:
            print(f"  - {table[0]}")

        # Close connection
        cursor.close()
        conn.close()

        print("\nMigration completed successfully!")

    except Exception as e:
        print(f"Error during migration: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Run database migration')
    parser.add_argument(
        '--connection-string',
        type=str,
        default=os.getenv('DATABASE_URL', 'postgresql://avm_user:changeme123@localhost:5432/property_avm'),
        help='PostgreSQL connection string'
    )
    parser.add_argument(
        '--schema-file',
        type=str,
        default='infra/db_schema.sql',
        help='Path to SQL schema file'
    )

    args = parser.parse_args()

    if not os.path.exists(args.schema_file):
        print(f"Error: Schema file not found: {args.schema_file}")
        return

    run_migration(args.connection_string, args.schema_file)


if __name__ == '__main__':
    main()
