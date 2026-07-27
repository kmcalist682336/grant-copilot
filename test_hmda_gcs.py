import os

import duckdb
from dotenv import load_dotenv


# Load variables from the .env file in the project root.
load_dotenv()

access_id = os.getenv("GCS_HMAC_ACCESS_ID")
hmac_secret = os.getenv("GCS_HMAC_SECRET")

if not access_id or not hmac_secret:
    raise RuntimeError(
        "GCS_HMAC_ACCESS_ID or GCS_HMAC_SECRET is missing."
    )

print("Credentials loaded successfully.")


def sql_string(value: str) -> str:
    """Safely format a value as a DuckDB SQL string."""
    return "'" + value.replace("'", "''") + "'"


connection = duckdb.connect()

connection.execute("INSTALL httpfs")
connection.execute("LOAD httpfs")

connection.execute(
    f"""
    CREATE OR REPLACE SECRET hmda_gcs (
        TYPE gcs,
        KEY_ID {sql_string(access_id)},
        SECRET {sql_string(hmac_secret)},
        SCOPE 'gs://demo-grant-bucket/'
    )
    """
)

print("DuckDB is connected to Google Cloud Storage.")


path = (
    "gs://demo-grant-bucket/"
    "variable_tree/"
    "table_id=hmda/"
    "year=2024/"
    "variable=c1aa5d4f3f72/"
    "part-hmda_lar_2024.parquet"
)

preview = connection.execute(
    "SELECT * FROM read_parquet(?) LIMIT 10",
    [path],
).df()

print("\nFirst ten rows:")
print(preview.to_string(index=False))

connection.close()