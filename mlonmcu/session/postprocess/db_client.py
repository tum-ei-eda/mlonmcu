import argparse
import json
import os
import humanize
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text
from rich.console import Console
from rich.table import Table
from rich import print

import boto3
from botocore.client import Config

console = Console()

# -------------------------
# CONFIG
# -------------------------

DB_USER = os.environ.get("POSTGRES_USER", "mlonmcu")
DB_PASS = os.environ.get("POSTGRES_PASSWORD", None)
DB_NAME = os.environ.get("POSTGRES_DB", "mlonmcu")
DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_PORT = os.environ.get("POSTGRES_PORT", 5432)
assert DB_PASS is not None
# DB_URL = os.environ.get("DB_URL", f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
DB_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:3900")
S3_KEY = os.environ.get("S3_KEY")
S3_SECRET = os.environ.get("S3_SECRET")
assert S3_KEY is not None
assert S3_SECRET is not None

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET,
    region_name="garage",
    # config=Config(signature_version='s3', s3={'addressing_style': 'path'})
    # config=Config(signature_version='v4', s3={'addressing_style': 'path'})
)

engine = create_engine(DB_URL)


# -------------------------
# HELPERS
# -------------------------


def build_filter_query(args):
    conditions = []
    params = {}

    if args.label:
        conditions.append("label = :label")
        params["label"] = args.label

    if args.after:
        conditions.append("timestamp >= :after")
        params["after"] = args.after

    if args.before:
        conditions.append("timestamp <= :before")
        params["before"] = args.before

    if args.tags:
        conditions.append("tags ?| :tags")
        params["tags"] = args.tags

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    return where, params


# -------------------------
# COMMANDS
# -------------------------


def list_experiments(args):
    where, params = build_filter_query(args)

    query = f"""
    SELECT id, label, timestamp
    FROM experiments
    {where}
    ORDER BY timestamp DESC
    LIMIT 50
    """

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()

    table = Table(title="Experiments")
    table.add_column("ID")
    table.add_column("Label")
    table.add_column("Timestamp")

    for r in rows:
        table.add_row(str(r.id), r.label or "-", str(r.timestamp))

    console.print(table)


def show_experiment(args):
    query = """
    SELECT * FROM experiments WHERE id = :id
    """

    with engine.connect() as conn:
        row = conn.execute(text(query), {"id": args.id}).fetchone()

    if not row:
        print("[red]Experiment not found[/red]")
        return

    console.print("[bold]Experiment[/bold]")
    for k, v in row._mapping.items():
        console.print(f"{k}: {v}")


def list_artifacts(args):
    query = """
    SELECT id, type, size, metadata
    FROM artifacts
    WHERE experiment_id = :id
    """

    with engine.connect() as conn:
        rows = conn.execute(text(query), {"id": args.id}).fetchall()

    table = Table(title=f"Artifacts for Exp {args.id}")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Size")
    table.add_column("Key")

    for r in rows:
        # meta = json.loads(r.metadata)
        meta = r.metadata if isinstance(r.metadata, dict) else json.loads(r.metadata)
        table.add_row(str(r.id), r.type, humanize.naturalsize(r.size), meta.get("s3_key", "?"))

    console.print(table)


def download_artifacts(args):
    os.makedirs(args.out, exist_ok=True)

    query = """
    SELECT type, metadata
    FROM artifacts
    WHERE experiment_id = :id
    """

    with engine.connect() as conn:
        rows = conn.execute(text(query), {"id": args.id}).fetchall()

    for r in rows:
        if args.name and r.type != args.name:
            continue

        # meta = json.loads(r.metadata)
        meta = r.metadata if isinstance(r.metadata, dict) else json.loads(r.metadata)
        bucket = meta["bucket"]
        obj = meta["s3_key"]

        filename = os.path.join(args.out, os.path.basename(obj))

        console.print(f"Downloading {obj} → {filename}")

        s3.download_file(bucket, obj, filename)


def show_report(args):
    query = """
    SELECT metadata FROM artifacts
    WHERE experiment_id = :id AND type = 'report'
    LIMIT 1
    """

    with engine.connect() as conn:
        row = conn.execute(text(query), {"id": args.id}).fetchone()

    if not row:
        print("[red]No report found[/red]")
        return

    # meta = json.loads(row.metadata)
    meta = row.metadata if isinstance(row.metadata, dict) else json.loads(row.metadata)
    print("meta", meta)

    bucket = meta["bucket"]
    # bucket = meta.get("bucket", "mlonmcu")
    # obj = meta["object"]
    obj = meta["s3_key"]

    # tmp_file = f"/tmp/{os.path.basename(obj)}"
    # print("bucket", bucket)
    # print("obj", obj)
    # print("tmp_file", tmp_file)

    # s3.download_file(bucket, obj, tmp_file)

    # if obj.endswith(".parquet"):
    #     df = pd.read_parquet(tmp_file)
    # else:
    #     df = pd.read_csv(tmp_file)

    df = pd.read_csv(
        f"s3://{bucket}/{obj}",
        storage_options={
            "key": S3_KEY,
            "secret": S3_SECRET,
            "client_kwargs": {"endpoint_url": S3_ENDPOINT, "region_name": "garage"},
            # "config_kwargs": {"signature_version": 's3', "s3": {'addressing_style': 'path'}},
        },
    )

    console.print(df.head(50))


# -------------------------
# CLI SETUP
# -------------------------


def main():
    parser = argparse.ArgumentParser(description="MLonMCU DB CLI")
    sub = parser.add_subparsers(dest="cmd")

    # list
    p = sub.add_parser("list")
    p.add_argument("--label")
    p.add_argument("--after")
    p.add_argument("--before")
    p.add_argument("--tags", nargs="+")
    p.set_defaults(func=list_experiments)

    # show
    p = sub.add_parser("show")
    p.add_argument("id", type=int)
    p.set_defaults(func=show_experiment)

    # artifacts
    p = sub.add_parser("artifacts")
    p.add_argument("id", type=int)
    p.set_defaults(func=list_artifacts)

    # download
    p = sub.add_parser("download")
    p.add_argument("id", type=int)
    p.add_argument("--name")
    p.add_argument("--out", default="downloads")
    p.set_defaults(func=download_artifacts)

    # report
    p = sub.add_parser("report")
    p.add_argument("id", type=int)
    p.set_defaults(func=show_report)

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
