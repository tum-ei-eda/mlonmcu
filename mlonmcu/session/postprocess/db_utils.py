#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Artifacts defintions internally used to refer to intermediate results."""

import os
import json
from pathlib import Path
from typing import Optional, List, Set, Tuple

from tqdm import tqdm

from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.logging import get_logger

logger = get_logger()

# TODO: share with client codebase

BUCKET = "mlonmcu"


def connect_to_postgres_db():
    logger.debug("Connecting to Postgres DB")

    DB_USER = os.environ.get("POSTGRES_USER", "mlonmcu")
    DB_PASS = os.environ.get("POSTGRES_PASSWORD", None)
    DB_NAME = os.environ.get("POSTGRES_DB", "mlonmcu")
    DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    DB_PORT = os.environ.get("POSTGRES_PORT", 5432)
    assert DB_PASS is not None

    PG_CONFIG = {
        "host": DB_HOST,
        "port": DB_PORT,
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASS,
    }

    import psycopg2

    conn = psycopg2.connect(**PG_CONFIG)
    logger.debug("Connected to Postgres DB: %s:%s [DB: %s]", DB_HOST, DB_PORT, DB_NAME)

    return conn


def connect_to_garage():
    logger.debug("Connecting to Garage (S3)")

    S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:3900")
    REGION = os.environ.get("REGION", "garage")
    S3_KEY = os.environ.get("S3_KEY")
    S3_SECRET = os.environ.get("S3_SECRET")
    assert S3_KEY is not None
    assert S3_SECRET is not None

    S3_CONFIG = {
        "endpoint_url": S3_ENDPOINT,
        "aws_access_key_id": S3_KEY,
        "aws_secret_access_key": S3_SECRET,
        "region_name": REGION,
    }

    import boto3

    s3 = boto3.client("s3", **S3_CONFIG)
    # TODO: close connection?
    logger.debug("Connected to Garage (S3): %s [Region: %s]", S3_ENDPOINT, REGION)
    return s3


def upload_report(s3, cur, conn, exp_id, report_df):
    report_bytes = report_df.to_csv(index=False).encode()
    report_key = f"experiments/{exp_id}/report.csv"
    s3.put_object(
        Bucket=BUCKET,
        Key=report_key,
        Body=report_bytes,
    )
    logger.debug("Uploaded report to S3: %s", report_key)
    cur.execute(
        """
        INSERT INTO artifacts (experiment_id, type, size, metadata)
        VALUES (%s, %s, %s, %s)
        """,
        (
            exp_id,
            "report",
            len(report_bytes),
            json.dumps({"s3_key": report_key, "bucket": BUCKET}),
        ),
    )
    conn.commit()
    logger.debug("Registered report in db: %s", report_key)


def upload_artifact(s3, cur, conn, exp_id, artifact: Tuple[str, str], run_idx: Optional[int] = None):
    artifact_name, artifact_path = artifact
    assert artifact_path is not None
    assert Path(artifact_path).is_file(), f"Not a file: {artifact_path}"
    if run_idx is not None:
        artifact_name = f"runs/{run_idx}/{artifact_name}"
    artifact_key = f"experiments/{exp_id}/{artifact_name}"
    s3.upload_file(artifact_path, BUCKET, artifact_key)
    logger.debug("Uploaded artifact to S3: %s", artifact_key)
    size = os.path.getsize(artifact_path)
    cur.execute(
        """
        INSERT INTO artifacts (experiment_id, type, size, metadata)
        VALUES (%s, %s, %s, %s)
        """,
        (
            exp_id,
            artifact_name,
            size,
            json.dumps({"s3_key": artifact_key, "bucket": BUCKET}),
        ),
    )
    conn.commit()
    logger.debug("Registered artifact in db: %s", artifact_key)


def push_session_to_mlonmcu_db(
    report,
    session_artifacts: Optional[List[Tuple[str, str]]] = None,
    run_artifacts: Optional[List[Tuple[str, str]]] = None,
    config_hash: Optional[str] = None,
    label: Optional[str] = None,
    timestamp: Optional[str] = None,
    tags: Set[str] = None,
    progress: bool = False,
):
    # TODO
    conn = connect_to_postgres_db()
    cur = conn.cursor()

    # Create experiment in DB
    logger.debug("Creating experiment in db")

    if tags is None:
        tags = []
    else:
        tags = sorted(list(tags))
    cur.execute(
        """
        INSERT INTO experiments (label, timestamp, config_hash, tags)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (label, timestamp, config_hash, json.dumps(tags)),
    )

    exp_id = cur.fetchone()[0]
    conn.commit()

    logger.debug("Created experiment %d", exp_id)

    s3 = connect_to_garage()

    report_df = report.df

    upload_report(s3, cur, conn, exp_id, report_df)

    if session_artifacts:
        for artifact in tqdm(session_artifacts, disable=not progress, desc="Uploading Session Artifacts"):
            upload_artifact(s3, cur, conn, exp_id, artifact)

    if run_artifacts:
        for run_idx, run_artifacts_ in tqdm(
            run_artifacts.items(), disable=not progress, desc="Uploading Run Artifacts"
        ):
            for artifact in run_artifacts_:
                upload_artifact(s3, cur, conn, exp_id, artifact, run_idx=run_idx)

    logger.debug("Closing DB connections")
    cur.close()
    conn.close()
    logger.debug("Closed DB connections")
