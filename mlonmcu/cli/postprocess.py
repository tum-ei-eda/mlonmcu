#
# Copyright (c) 2026 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
"""Apply a session postprocess to an archived MLonMCU session."""

from pathlib import Path

import pandas as pd
import yaml

from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.cli.common import add_common_options, add_context_options
from mlonmcu.cli.helper.parse import extract_config
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.report import Report
from mlonmcu.session.postprocess import get_postprocesses
from mlonmcu.session.postprocess.postprocess import RunPostprocess, SessionPostprocess


# These columns are produced by Run.get_report().  A report is persisted as one
# dataframe, so use the known boundaries to recreate the three report sections
# expected by session postprocesses.
PRE_COLUMNS = {"Session", "Run", "Model", "Frontend", "Framework", "Backend", "Platform", "Target"}
POST_COLUMNS = {"Features", "Config", "Postprocesses", "Stages", "Comment", "Failing", "Reason"}


def load_report(path):
    """Load a persisted report and restore its pre/main/post dataframe layout."""
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported report format: {path.suffix}")

    report = Report()
    report.pre_df = df[[column for column in df.columns if column in PRE_COLUMNS]].copy()
    report.post_df = df[[column for column in df.columns if column in POST_COLUMNS]].copy()
    report.main_df = df[[column for column in df.columns if column not in PRE_COLUMNS | POST_COLUMNS]].copy()
    return report


def get_session_artifacts(session_dir):
    """Restore the session metadata artifacts supplied to session postprocesses."""
    artifacts = []
    for name, flag in (("label.txt", "label"), ("timestamp.txt", "timestamp")):
        path = Path(session_dir) / name
        if path.is_file():
            artifacts.append(Artifact(name, path=path, fmt=ArtifactFormat.PATH, flags=(flag,)))
    return artifacts


def get_run_artifacts(run_dir):
    """Restore the artifact paths and flags recorded for an archived run."""
    artifacts_file = Path(run_dir) / "artifacts.yml"
    if not artifacts_file.is_file():
        return []
    with open(artifacts_file, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    artifacts = []
    for item in data.get("artifacts", []):
        path = item.get("path")
        if path is None:
            path = Path(run_dir) / item["name"]
        else:
            path = Path(path)
        if path.exists():
            # Artifact contents are not saved in artifacts.yml. PATH preserves
            # the on-disk artifact while retaining the flags used by run
            # postprocesses to locate their inputs.
            artifacts.append(
                Artifact(item["name"], path=path, fmt=ArtifactFormat.PATH, flags=tuple(item.get("flags", [])))
            )
    return artifacts


def apply_run_postprocess(postprocess, report, session_dir):
    """Apply a run postprocess to every row of an archived session report."""
    run_ids = report.pre_df["Run"] if "Run" in report.pre_df else report.main_df.index
    for row, run_id in enumerate(run_ids):
        run_report = Report()
        run_report.pre_df = report.pre_df.iloc[[row]].copy()
        run_report.main_df = report.main_df.iloc[[row]].copy()
        run_report.post_df = report.post_df.iloc[[row]].copy()
        run_dir = Path(session_dir) / "runs" / str(run_id)
        artifacts = postprocess.post_run(run_report, get_run_artifacts(run_dir)) or []
        for artifact in artifacts:
            artifact.export(run_dir)
        for attr, run_df in (
            ("pre_df", run_report.pre_df),
            ("main_df", run_report.main_df),
            ("post_df", run_report.post_df),
        ):
            report_df = getattr(report, attr)
            new_columns = run_df.columns.difference(report_df.columns)
            if len(new_columns) > 0:
                report_df = pd.concat(
                    [report_df, pd.DataFrame(pd.NA, index=report_df.index, columns=new_columns)], axis=1
                )
                setattr(report, attr, report_df)
            report_df.loc[report_df.index[row], run_df.columns] = run_df.iloc[0].values


def add_postprocess_options(parser):
    postprocess_parser = parser.add_argument_group("postprocess options")
    postprocess_parser.add_argument(
        "postprocess",
        choices=get_postprocesses().keys(),
        help="Session postprocess to apply",
    )
    postprocess_parser.add_argument(
        "-s",
        "--session",
        metavar="SESSION",
        type=int,
        default=-1,
        help="Saved session id to postprocess (default: latest session)",
    )
    postprocess_parser.add_argument(
        "-o",
        "--output",
        metavar="REPORT",
        type=Path,
        default=None,
        help="Write the postprocessed report to REPORT instead of the saved session report",
    )
    postprocess_parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print the postprocessed report to stdout",
    )


def get_parser(subparsers):
    """Define the parser for the postprocess subcommand."""
    parser = subparsers.add_parser("postprocess", description="Apply a postprocess to a saved MLonMCU session.")
    parser.set_defaults(func=handle)
    add_common_options(parser)
    add_context_options(parser)
    # Keep the regular configuration syntax: -c filter_cols.drop=Config.
    parser.add_argument("-c", "--config", metavar="KEY=VALUE", nargs="+", action="append")
    add_postprocess_options(parser)
    return parser


def handle(args):
    with MlonMcuContext(path=args.home, deps_lock="read") as context:
        if not context.sessions:
            raise RuntimeError("There are no saved sessions in this environment")
        session = (
            context.sessions[-1]
            if args.session == -1
            else next((item for item in context.sessions if item.idx == args.session), None)
        )
        if session is None:
            available = ", ".join(str(item.idx) for item in context.sessions)
            raise RuntimeError(f"Session {args.session} was not found (available: {available})")

        postprocess_cls = get_postprocesses()[args.postprocess]
        report_path = next(
            (path for path in (session.dir / "report.csv", session.dir / "report.xlsx") if path.is_file()), None
        )
        if report_path is None:
            raise RuntimeError(f"Saved report does not exist in {session.dir}")
        report = load_report(report_path)
        config, _ = extract_config(args)
        postprocess = postprocess_cls(config={**context.environment.vars, **config})
        if isinstance(postprocess, SessionPostprocess):
            artifacts = postprocess.post_session(report, get_session_artifacts(session.dir)) or []
        elif isinstance(postprocess, RunPostprocess):
            apply_run_postprocess(postprocess, report, session.dir)
            artifacts = []
        else:
            raise RuntimeError(f"Unsupported postprocess type: {type(postprocess).__name__}")

        output = args.output if args.output is not None else report_path
        output = Path(output)
        report.export(output)
        for artifact in artifacts:
            artifact.export(output.parent)
        if args.print_report:
            print(report.df)
