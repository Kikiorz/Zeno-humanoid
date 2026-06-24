#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def read_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()

    token = sys.stdin.readline().strip()
    if not token:
        raise SystemExit("No Hugging Face token provided on stdin or HF_TOKEN.")
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a local folder to a Hugging Face dataset repo.")
    parser.add_argument("--folder", required=True, help="Local folder to upload.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. user/repo.")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    parser.add_argument("--path-in-repo", default=None)
    parser.add_argument("--commit-message", default="Upload 20260623 robot rosbag data")
    parser.add_argument("--public", action="store_true", help="Create repo as public instead of private.")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    token = read_token()
    api = HfApi(token=token)
    print(f"Creating or reusing {args.repo_type} repo: {args.repo_id} (private={not args.public})", flush=True)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=not args.public,
        exist_ok=True,
        token=token,
    )

    print(f"Uploading folder: {folder}", flush=True)
    info = api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(folder),
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
        token=token,
    )
    print(f"Upload complete: {info.commit_url}", flush=True)


if __name__ == "__main__":
    main()
