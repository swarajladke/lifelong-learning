"""
research/lifelong_01/publish_results.py -- Automated results publishing utility for SSRL.

Publishes experiment artifacts from Kaggle executions to the dedicated
'experiment-results' branch of the repository:
1. Operates on a dedicated, isolated publishing clone in a temporary directory.
2. Code on 'main' remains strictly untouched at its pinned commit.
3. Stages only the explicit path: runs/ssrl_02/<unique_run_id>/.
4. Strictly forbids force-pushes, pushing to 'main', or overwriting existing run directories.
5. Guards against credential exposure: credentials never appear in command-line arguments,
   git remote URLs, environment prints, logs, or committed artifacts.
6. Enforces a 10 MB maximum directory size ceiling to protect Git history.
7. Refuses large weight checkpoints (.pt, .ckpt, .bin) and credentials (.env, keys).
8. Reports experiment outcome and upload outcome as strictly separate statuses.
"""

from __future__ import annotations
import os
import sys
import shutil
import tempfile
import argparse
import subprocess
from typing import Optional, Dict, Any


MAX_ALLOWED_RUN_BYTES: int = 10 * 1024 * 1024  # 10 MB ceiling
FORBIDDEN_EXTENSIONS = {".pt", ".pth", ".bin", ".ckpt", ".pkl"}
FORBIDDEN_SUBSTRINGS = {"id_rsa", "id_ed25519", ".env", "token", "secret", "credential"}


def get_token(secret_name: str = "GITHUB_RESULTS_TOKEN") -> Optional[str]:
    """Retrieves GitHub PAT from Kaggle secrets or environment variable."""
    # 1. Try Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret(secret_name)
        if tok and len(tok.strip()) > 0:
            return tok.strip()
    except Exception:
        pass
    # 2. Try environment variable
    tok = os.environ.get(secret_name)
    if tok and len(tok.strip()) > 0:
        return tok.strip()
    return None


def calculate_and_verify_run_size(run_dir: str) -> int:
    """Calculates total size of run directory and asserts size ceiling."""
    total_bytes = 0
    for root, _, files in os.walk(run_dir):
        for f in files:
            fp = os.path.join(root, f)
            total_bytes += os.path.getsize(fp)
    if total_bytes > MAX_ALLOWED_RUN_BYTES:
        raise ValueError(
            f"Run artifacts size ({total_bytes} bytes) exceeds the 10 MB ceiling ({MAX_ALLOWED_RUN_BYTES} bytes). "
            f"Upload to Git is refused."
        )
    return total_bytes


def verify_no_forbidden_files(run_dir: str) -> None:
    """Rejects large binary checkpoints and credential-like files."""
    for root, _, files in os.walk(run_dir):
        for f in files:
            f_lower = f.lower()
            ext = os.path.splitext(f_lower)[1]
            if ext in FORBIDDEN_EXTENSIONS:
                raise ValueError(f"Forbidden checkpoint file detected in run artifacts: {f}")
            for sub in FORBIDDEN_SUBSTRINGS:
                if sub in f_lower:
                    raise ValueError(f"Forbidden credential-like filename detected in run artifacts: {f}")


def publish_run_artifacts(
    run_dir: str,
    run_id: str,
    repo_url: str,
    source_commit: str,
    experiment_exit_code: int,
    target_branch: str = "experiment-results",
    secret_name: str = "GITHUB_RESULTS_TOKEN",
) -> Dict[str, Any]:
    """
    Publishes run artifacts to the dedicated results branch in a fresh clone.
    Returns status dictionary with separate experiment and upload outcomes.
    """
    assert target_branch != "main", "Publishing to 'main' branch is strictly forbidden."
    abs_run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(abs_run_dir):
        raise FileNotFoundError(f"Run directory not found: {abs_run_dir}")

    publish_status: Dict[str, Any] = {
        "run_id": run_id,
        "source_commit": source_commit,
        "experiment_exit_code": experiment_exit_code,
        "upload_attempted": False,
        "upload_successful": False,
        "github_url": None,
        "error_message": None,
    }

    # 1. Enforce size and file guards
    try:
        run_bytes = calculate_and_verify_run_size(abs_run_dir)
        verify_no_forbidden_files(abs_run_dir)
        publish_status["run_bytes"] = run_bytes
    except Exception as e:
        publish_status["error_message"] = f"Pre-upload validation failed: {e}"
        print(f"[Publish Error] {publish_status['error_message']}")
        return publish_status

    # 2. Retrieve authentication token
    token = get_token(secret_name)
    if not token:
        publish_status["error_message"] = f"Kaggle secret '{secret_name}' is not configured or empty."
        print(f"[Publish Note] Automated upload skipped: {publish_status['error_message']}")
        return publish_status

    publish_status["upload_attempted"] = True

    # 3. Create isolated temporary publishing workspace
    publish_workspace = tempfile.mkdtemp(prefix="ssrl_publish_")
    askpass_sh = os.path.join(publish_workspace, "askpass.sh")
    try:
        # Create secure askpass script
        with open(askpass_sh, "w") as f:
            f.write('#!/bin/sh\necho "$RESULTS_TOKEN"\n')
        os.chmod(askpass_sh, 0o700)

        git_env = os.environ.copy()
        git_env["GIT_ASKPASS"] = askpass_sh
        git_env["RESULTS_TOKEN"] = token

        # Normalise remote URL for askpass (use https without embedded tokens)
        clean_repo_url = repo_url
        if "@" in clean_repo_url and clean_repo_url.startswith("https://"):
            clean_repo_url = "https://" + clean_repo_url.split("@", 1)[1]

        # Use dummy username so askpass supplies token as password
        auth_url = clean_repo_url.replace("https://", "https://x-access-token@")

        # Check if remote target branch exists
        ls_proc = subprocess.run(
            ["git", "ls-remote", "--heads", auth_url, target_branch],
            capture_output=True,
            text=True,
            env=git_env,
        )
        branch_exists = (ls_proc.returncode == 0 and target_branch in ls_proc.stdout)

        clone_dir = os.path.join(publish_workspace, "results_clone")

        if branch_exists:
            # Clone only the target results branch
            clone_cmd = [
                "git", "clone", "--branch", target_branch, "--depth", "1",
                auth_url, clone_dir
            ]
            subprocess.run(clone_cmd, check=True, capture_output=True, env=git_env)
        else:
            # Initialize dedicated orphan results branch
            clone_cmd = ["git", "clone", auth_url, clone_dir]
            subprocess.run(clone_cmd, check=True, capture_output=True, env=git_env)
            subprocess.run(
                ["git", "-C", clone_dir, "checkout", "--orphan", target_branch],
                check=True,
                capture_output=True,
                env=git_env,
            )
            subprocess.run(
                ["git", "-C", clone_dir, "rm", "-rf", "."],
                check=True,
                capture_output=True,
                env=git_env,
            )
            readme_path = os.path.join(clone_dir, "README.md")
            with open(readme_path, "w") as f:
                f.write(
                    "# SSRL Experiment Results Branch\n\n"
                    "Dedicated branch for immutable run artifacts and diagnostics.\n"
                )
            subprocess.run(
                ["git", "-C", clone_dir, "add", "README.md"],
                check=True,
                capture_output=True,
                env=git_env,
            )
            subprocess.run(
                ["git", "-C", clone_dir, "commit", "-m", "chore: initialize experiment-results branch"],
                check=True,
                capture_output=True,
                env=git_env,
            )

        # 4. Destination path in dedicated results branch
        target_run_dir = os.path.join(clone_dir, "runs", "ssrl_02", run_id)
        if os.path.exists(target_run_dir):
            raise FileExistsError(
                f"Run directory already exists on '{target_branch}': {target_run_dir}. "
                f"Refusing to overwrite historical run artifacts."
            )

        os.makedirs(target_run_dir, exist_ok=False)

        # Copy run artifacts
        for item in os.listdir(abs_run_dir):
            src_item = os.path.join(abs_run_dir, item)
            dst_item = os.path.join(target_run_dir, item)
            if os.path.isfile(src_item):
                shutil.copy2(src_item, dst_item)
            elif os.path.isdir(src_item):
                shutil.copytree(src_item, dst_item)

        # Configure minimal committer metadata
        subprocess.run(
            ["git", "-C", clone_dir, "config", "user.name", "Kaggle Automated Runner"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", clone_dir, "config", "user.email", "kaggle-runner@users.noreply.github.com"],
            check=True,
            capture_output=True,
        )

        # Stage ONLY the current run's explicit path
        rel_target = os.path.relpath(target_run_dir, clone_dir)
        subprocess.run(
            ["git", "-C", clone_dir, "add", rel_target],
            check=True,
            capture_output=True,
            env=git_env,
        )

        commit_msg = (
            f"chore(results): record SSRL-02 run {run_id}\n\n"
            f"Source Commit: {source_commit}\n"
            f"Experiment Exit Code: {experiment_exit_code}\n"
            f"Artifact Size: {run_bytes} bytes\n"
        )
        subprocess.run(
            ["git", "-C", clone_dir, "commit", "-m", commit_msg],
            check=True,
            capture_output=True,
            env=git_env,
        )

        # Push to experiment-results branch without force
        push_cmd = ["git", "-C", clone_dir, "push", "origin", target_branch]
        push_res = subprocess.run(push_cmd, capture_output=True, text=True, env=git_env)
        if push_res.returncode != 0:
            raise RuntimeError(f"Git push to '{target_branch}' failed: {push_res.stderr}")

        publish_status["upload_successful"] = True
        publish_status["github_url"] = (
            f"https://github.com/swarajladke/lifelong-learning/tree/{target_branch}/runs/ssrl_02/{run_id}"
        )
        print(f"[Publish Success] Run artifacts published to: {publish_status['github_url']}")

    except Exception as e:
        publish_status["error_message"] = str(e)
        print(f"[Publish Failure] Failed to publish results to GitHub: {e}")
        print("[Publish Fallback] Local artifact bundle remains preserved on disk.")
    finally:
        shutil.rmtree(publish_workspace, ignore_errors=True)

    return publish_status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish SSRL experiment artifacts to dedicated branch")
    parser.add_argument("--run_dir", required=True, help="Directory containing run artifacts")
    parser.add_argument("--run_id", required=True, help="Unique run identifier")
    parser.add_argument("--repo_url", default="https://github.com/swarajladke/lifelong-learning.git", help="Repository URL")
    parser.add_argument("--source_commit", required=True, help="Commit SHA of code tested")
    parser.add_argument("--exit_code", type=int, required=True, help="Exit code of experiment execution")
    parser.add_argument("--secret_name", default="GITHUB_RESULTS_TOKEN", help="Kaggle secret name")
    args = parser.parse_args()

    res = publish_run_artifacts(
        run_dir=args.run_dir,
        run_id=args.run_id,
        repo_url=args.repo_url,
        source_commit=args.source_commit,
        experiment_exit_code=args.exit_code,
        secret_name=args.secret_name,
    )
    # Never exit nonzero solely because upload failed: experiment exit code is primary
    print(f"[Publish Status Summary] Attempted: {res['upload_attempted']}, Succeeded: {res['upload_successful']}")
