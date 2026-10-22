"""
research/lifelong_01/ssrl_checkpoints.py -- Size-bounded checkpoint capture and verification.

Implements checkpoint serialization with strict safety invariants:
1. Checkpoint filenames restricted to explicit allowlist.
2. Individual checkpoint file size strictly capped (< 100 KB).
3. Serialization consumes zero training RNG state (torch and random RNG states asserted identical).
4. Serialization does not mutate model state.
5. Checkpoint is reloaded and asserted bit-for-bit identical before being approved.
6. Canonical tensor fingerprints and SHA-256 file hashes recorded in diagnostics.
"""

from __future__ import annotations
import os
import random
import hashlib
from typing import Dict, Any, Tuple
import torch


ALLOWED_CHECKPOINT_FILENAMES = {
    # SSRL-02 sequential checkpoints
    "checkpoint_t0_init.pt",
    "checkpoint_t1_post_a.pt",
    "checkpoint_t2_seq_ab.pt",
    "checkpoint_t2_fresh_b.pt",
    "checkpoint_t2_freeze_a.pt",
    # SSRL-03 joint reference checkpoints
    "checkpoint_joint_t0_init.pt",
    "checkpoint_joint_step500.pt",
    "checkpoint_joint_step1000.pt",
    "checkpoint_joint_step2000.pt",
    "checkpoint_joint_step2500_final.pt",
    # SSRL-04 replay baseline checkpoints
    "checkpoint_replay_step625_final.pt",
}

MAX_CHECKPOINT_FILE_BYTES: int = 100 * 1024  # 100 KB ceiling per checkpoint


def compute_tensor_fingerprint_from_state_dict(state_dict: Dict[str, Any]) -> str:
    """Computes deterministic SHA-256 fingerprint from tensor parameter bytes."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode("utf-8"))
        tensor = state_dict[key].cpu().detach().contiguous()
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def compute_file_sha256(path: str) -> str:
    """Computes SHA-256 hash of bytes on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def capture_and_verify_checkpoint(
    state_dict: Dict[str, Any],
    run_dir: str,
    filename: str,
) -> Dict[str, Any]:
    """
    Saves a model state_dict, verifying RNG independence, non-mutation, size limit,
    and reload integrity.
    """
    if filename not in ALLOWED_CHECKPOINT_FILENAMES:
        raise ValueError(
            f"Forbidden checkpoint filename '{filename}'. "
            f"Must be one of: {sorted(list(ALLOWED_CHECKPOINT_FILENAMES))}"
        )

    out_path = os.path.join(run_dir, filename)
    if os.path.exists(out_path):
        raise FileExistsError(f"Checkpoint already exists, refusing overwrite: {out_path}")

    # 1. Snapshot RNG states before save
    torch_rng_before = torch.get_rng_state()
    py_rng_before = random.getstate()

    # 2. Compute canonical tensor fingerprint before save
    tensor_fingerprint = compute_tensor_fingerprint_from_state_dict(state_dict)

    # 3. Save checkpoint
    torch.save(state_dict, out_path)

    # 4. Assert RNG state was not mutated by serialization
    torch_rng_after = torch.get_rng_state()
    py_rng_after = random.getstate()
    if not torch.equal(torch_rng_before, torch_rng_after):
        raise RuntimeError(f"Checkpoint serialization mutated torch RNG state: {filename}")
    if py_rng_before != py_rng_after:
        raise RuntimeError(f"Checkpoint serialization mutated python RNG state: {filename}")

    # 5. Check file size ceiling
    size_bytes = os.path.getsize(out_path)
    if size_bytes > MAX_CHECKPOINT_FILE_BYTES:
        raise ValueError(
            f"Checkpoint {filename} size ({size_bytes} bytes) exceeds limit ({MAX_CHECKPOINT_FILE_BYTES} bytes)."
        )

    # 6. Compute file byte hash
    file_sha256 = compute_file_sha256(out_path)

    # 7. Reload and verify bit-for-bit equality
    reloaded = torch.load(out_path, weights_only=True)
    if set(reloaded.keys()) != set(state_dict.keys()):
        raise RuntimeError(f"Reloaded checkpoint keys mismatch for {filename}")

    for k in state_dict:
        if not torch.equal(state_dict[k], reloaded[k]):
            raise RuntimeError(f"Reloaded tensor value mismatch for key '{k}' in {filename}")

    reloaded_fingerprint = compute_tensor_fingerprint_from_state_dict(reloaded)
    if reloaded_fingerprint != tensor_fingerprint:
        raise RuntimeError(f"Reloaded tensor fingerprint mismatch for {filename}")

    return {
        "filename": filename,
        "path": out_path,
        "size_bytes": size_bytes,
        "file_sha256": file_sha256,
        "tensor_fingerprint": tensor_fingerprint,
    }
