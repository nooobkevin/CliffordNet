from __future__ import annotations

import signal
from pathlib import Path

import lightning as L


def resolve_resume_checkpoint(resume: str | None, run_dir: Path, output_dir: Path) -> str | None:
    if resume is None or str(resume).lower() in {"", "none", "false", "no"}:
        return None
    if str(resume).lower() != "auto":
        return str(Path(resume).expanduser())

    candidates = []
    checkpoint_dir = run_dir / "checkpoints"
    for name in ("last.ckpt", "preempted.ckpt"):
        candidate = checkpoint_dir / name
        if candidate.exists():
            candidates.append(candidate)
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


class PreemptionCheckpoint(L.Callback):
    def __init__(self, checkpoint_dir: str | Path):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self._preempted = False
        self._previous_sigterm_handler = None

    def setup(self, trainer, pl_module, stage=None):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle_sigterm)

    def teardown(self, trainer, pl_module, stage=None):
        if self._previous_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self._previous_sigterm_handler)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._preempted:
            return
        if trainer.is_global_zero:
            trainer.save_checkpoint(str(self.checkpoint_dir / "preempted.ckpt"))
        raise SystemExit(143)

    def on_exception(self, trainer, pl_module, exception):
        text = str(exception).lower()
        if "out of memory" in text or "cuda oom" in text:
            return
        if trainer.is_global_zero:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            try:
                trainer.save_checkpoint(str(self.checkpoint_dir / "exception.ckpt"))
            except Exception as exc:
                print(f"[checkpoint] failed to save exception checkpoint: {exc}")

    def _handle_sigterm(self, signum, frame):
        self._preempted = True
