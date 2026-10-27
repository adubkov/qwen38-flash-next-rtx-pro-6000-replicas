"""shared_ram PLE backend — SGLang plugin.

Shape deliberately copied from the NVMe adapter that ships in the same image
(`/opt/pennyroyal/tools/ple_nvme/ssd_stream`): entry-point registration, an env
gate that makes installing the wheel a no-op for every other deployment, a
source-integrity guard, and REQUIRED-HOOK ENFORCEMENT so an explicitly selected
backend aborts the process instead of silently serving from a private table.

Activated only by:
    PENNY_SHARED_PLE=1
    PENNY_PLE_SHARED_PATH=/ple/layer-0.bin
    (+ PYTHONPATH entry exposing this distribution, and SGLANG_PLUGINS=shared_ram)

Deliberately NOT PENNY_PLE_BACKEND: the image's configs/pennyroyal/ple-backend.sh
owns that variable, accepts only {ram, nvme}, and exits before the plugin loader
runs. shared_ram keeps the ram path's arguments (--ple-offload-embedding,
PLE_OFFLOAD_EMBEDDING=true) and only swaps its allocator, so it must not occupy
the engine's selector.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path

from . import config

HOOK_TARGET = "sglang.srt.models.qwen4_exp._allocate_pinned_table"
GUARD_FILE = "shared-ram-guard.json"


def _bind_manifest_to_checkpoint(manifest_path: str, model_path: str) -> None:
    """Prove the shared table belongs to the checkpoint this process is serving.

    prepare_ple_nvme.py records source.path/config_sha256/index_sha256 for the
    checkpoint it was run against, and until now nothing in the boot path read them.
    A stale-but-self-consistent (payload, manifest) pair therefore passed every other
    check -- size, rows, columns, dtype, and the sha of the payload against itself --
    and served shape-correct, content-wrong n-gram embeddings with no log line. This
    is the ordinary operational case, not a paranoia case: re-prepare from a new
    checkpoint, repoint TARGET_MODEL, leave the tmpfs copy in place, and rows/columns/
    dtype still match, because every checkpoint of this architecture has the same
    table geometry.

    Two hashes, both trivial next to the table (the 30 MB index costs ~1 s), taken
    once per boot from the read-only binds the compose file already provides.
    """
    try:
        with open(manifest_path, "rb") as fh:
            doc = json.loads(fh.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise config.ConfigError("cannot read manifest %s for checkpoint binding: %r"
                                 % (manifest_path, exc))
    source = doc.get("source") or {}
    src_path = (source.get("path") or "").strip()
    want_cfg = str(source.get("config_sha256") or "")
    want_idx = str(source.get("index_sha256") or "")
    if not (src_path and want_cfg and want_idx):
        raise config.ConfigError(
            "manifest %s carries no source.path/config_sha256/index_sha256, so the "
            "shared table cannot be bound to any checkpoint; refusing to serve an "
            "artifact that may belong to a different one" % manifest_path)
    for name, expected, label in (("config.json", want_cfg, "config"),
                                  ("model.safetensors.index.json", want_idx, "index")):
        src_file = os.path.join(src_path, name)
        try:
            actual = _sha256(src_file)
        except OSError as exc:
            raise config.ConfigError("cannot read %s for %s binding: %r" % (src_file, label, exc))
        if actual != expected:
            raise config.ConfigError(
                "%s binding failed: %s is %s but the manifest the shared table was "
                "verified against records %s -- the table was prepared from a "
                "different checkpoint than the one being served"
                % (label, src_file, actual[:16], expected[:16]))
    # And the checkpoint actually loaded by this process must be that same one.
    served_cfg = os.path.join(model_path, "config.json")
    try:
        actual = _sha256(served_cfg)
    except OSError as exc:
        raise config.ConfigError("cannot read served %s for binding: %r" % (served_cfg, exc))
    if actual != want_cfg:
        raise config.ConfigError(
            "served checkpoint %s hashes to %s but the shared table's manifest records "
            "%s -- %s and %s are not the same checkpoint"
            % (served_cfg, actual[:16], want_cfg[:16], model_path, src_path))
    print("[shared_ram_ple] bound to checkpoint %s (config %s, index %s)"
          % (src_path, want_cfg[:16], want_idx[:16]), flush=True)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_guard() -> None:
    """Refuse to patch a build we were not qualified against.

    A silent no-op on a changed engine is the worst outcome: the replica would
    come up in private-RAM mode and the operator would only notice when the host
    thrashes. Same reasoning as the NVMe adapter's SUPPORTED_SGLANG table.
    """
    guard_path = Path(__file__).with_name(GUARD_FILE)
    if not guard_path.exists():
        raise config.ConfigError(
            "%s is missing; this plugin must be deployed with a guard generated "
            "against the running image (deploy-shared-ram.sh)" % guard_path.name)
    guard = json.loads(guard_path.read_text())
    for module_name, expected in guard["modules"].items():
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise config.ConfigError("cannot locate required module %s" % module_name)
        actual = _sha256(spec.origin)
        if actual != expected:
            raise config.ConfigError(
                "engine drift: %s is %s, this plugin was qualified against %s "
                "(image %s)" % (module_name, actual[:16], expected[:16], guard.get("image")))


def _check_prepared_checkpoint_is_overlay(model_path: str) -> None:
    """The shared table is only ever *read*, so the checkpoint must not still carry
    the n-gram weights: if it does, the loader would try to copy 47.68 GiB into a
    mapping we have already write-protected. Abort with a clear message at startup
    instead of taking a SIGSEGV partway through the load."""
    if not model_path:
        raise config.ConfigError("overlay guard invoked without a served checkpoint")
    index = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index):
        raise config.ConfigError(
            "%s has no model.safetensors.index.json, so it cannot be proven that the "
            "served checkpoint drops the n-gram weights; refusing to start rather than "
            "let the loader write into a read-only shared mapping" % model_path)
    try:
        with open(index, "rb") as fh:
            blob = fh.read()
    except OSError as exc:
        raise config.ConfigError("cannot read %s: %r" % (index, exc))
    if b"ngram_embedding.shard_" in blob:
        raise config.ConfigError(
            "%s still contains ngram_embedding.shard_* weights; point TARGET_MODEL at "
            "the prepared overlay (prepare_ple_nvme.py output), whose index drops them"
            % index)


def _replacement(shape, dtype):
    """Drop-in for qwen4_exp._allocate_pinned_table.

    Three-way, deliberately fail-closed in the middle case:
      * exactly the manifest's table shape -> the shared registered mapping;
      * same row width but a different row count -> ABORT. In this code path the
        only [rows,160] fp8 allocation is the PLE table, so a mismatch means the
        manifest and the served checkpoint disagree, and serving anyway would
        gather from rows that are not what the model expects.
      * anything else -> _fallback_allocate, which ABORTS. There is no working
        private fallback: under an explicitly selected backend, silently taking a
        second 47.68 GiB copy is the failure mode this plugin exists to prevent.
    """
    shape = tuple(int(d) for d in shape)
    from .backend import _load_manifest, allocate_shared_table

    path = config.shared_path()
    manifest = config.manifest_path(path)
    table = _load_manifest(manifest)
    want = (int(table["rows"]), int(table["columns"]))

    if len(shape) != 2 or shape[1] != want[1]:
        return _fallback_allocate(shape, dtype)          # clearly not the PLE table
    if shape != want:
        raise config.ConfigError(
            "PLE table shape %s != manifest %s -- checkpoint and prepared manifest "
            "disagree; refusing to serve from the shared table" % (shape, want))
    if dtype not in (torch_dtype_for(table["dtype"]), torch_dtype_for("float8_e4m3fn")):
        return _fallback_allocate(shape, dtype)

    return allocate_shared_table(
        shape, dtype,
        path=path,
        manifest=manifest,
        verify=config.verify_mode(),
        reg_flags=config.reg_flags(),
        log=lambda msg: print("[shared_ram_ple] %s" % msg, flush=True),
    )


def torch_dtype_for(name: str):
    import torch

    return {"float8_e4m3fn": torch.float8_e4m3fn, "bfloat16": torch.bfloat16}.get(name)


_replacement._is_shared_replacement = True


def _fallback_allocate(shape, dtype):
    """There is intentionally no private fallback.

    `_allocate_pinned_table` has exactly one caller in the pinned-host path
    (Qwen4ExpPinnedHostEmbedding.__init__ -> the PLE table), so a foreign shape
    here means the engine is not what this plugin was qualified against. Under an
    explicitly selected backend the plan's rule is to abort rather than let a
    replica silently allocate a second private 47.68 GiB copy -- which is also
    the only way to avoid a self-referential fallback (sglang may apply_hooks
    during import, so any "original" we capture can already be our own wrapper,
    and calling it recursed).
    """
    raise config.ConfigError(
        "unexpected _allocate_pinned_table request shape=%s dtype=%s: this backend "
        "serves only the PLE table; aborting instead of allocating a private copy"
        % (tuple(shape), dtype))


def _install_required_hook_enforcement(HookRegistry) -> None:
    current_descriptor = HookRegistry.__dict__["apply_hooks"]
    current = current_descriptor.__func__
    if getattr(current, "_pennyroyal_shared_enforcement", False):
        return

    def apply_hooks_and_require_shared(cls) -> None:
        registrations = cls._hooks.get(HOOK_TARGET, ())
        if not any(hook_type.name == "REPLACE" and registered_hook is _replacement
                   for hook_type, registered_hook, _ in registrations):
            raise SystemExit(
                "Pennyroyal shared_ram PLE required hook failed registration: " + HOOK_TARGET)
        if not getattr(cls, "_pennyroyal_shared_hooks_applied", False):
            if HOOK_TARGET in cls._patched:
                raise SystemExit(
                    "Pennyroyal shared_ram PLE hook was patched before registration "
                    "at " + HOOK_TARGET)
        current(cls)
        if HOOK_TARGET not in cls._patched:
            # HookRegistry logs and continues after an individual application
            # failure; an explicitly selected backend must stop the process, or
            # the replica serves from a private 47.68 GiB copy and the operator
            # learns only when the host runs out of RAM.
            raise SystemExit(
                "Pennyroyal shared_ram PLE required hook failed to apply: " + HOOK_TARGET)
        cls._pennyroyal_shared_hooks_applied = True

    apply_hooks_and_require_shared._pennyroyal_shared_enforcement = True
    HookRegistry.apply_hooks = classmethod(apply_hooks_and_require_shared)


def register() -> None:
    # Merely installing this distribution must change nothing: sglang auto
    # -discovers entry points even when SGLANG_PLUGINS is unset.
    if not config.active():
        # print, never raise: sglang's loader swallows Exception here and would
        # keep booting, so an inactive config must stay silent-but-visible.
        print("[shared_ram_ple] inactive (set %s=1 to enable); engine backend=%s"
              % (config.BACKEND_ENV, config.engine_backend() or "ram"), flush=True)
        return
    # EVERY failure of an active config must be inside this try. sglang's
    # load_plugins() catches Exception and continues booting, so anything raised
    # outside the wrapper -- including the engine-backend conflict check, which
    # it once was -- is logged and the replica silently serves a private table.
    try:
        config.check_engine_backend()
        _register()
    except BaseException as exc:
        raise SystemExit("Pennyroyal shared_ram PLE initialization failed: %s" % exc) from exc


def _register() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _check_guard()

    spec = importlib.util.find_spec("sglang.srt.models.qwen4_exp")
    if spec is None or spec.origin is None:
        raise config.ConfigError("cannot locate sglang.srt.models.qwen4_exp")
    qwen4 = importlib.import_module("sglang.srt.models.qwen4_exp")
    if not hasattr(qwen4, "_allocate_pinned_table"):
        raise config.ConfigError(
            "sglang.srt.models.qwen4_exp has no _allocate_pinned_table; this engine "
            "build does not use the pinned-host PLE path the shared backend replaces")

    # A writable shared table is only safe to mount read-protected if the engine
    # never loads the n-gram weights into it. Resolve the served checkpoint from
    # the engine's OWN argv first: the recipe composes TARGET_MODEL internally, so
    # an env-only lookup silently resolved to "" and the guard never ran.
    model_path = ""
    try:
        with open("/proc/self/cmdline", "rb") as fh:
            argv = [a.decode("utf-8", "replace") for a in fh.read().split(b"\0") if a]
        for i, arg in enumerate(argv):
            if arg == "--model-path" and i + 1 < len(argv):
                model_path = argv[i + 1]
                break
            if arg.startswith("--model-path="):
                model_path = arg.split("=", 1)[1]
                break
    except OSError:
        pass
    if not model_path:
        for env in ("TARGET_MODEL", "TARGET_MODEL_DIR", "MODEL_PATH", "PENNY_TARGET_MODEL"):
            if os.environ.get(env):
                model_path = os.environ[env]
                break
    if not model_path:
        raise config.ConfigError(
            "cannot determine the served checkpoint (no --model-path in argv, no "
            "TARGET_MODEL/MODEL_PATH in env); the overlay guard must run rather than "
            "be skipped, because serving a checkpoint that still carries the n-gram "
            "weights would have the loader write into a mapping we protect read-only")
    _check_prepared_checkpoint_is_overlay(model_path)
    _bind_manifest_to_checkpoint(config.manifest_path(config.shared_path()), model_path)

    if getattr(HookRegistry, "_pennyroyal_shared_hooks_registered", False):
        return
    # NOTE: no "original" is captured on purpose. sglang can load plugins and run
    # apply_hooks during package import, so the attribute we would capture may
    # already be our own wrapper; calling it as a fallback recursed (RecursionError).
    # The allocator has a single legitimate caller, so _fallback_allocate aborts.
    HookRegistry.register(HOOK_TARGET, _replacement, HookType.REPLACE)
    _install_required_hook_enforcement(HookRegistry)
    HookRegistry._pennyroyal_shared_hooks_registered = True
    print("[shared_ram_ple] registered %s -> shared table (%s)" % (HOOK_TARGET, config.shared_path()),
          flush=True)
