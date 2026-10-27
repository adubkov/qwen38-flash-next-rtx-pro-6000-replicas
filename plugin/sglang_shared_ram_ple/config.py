"""Environment knobs for the shared_ram PLE backend.

Nothing here changes default behaviour: the plugin activates ONLY when
PENNY_SHARED_PLE=1 is set explicitly, so `ram` remains the untouched reference
and rollback.

Why our own variable: PENNY_PLE_BACKEND is owned by the image
(configs/pennyroyal/ple-backend.sh), which validates a closed {ram, nvme}
domain and exits before the plugin loader ever runs. That script also decides
whether --ple-offload-embedding is passed; shared_ram *keeps* ram's semantics
(the engine still builds the host table, we only replace its allocator), so the
engine selector stays "ram" and this plugin is selected by an independent knob.
"""
from __future__ import annotations

import os

BACKEND_ENV = "PENNY_SHARED_PLE"            # 1|on|true -> use the shared table
ENGINE_BACKEND_ENV = "PENNY_PLE_BACKEND"    # the image's own selector (must stay ram)
PATH_ENV = "PENNY_PLE_SHARED_PATH"          # .../ple/layer-0.bin (the prepared payload)
MANIFEST_ENV = "PENNY_PLE_SHARED_MANIFEST"  # .../ssd-stream.json (defaults alongside PATH)
VERIFY_ENV = "PENNY_PLE_SHARED_VERIFY"      # full | none  (default full)
FLAGS_ENV = "PENNY_PLE_REG_FLAGS"           # override cudaHostRegister flags (debug)
# There is deliberately no strict/writable-mapping knob: post-registration
# mprotect(PROT_READ) is unconditional and fatal on failure, and "sample" hashing
# was removed because it compared nothing.

# Measured on this box (spike, 2026-10-03): PORTABLE|MAPPED on a CPU-writable
# shmem VMA -> cudaSuccess, nr_foll_pin_acquired advances by exactly the byte
# length / 4096, and cudaHostGetDevicePointer returns dp == host ptr.
DEFAULT_REG_FLAGS = 0x01 | 0x02

EXPECTED_WIDTH = 160          # columns in the manifest (fp8 bytes per stored row)
EXPECTED_DTYPE = "float8_e4m3fn"


class ConfigError(RuntimeError):
    """Raised for any misconfiguration. The plugin turns this into a startup
    abort: an explicitly selected backend that cannot install must never fall
    through to the private per-process table."""


def enabled_value() -> str:
    return os.environ.get(BACKEND_ENV, "").strip().lower()


def engine_backend() -> str:
    return os.environ.get(ENGINE_BACKEND_ENV, "").strip().lower()


def active() -> bool:
    return enabled_value() in ("1", "true", "yes", "on")


def check_engine_backend() -> None:
    """shared_ram replaces the allocator used by the engine's *ram* path.

    Running it on top of the image's nvme backend would mean two plugins both
    claiming the PLE table, so refuse rather than let that race happen.
    """
    eng = engine_backend()
    if eng not in ("", "ram"):
        raise ConfigError("%s=1 requires %s unset or ram (got %r); the nvme backend "
                          "already replaces the PLE path"
                          % (BACKEND_ENV, ENGINE_BACKEND_ENV, eng))


def shared_path() -> str:
    p = os.environ.get(PATH_ENV, "").strip()
    if not p:
        raise ConfigError("%s must be set when %s=1" % (PATH_ENV, BACKEND_ENV))
    return p


def manifest_path(path: str) -> str:
    override = os.environ.get(MANIFEST_ENV, "").strip()
    if override:
        return override
    # prepare_ple_nvme.py writes ssd-stream.json at the prepared-checkpoint root,
    # with the payload at ple/layer-<n>.bin beneath it.
    for cand in (os.path.join(os.path.dirname(path), "..", "ssd-stream.json"),
                 os.path.join(os.path.dirname(os.path.dirname(path)), "ssd-stream.json")):
        if os.path.exists(cand):
            return os.path.normpath(cand)
    raise ConfigError("cannot locate ssd-stream.json near %s (set %s)" % (path, MANIFEST_ENV))


def verify_mode() -> str:
    v = os.environ.get(VERIFY_ENV, "full").strip().lower()
    if v not in ("full", "none"):
        raise ConfigError('%s must be "full" or "none", got %r' % (VERIFY_ENV, v))
    return v


def reg_flags() -> int:
    raw = os.environ.get(FLAGS_ENV, "").strip()
    return int(raw, 0) if raw else DEFAULT_REG_FLAGS
