"""The shared_ram PLE table: one physical shmem mapping, registered per process.

This replaces ONLY the fork's `_allocate_pinned_table()` for the PLE table. Every
other step of the pinned-host path (`Qwen4ExpPinnedHostEmbedding`, its `gather()`,
the Triton kernel that dereferences the raw host pointer) is left untouched, which
is what keeps the blast radius small and `ram` mode a bit-for-bit fallback.

Measured contract this code assumes (see `apps/tools/ple-spike/`, 2026-10-03):
  * substrate MUST be shmem (tmpfs / memfd). A file on a block device — the page
    cache — is rejected by cudaHostRegister with cudaErrorInvalidValue even when
    the VMA is writable.
  * the mapping MUST be CPU-writable (PROT_READ|PROT_WRITE). Read-only VMAs cannot
    be registered at all on this driver: cudaDevAttrHostRegisterReadOnlySupported=0,
    so 0x08/0x0b return cudaErrorNotSupported and 0x03 on a PROT_READ VMA returns
    cudaErrorInvalidValue.
  * flags 0x01|0x02 (PORTABLE|MAPPED) -> cudaSuccess; /proc/vmstat
    nr_foll_pin_acquired advances by exactly bytes/4096; cudaHostGetDevicePointer
    returns dp == host ptr, so the gather kernel keeps taking the host pointer.

The table is written by the preparer, never by a serving process: we open O_RDWR
only because the driver demands a writable VMA, then mprotect the mapping read-only
after registering (the device reads the already-pinned pages, not this VMA's
permissions) and abort if that downgrade fails. Integrity is then proven, not
assumed: the sha256 is recomputed THROUGH the mapping after it went read-only, and
the identity recorded in diagnostics() is the fstat of the descriptor that was
actually mapped -- never a second lookup of the pathname, which a republish between
open and check could make describe a file this process is not reading.

Measured 2026-10-06, so the next line is fact and not inference: on tmpfs neither
open(O_RDWR) nor a writable MAP_SHARED mapping advances mtime/ctime, while a store
through the mapping advances both. The live payload took such a store during a
replica's cudaHostRegister window, its bytes verified unchanged afterwards. So
timestamps are an alarm, never an identity: identity is (dev, inode, size) from
fstat, content is the sha256.

Residual risk inherent to sharing, not to this code: one physical copy means one
corruption event takes out both replicas, and because read-only registration is
unsupported on this driver the pages are device-writable, so a rogue kernel in
either replica can spoil the shared table. The cover is ple-shared-ram-check.sh and
its hourly host timer: they compare every replica's VMA for the payload against the
host inode and re-hash the file against the manifest off the serving path (the boot
hash costs ~13 s; measured 69 s host-side at nice 19 / idle io class).
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import os
import time

import torch

from . import config

# Mirrors the fork's _PINNED_TABLE_BUFFERS: a shared table lives as long as the
# process, so mappings are never unmapped and registrations never revoked.
_SHARED_TABLES: list = []
_STATE: dict = {}


def _libcudart():
    for name in ("libcudart.so.13", "libcudart.so", "libcudart.so.13.0"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise config.ConfigError("libcudart is not loadable in this process")


def _vmstat_pin() -> dict:
    out = {}
    try:
        with open("/proc/vmstat") as fh:
            for line in fh:
                parts = line.split()
                if parts and parts[0] in ("nr_foll_pin_acquired", "nr_foll_pin_released"):
                    out[parts[0]] = int(parts[1])
    except OSError:
        pass
    return out


def _mprotect_read_only(addr: int, length: int) -> object:
    """Downgrade the CPU side of the mapping after registration.

    Returns None on success, else the errno-ish failure. Device access is unaffected:
    the kernel reads the pages the driver already pinned, not this VMA's permissions.
    A failure is FATAL to the boot (see allocate_shared_table): a shared table that
    stays CPU-writable in a serving process means one stray write corrupts every
    replica at once, so "logged and continued" is the wrong shape for this check.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PROT_READ = 0x1
        page = 0x1000
        aligned = (addr + page - 1) & ~(page - 1)
        if libc.mprotect(ctypes.c_void_p(aligned), ctypes.c_size_t(length), ctypes.c_int(PROT_READ)) != 0:
            return "mprotect rc!=0"
        return None
    except Exception as exc:  # pragma: no cover
        return repr(exc)


def _load_manifest(manifest_path: str) -> dict:
    try:
        with open(manifest_path, "rb") as fh:
            manifest = json.loads(fh.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise config.ConfigError("cannot read manifest %s: %r" % (manifest_path, exc))
    if manifest.get("format") != "sglang-ssd-stream":
        raise config.ConfigError(
            "unexpected manifest format %r (expected sglang-ssd-stream)" % manifest.get("format"))
    tables = manifest.get("tables") or []
    if len(tables) != 1:
        raise config.ConfigError("expected exactly 1 PLE table in manifest, got %d" % len(tables))
    return tables[0]


def _verify_payload(path: str, table: dict, mode: str, log) -> None:
    """Fail closed unless the artifact is exactly what the manifest says it is."""
    expected_bytes = int(table["bytes"])
    expected_sha = str(table.get("sha256") or "")
    try:
        actual = os.path.getsize(path)
    except OSError as exc:
        raise config.ConfigError("shared table missing: %s (%r)" % (path, exc))
    if actual != expected_bytes:
        raise config.ConfigError(
            "shared table size %d != manifest %d -- refuse to serve from a partial or "
            "stale artifact" % (actual, expected_bytes))
    if mode == "none":
        log("verify=none (accepted size only; sha256 NOT checked)")
        return
    if mode != "full":
        # "sample" used to hash strided blocks and return without comparing them:
        # it verified nothing while looking like it did, and every artifact for this
        # architecture is the same size, so a stale table would have passed.
        raise config.ConfigError("verify mode %r is not supported (full|none)" % mode)
    if not expected_sha:
        raise config.ConfigError("manifest carries no sha256; refuse to skip verification")
    # The authoritative hash is taken AFTER registration and mprotect, THROUGH the
    # mapping (see allocate_shared_table): hashing the path and then reopening by
    # path leaves a window in which a writer can change the bytes we serve.
    log("verify=full: size ok, sha256 will be checked through the mapping")


def _hash_mapping(buf, nbytes: int, chunk: int = 1 << 24) -> str:
    """sha256 of the bytes the serving process can actually read."""
    digest = hashlib.sha256()
    for off in range(0, nbytes, chunk):
        digest.update(buf[off:off + chunk])
    return digest.hexdigest()


def allocate_shared_table(shape, dtype, *, path: str, manifest: str,
                          verify: str = "full",
                          reg_flags: int = None, log=print):
    """Return a device-readable view of the ONE shared table.

    Deliberately narrow: it only ever serves the PLE n-gram shape. Anything else
    falls through to the caller's private allocation, so a model that grows a
    second host table cannot silently end up sharing the wrong buffer.
    """
    from .config import EXPECTED_DTYPE, EXPECTED_WIDTH

    shape = tuple(int(d) for d in shape)
    if dtype not in (torch.float8_e4m3fn, torch.uint8):
        raise config.ConfigError("shared backend expects fp8/uint8, got %s" % dtype)
    if len(shape) != 2 or shape[1] != EXPECTED_WIDTH:
        raise config.ConfigError(
            "shared backend only serves [rows,%d] PLE tables, got %s" % (EXPECTED_WIDTH, shape))

    table = _load_manifest(manifest)
    expected_sha = str(table.get("sha256") or "")
    if int(table["rows"]) != shape[0] or int(table["columns"]) != shape[1]:
        raise config.ConfigError(
            "requested shape %s != manifest [%d,%d]" % (shape, table["rows"], table["columns"]))
    if str(table["dtype"]) != EXPECTED_DTYPE:
        raise config.ConfigError("manifest dtype %r != %r" % (table["dtype"], EXPECTED_DTYPE))
    if int(table["row_start"]) != 0:
        raise config.ConfigError("shared backend requires row_start 0, got %s" % table["row_start"])

    _verify_payload(path, table, verify, log)

    flags = config.DEFAULT_REG_FLAGS if reg_flags is None else int(reg_flags)
    nbytes = shape[0] * shape[1]
    if nbytes % 0x1000:
        raise config.ConfigError("registered length %d is not page-aligned" % nbytes)

    fd = os.open(path, os.O_RDWR)          # writable VMA is a driver requirement
    try:
        # Identity must come from the descriptor that is about to be mapped. A later
        # os.stat(path) re-resolves the pathname, so a republish in the open->check
        # window would leave the logged inode describing some other file.
        mapped_st = os.fstat(fd)
        buf = mmap.mmap(fd, nbytes, mmap.MAP_SHARED,
                        mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)

    try:
        raw = torch.frombuffer(buf, dtype=torch.uint8)
    except Exception:
        import numpy as np
        raw = torch.from_numpy(np.frombuffer(buf, dtype=np.uint8))
    _SHARED_TABLES.append(buf)

    lib = _libcudart()
    lib.cudaHostRegister.restype = ctypes.c_int
    lib.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    before = _vmstat_pin()
    t0 = time.perf_counter()
    rc = lib.cudaHostRegister(ctypes.c_void_p(raw.data_ptr()), ctypes.c_size_t(nbytes),
                              ctypes.c_uint(flags))
    reg_s = time.perf_counter() - t0
    after = _vmstat_pin()
    if rc != 0:
        buf.close()
        _SHARED_TABLES.remove(buf)
        raise config.ConfigError(
            "cudaHostRegister(0x%02x) on %s failed rc=%d; the shared backend cannot "
            "serve and will NOT fall back to a private table" % (flags, path, rc))

    def _drop(reason: str):
        buf.close()
        if buf in _SHARED_TABLES:
            _SHARED_TABLES.remove(buf)
        raise config.ConfigError(reason)

    dp = ctypes.c_void_p(0)
    gdp_rc = lib.cudaHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(raw.data_ptr()), 0)
    eq_host = bool(dp.value and dp.value == raw.data_ptr())
    if gdp_rc != 0 or not eq_host:
        # With PageableMemoryAccess=1 a wrong device pointer does not fault loudly:
        # the kernel can be served through page tables at a fraction of the rate,
        # which looks like a slow GPU rather than a broken backend. Never allow it.
        _drop("cudaHostGetDevicePointer(rc=%d) returned %s for host %s -- the shared "
              "backend cannot guarantee the gather reads pinned pages"
              % (gdp_rc, hex(dp.value or 0), hex(raw.data_ptr())))

    pin_delta = after.get("nr_foll_pin_acquired", 0) - before.get("nr_foll_pin_acquired", 0)
    expected_pin = nbytes // 0x1000
    if pin_delta < expected_pin * 0.98:
        _drop("registered but not page-locked: nr_foll_pin_acquired advanced %d for %d "
              "pages -- treating as failure" % (pin_delta, expected_pin))

    # Always applied, and its failure is fatal. It used to be gated on
    # PENNY_PLE_SHARED_STRICT and best-effort, which meant a replica could serve
    # with the shared mapping still CPU-writable.
    mp = _mprotect_read_only(raw.data_ptr(), nbytes)
    if mp is not None:
        _drop("mprotect(PROT_READ) failed (%s): a shared table must not stay "
              "CPU-writable in the serving process" % mp)

    st = os.stat(path)
    if (st.st_ino, st.st_dev, st.st_size) != (mapped_st.st_ino, mapped_st.st_dev,
                                              mapped_st.st_size):
        _drop("shared table was replaced under us: the mapped object is (dev=%s "
              "inode=%d size=%d) but %s now resolves to (dev=%s inode=%d size=%d) -- "
              "the bytes this process reads are not the bytes the manifest describes"
              % (mapped_st.st_dev, mapped_st.st_ino, mapped_st.st_size, path,
                 st.st_dev, st.st_ino, st.st_size))
    got = None
    if verify == "full":
        t1 = time.time()
        got = _hash_mapping(buf, nbytes)
        if got != expected_sha:
            _drop("shared table sha256 through the mapping %s != manifest %s (%.1fs): "
                  "the bytes this process reads are not the bytes that were prepared"
                  % (got[:16], expected_sha[:16], time.time() - t1))
        log("verify=full ok sha256=%s through the mapping in %.1fs"
            % (got[:16], time.time() - t1))

    _STATE.update({
        "path": path, "manifest": manifest, "inode": mapped_st.st_ino,
        "dev": mapped_st.st_dev,
        "bytes": nbytes, "shape": shape, "reg_flags": flags, "reg_rc": rc,
        "reg_seconds": round(reg_s, 3), "gdp_rc": gdp_rc, "devptr_eq_host": eq_host,
        "pin_pages_delta": pin_delta, "pin_pages_expected": expected_pin,
        "mprotect": mp, "pid": os.getpid(), "sha256_through_mapping": got,
        "file_size": mapped_st.st_size, "file_mtime_ns": mapped_st.st_mtime_ns,
        "file_ctime_ns": mapped_st.st_ctime_ns, "verify": verify,
        # The pathname re-resolved at the end of allocation. Equal to the mapped
        # identity above unless something republished the payload mid-boot -- and
        # that case already dropped the boot. Kept so an operator can see both.
        "path_inode": st.st_ino, "path_dev": st.st_dev, "path_size": st.st_size,
        "path_mtime_ns": st.st_mtime_ns, "path_ctime_ns": st.st_ctime_ns,
    })
    log("shared_ram PLE: inode=%s %s bytes=%.2f GiB registered in %.2fs "
        "pin_pages=%d/%d devptr_eq_host=True mprotect=ok sha=%s"
        % (mapped_st.st_ino, os.path.basename(path), nbytes / 2**30, reg_s,
           pin_delta, expected_pin, (got[:16] if got else "not-checked")))
    return raw.view(dtype).view(shape)


def diagnostics() -> dict:
    return dict(_STATE)
