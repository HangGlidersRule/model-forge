#!/usr/bin/env python3
"""Stop a process at the first publishing syscall on one path and swap that path.

The swap happens after metadata checks (stat, lstat, statx) and before link or
open uses the path. A builder that looks the path up again, then links or
copies whatever is there now, publishes the swapped symlink. A builder that
keeps the inode it already opened, or refuses when that use sees a symlink,
does not.

``--when link`` swaps at the first link or linkat call instead, whatever path
it names. A builder that opens each input before it links any of them is past
its checks by then, so a link that names the input path again, rather than the
descriptor it holds, publishes the swapped symlink.
"""

from __future__ import annotations

import ctypes
import os
import signal
import sys

AT_EMPTY_PATH = 0x1000
AT_FDCWD = -100
PTRACE_GETREGS = 12
PTRACE_SETOPTIONS = 0x4200
PTRACE_SYSCALL = 24
PTRACE_O_TRACESYSGOOD = 1
PTRACE_TRACEME = 0
SIGTRAP = 5
SYSCALL_STOP = SIGTRAP | 0x80

# x86_64 syscall numbers. Publishing calls only; stat-family checks are ignored
# so the swap lands between a metadata check and the later use.
_OPEN = 2
_LINK = 86
_OPENAT = 257
_LINKAT = 265
_OPENAT2 = 437


class _Regs(ctypes.Structure):
    _fields_ = [
        ("r15", ctypes.c_ulonglong),
        ("r14", ctypes.c_ulonglong),
        ("r13", ctypes.c_ulonglong),
        ("r12", ctypes.c_ulonglong),
        ("rbp", ctypes.c_ulonglong),
        ("rbx", ctypes.c_ulonglong),
        ("r11", ctypes.c_ulonglong),
        ("r10", ctypes.c_ulonglong),
        ("r9", ctypes.c_ulonglong),
        ("r8", ctypes.c_ulonglong),
        ("rax", ctypes.c_ulonglong),
        ("rcx", ctypes.c_ulonglong),
        ("rdx", ctypes.c_ulonglong),
        ("rsi", ctypes.c_ulonglong),
        ("rdi", ctypes.c_ulonglong),
        ("orig_rax", ctypes.c_ulonglong),
        ("rip", ctypes.c_ulonglong),
        ("cs", ctypes.c_ulonglong),
        ("eflags", ctypes.c_ulonglong),
        ("rsp", ctypes.c_ulonglong),
        ("ss", ctypes.c_ulonglong),
        ("fs_base", ctypes.c_ulonglong),
        ("gs_base", ctypes.c_ulonglong),
        ("ds", ctypes.c_ulonglong),
        ("es", ctypes.c_ulonglong),
        ("fs", ctypes.c_ulonglong),
        ("gs", ctypes.c_ulonglong),
    ]


_libc = ctypes.CDLL(None, use_errno=True)
_libc.ptrace.restype = ctypes.c_long
_libc.ptrace.argtypes = [
    ctypes.c_ulong,
    ctypes.c_ulong,
    ctypes.c_ulong,
    ctypes.c_ulong,
]


def _ptrace(request: int, pid: int, addr: int, data: int) -> int:
    result = _libc.ptrace(request, pid, addr, data)
    if result < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), f"ptrace {request} pid {pid}")
    return result


def _read_cstr(pid: int, address: int) -> bytes:
    if address == 0:
        return b""
    fd = os.open(f"/proc/{pid}/mem", os.O_RDONLY)
    try:
        buf = os.pread(fd, 4096, address)
    except OSError:
        return b""
    finally:
        os.close(fd)
    return buf.split(b"\0", 1)[0]


def _signed(value: int) -> int:
    if value >= 2**63:
        return value - 2**64
    return value


def _resolve(pid: int, raw: bytes, dirfd: int | None) -> str:
    if not raw:
        return ""
    text = os.fsdecode(raw)
    if text.startswith("/"):
        return os.path.normpath(text)
    if dirfd is None or dirfd == AT_FDCWD:
        base = os.readlink(f"/proc/{pid}/cwd")
    else:
        try:
            base = os.readlink(f"/proc/{pid}/fd/{dirfd}")
        except OSError:
            return os.path.normpath(text)
    return os.path.normpath(os.path.join(base, text))


def _use_path(pid: int, regs: _Regs) -> str | None:
    """Return the source path of a publishing syscall, or None to ignore it."""
    number = regs.orig_rax
    if number == _OPEN:
        return _resolve(pid, _read_cstr(pid, regs.rdi), None)
    if number == _LINK:
        return _resolve(pid, _read_cstr(pid, regs.rdi), None)
    if number in (_OPENAT, _OPENAT2):
        return _resolve(pid, _read_cstr(pid, regs.rsi), _signed(regs.rdi))
    if number == _LINKAT:
        raw = _read_cstr(pid, regs.rsi)
        if raw == b"" and (regs.r8 & AT_EMPTY_PATH):
            return None
        return _resolve(pid, raw, _signed(regs.rdi))
    return None


def _is_link(regs: _Regs) -> bool:
    return regs.orig_rax in (_LINK, _LINKAT)


def _swap(source: str, outside: str) -> None:
    if os.path.lexists(source):
        os.unlink(source)
    os.symlink(outside, source)


def _trace(source: str, outside: str, command: list[str], when: str = "use") -> int:
    source = os.path.abspath(source)
    outside = os.path.abspath(outside)
    pid = os.fork()
    if pid == 0:
        try:
            _ptrace(PTRACE_TRACEME, 0, 0, 0)
            os.kill(os.getpid(), signal.SIGSTOP)
            os.execv(command[0], command)
        except OSError as exc:
            os.write(2, f"tracer child: {exc}\n".encode())
        os._exit(127)

    exited = False
    try:
        _, status = os.waitpid(pid, 0)
        if not os.WIFSTOPPED(status):
            exited = True
            return 1 if not os.WIFEXITED(status) else os.WEXITSTATUS(status)
        _ptrace(PTRACE_SETOPTIONS, pid, 0, PTRACE_O_TRACESYSGOOD)
        _ptrace(PTRACE_SYSCALL, pid, 0, 0)
        entering = False
        swapped = False
        while True:
            _, status = os.waitpid(pid, 0)
            if os.WIFEXITED(status):
                exited = True
                return os.WEXITSTATUS(status)
            if os.WIFSIGNALED(status):
                exited = True
                return 128 + os.WTERMSIG(status)
            if not os.WIFSTOPPED(status):
                exited = True
                return 1
            stopsig = os.WSTOPSIG(status)
            if stopsig != SYSCALL_STOP:
                deliver = 0 if stopsig in (SIGTRAP, signal.SIGSTOP) else stopsig
                _ptrace(PTRACE_SYSCALL, pid, 0, deliver)
                continue
            if not entering:
                entering = True
                if not swapped:
                    regs = _Regs()
                    _ptrace(PTRACE_GETREGS, pid, 0, ctypes.addressof(regs))
                    if when == "link":
                        hit = _is_link(regs)
                    else:
                        hit = _use_path(pid, regs) == source
                    if hit:
                        _swap(source, outside)
                        swapped = True
            else:
                entering = False
            _ptrace(PTRACE_SYSCALL, pid, 0, 0)
    finally:
        if not exited:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)


def main(argv: list[str]) -> int:
    when = "use"
    if len(argv) > 2 and argv[1] == "--when":
        when = argv[2]
        argv = [argv[0], *argv[3:]]
    if when not in ("use", "link") or len(argv) < 4 or "--" not in argv:
        sys.stderr.write(
            "usage: symlink_use_tracer.py [--when use|link] SOURCE OUTSIDE -- COMMAND...\n"
        )
        return 2
    split = argv.index("--")
    source, outside = argv[1], argv[2]
    command = argv[split + 1 :]
    if not command:
        sys.stderr.write("tracer: missing command\n")
        return 2
    return _trace(source, outside, command, when)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
