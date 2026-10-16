"""Windows job limits for owned, suspended code-test processes."""

from __future__ import annotations

import ctypes
import sys
from types import TracebackType
from typing import Self


def _windows_error() -> OSError:
    """Capture the Windows last-error value without importing APIs on Unix."""
    if sys.platform != "win32":
        return OSError("Windows error information is unavailable")
    return ctypes.WinError(ctypes.get_last_error())


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_int64),
        ("job_time", ctypes.c_int64),
        ("flags", ctypes.c_uint32),
        ("min_working_set", ctypes.c_size_t),
        ("max_working_set", ctypes.c_size_t),
        ("active_processes", ctypes.c_uint32),
        ("affinity", ctypes.c_size_t),
        ("priority", ctypes.c_uint32),
        ("scheduling", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("read_operations", ctypes.c_uint64),
        ("write_operations", ctypes.c_uint64),
        ("other_operations", ctypes.c_uint64),
        ("read_bytes", ctypes.c_uint64),
        ("write_bytes", ctypes.c_uint64),
        ("other_bytes", ctypes.c_uint64),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", _BasicLimits),
        ("io", _IoCounters),
        ("process_memory", ctypes.c_size_t),
        ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t),
        ("peak_job_memory", ctypes.c_size_t),
    ]


class WindowsJob:
    """Own a job that limits CPU/memory and kills its children when closed.

    The benchmark process is never assigned to the job. Only its explicitly
    created, still-suspended code-test child is assigned before it can run.
    """

    _kernel: ctypes.CDLL
    _handle: int | None

    def __init__(self, memory_bytes: int, cpu_seconds: int) -> None:
        """Create and configure the job.

        Args:
            memory_bytes: Combined committed-memory limit of job processes.
            cpu_seconds: Combined user-mode CPU-time limit.

        Raises:
            ValueError: If either limit is not positive.
            OSError: If unavailable or Windows cannot enforce the limits.
        """
        if memory_bytes <= 0 or cpu_seconds <= 0:
            raise ValueError("Code-test resource limits must be positive")
        if sys.platform != "win32":
            raise OSError("Windows job objects require Windows")
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        kernel.CreateJobObjectW.restype = ctypes.c_void_p
        kernel.SetInformationJobObject.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32
        ]
        kernel.SetInformationJobObject.restype = ctypes.c_int32
        kernel.OpenProcess.argtypes = [
            ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32
        ]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.AssignProcessToJobObject.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p
        ]
        kernel.AssignProcessToJobObject.restype = ctypes.c_int32
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle.restype = ctypes.c_int32
        self._kernel = kernel
        self._handle = kernel.CreateJobObjectW(None, None)
        if not self._handle:
            raise _windows_error()

        limits = _ExtendedLimits()
        # JOB_TIME | JOB_MEMORY | KILL_ON_JOB_CLOSE.
        limits.basic.flags = 0x00000004 | 0x00000200 | 0x00002000
        limits.basic.job_time = cpu_seconds * 10_000_000
        limits.job_memory = memory_bytes
        try:
            if not kernel.SetInformationJobObject(
                self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            ):
                raise _windows_error()
        except OSError:
            self.close()
            raise

    def assign(self, pid: int) -> None:
        """Assign the owned child before resuming its initial thread.

        Args:
            pid: PID returned by the caller's suspended subprocess creation.

        Raises:
            ValueError: If the PID is invalid or the job has been closed.
            OSError: If Windows refuses process access or job assignment.
        """
        if pid <= 0 or self._handle is None:
            raise ValueError("An open job and a positive child PID are required")
        # PROCESS_SET_QUOTA | PROCESS_TERMINATE, required for job assignment.
        process = self._kernel.OpenProcess(0x0100 | 0x0001, False, pid)
        if not process:
            raise _windows_error()
        try:
            if not self._kernel.AssignProcessToJobObject(self._handle, process):
                raise _windows_error()
        finally:
            if not self._kernel.CloseHandle(process):
                raise _windows_error()

    def close(self) -> None:
        """Close the job, terminating any code-test descendants still running.

        Raises:
            OSError: If Windows cannot close the job handle.
        """
        if self._handle is not None:
            if not self._kernel.CloseHandle(self._handle):
                raise _windows_error()
            self._handle = None

    def __enter__(self) -> Self:
        """Return the configured job."""
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Release the job and its owned processes."""
        self.close()
