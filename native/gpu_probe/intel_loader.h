// The public IGCL v1 entry point follows Intel's pinned release wrapper.
#pragma once

#include <Windows.h>

namespace probe {

struct IntelLibraryLoad {
    HMODULE module = nullptr;
    DWORD error = ERROR_SUCCESS;
};

using IntelLibraryLoader = decltype(&LoadLibraryExW);

// No CWD, application-directory, PATH or undocumented runtime override.
IntelLibraryLoad load_intel_library(
    IntelLibraryLoader loader = &LoadLibraryExW);

}  // namespace probe
