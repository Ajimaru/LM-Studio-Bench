#include "intel_loader.h"

namespace probe {

IntelLibraryLoad load_intel_library(IntelLibraryLoader loader) {
    if (!loader) {
        return {nullptr, ERROR_INVALID_PARAMETER};
    }
    SetLastError(ERROR_SUCCESS);
    const auto module = loader(
        L"ControlLib.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    return {module, module ? ERROR_SUCCESS : GetLastError()};
}

}  // namespace probe
