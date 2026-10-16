// DXGI identity and system-wide WDDM/PDH adapter memory.
#pragma once

#include "probe_types.h"

#include <memory>
#include <string_view>

namespace probe {

std::string utf8(std::wstring_view text);
std::vector<Device> enumerate_devices(std::vector<std::string>& errors);

class WddmMemory {
public:
    WddmMemory();
    ~WddmMemory();
    WddmMemory(const WddmMemory&) = delete;
    WddmMemory& operator=(const WddmMemory&) = delete;
    void sample(std::vector<Device>& devices,
                std::vector<std::string>& errors);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace probe
