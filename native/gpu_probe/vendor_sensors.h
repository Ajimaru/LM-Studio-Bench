// Read-only vendor sensors, matched exclusively through verified DXGI LUIDs.
#pragma once

#include "probe_types.h"

#include <memory>

namespace probe {

class AmdSensors {
public:
    AmdSensors(std::vector<Device>& devices,
               std::vector<std::string>& errors);
    ~AmdSensors();
    AmdSensors(const AmdSensors&) = delete;
    AmdSensors& operator=(const AmdSensors&) = delete;
    void sample(std::vector<Device>& devices);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class IntelSensors {
public:
    IntelSensors(std::vector<Device>& devices,
                 std::vector<std::string>& errors, SamplingEpoch& epoch,
                 std::chrono::milliseconds interval);
    ~IntelSensors();
    IntelSensors(const IntelSensors&) = delete;
    IntelSensors& operator=(const IntelSensors&) = delete;
    void sample(std::vector<Device>& devices);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace probe
