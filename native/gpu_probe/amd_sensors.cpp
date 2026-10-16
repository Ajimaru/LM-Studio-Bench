#include "vendor_sensors.h"
#include "vendor_status.h"

#include <ADLXHelper.h>
#include <IPerformanceMonitoring.h>
#include <ISystem2.h>

#include <algorithm>
#include <map>
#include <set>
#include <utility>

namespace probe {
namespace {

void sensor_error(Device& device, const Metric& error) {
    device.metrics[temperature] = error;
    device.metrics[power] = error;
}

void vendor_error(std::vector<Device>& devices, const Metric& error) {
    for (auto& device : devices) {
        if (device.vendor_id == 0x1002) {
            sensor_error(device, error);
        }
    }
}

Metric support_result(ADLX_RESULT result, adlx_bool supported,
                      const std::string& name) {
    if (result != ADLX_OK) {
        return adlx_failure(result, name);
    }
    if (!supported) {
        return missing(Status::unsupported, "ADLX " + name,
                       "The AMD driver explicitly reports this sensor "
                       "unsupported.");
    }
    return missing(Status::unavailable, "ADLX " + name,
                   "Waiting for the current sensor sample.");
}

}  // namespace

struct AmdSensors::Impl {
    // Reverse member destruction releases every interface before Terminate.
    ADLXHelper helper;
    adlx::IADLXPerformanceMonitoringServicesPtr service;
    std::map<AdapterId, adlx::IADLXGPU2Ptr> gpus;
    std::set<AdapterId> ambiguous;

    Impl(std::vector<Device>& devices, std::vector<std::string>& errors) {
        const bool present = std::any_of(
            devices.begin(), devices.end(),
            [](const Device& device) { return device.vendor_id == 0x1002; });
        if (!present) {
            return;
        }
        const auto initialized = helper.Initialize();
        if (initialized != ADLX_OK) {
            vendor_error(devices, adlx_failure(
                initialized, "ADLX Initialize (installed AMD driver)", false));
            return;
        }
        auto* system = helper.GetSystemServices();
        if (!system) {
            vendor_error(devices, missing(Status::error, "ADLX",
                                         "ADLX returned no system interface."));
            return;
        }
        const auto performance =
            system->GetPerformanceMonitoringServices(&service);
        if (performance != ADLX_OK || service == nullptr) {
            vendor_error(devices, adlx_failure(
                performance, "GetPerformanceMonitoringServices", false));
            return;
        }
        adlx::IADLXGPUListPtr list;
        const auto listed = system->GetGPUs(&list);
        if (listed != ADLX_OK || list == nullptr) {
            vendor_error(devices, adlx_failure(listed, "GetGPUs", false));
            return;
        }
        if (list->Size() > 256) {
            vendor_error(devices, missing(Status::error, "ADLX",
                                         "Too many AMD adapters returned."));
            return;
        }
        for (adlx_uint i = list->Begin(); i < list->End(); ++i) {
            adlx::IADLXGPUPtr gpu;
            const auto at = list->At(i, &gpu);
            if (at != ADLX_OK || gpu == nullptr) {
                errors.push_back(
                    adlx_failure(at, "GPU list At", false).message);
                continue;
            }
            void* raw = nullptr;
            const auto queried =
                gpu->QueryInterface(adlx::IADLXGPU2::IID(), &raw);
            if (queried != ADLX_OK || !raw) {
                errors.push_back(adlx_failure(
                    queried, "QueryInterface(IADLXGPU2) for LUID identity",
                    false).message);
                continue;
            }
            adlx::IADLXGPU2Ptr gpu2;
            gpu2.Attach(static_cast<adlx::IADLXGPU2*>(raw));
            ADLX_LUID luid{};
            const auto identified = gpu2->LUID(&luid);
            if (identified != ADLX_OK) {
                errors.push_back(
                    adlx_failure(identified, "GPU LUID", false).message);
                continue;
            }
            const AdapterId id = {static_cast<std::uint32_t>(luid.highPart),
                                  static_cast<std::uint32_t>(luid.lowPart)};
            auto* device = match_device(devices, id, 0x1002);
            if (!device) {
                errors.push_back("ADLX adapter " + id.text() +
                                 " has no unique AMD DXGI LUID match.");
                continue;
            }
            if (gpus.count(id) || ambiguous.count(id)) {
                ambiguous.insert(id);
                gpus.erase(id);
                sensor_error(*device, missing(
                    Status::error, "ADLX",
                    "Multiple AMD GPUs returned the same LUID; no readings "
                    "will be assigned."));
                continue;
            }
            const char* version = nullptr;
            if (gpu2->AMDWindowsDriverVersion(&version) == ADLX_OK &&
                version && *version) {
                device->driver_version = std::string(version);
            }
            gpus.emplace(id, gpu2);
        }
        for (auto& device : devices) {
            if (device.vendor_id == 0x1002 && !gpus.count(device.luid) &&
                !ambiguous.count(device.luid)) {
                sensor_error(device, missing(
                    Status::error, "ADLX LUID association",
                    "AMD adapter could not be matched by LUID. A compatible "
                    "ADLX driver with IADLXGPU2::LUID is required; vendor "
                    "names and enumeration order are not used as fallbacks."));
            }
        }
    }

    void sample(Device& device, adlx::IADLXGPU2* gpu) {
        adlx::IADLXGPUMetricsSupportPtr support;
        const auto supported = service->GetSupportedGPUMetrics(gpu, &support);
        if (supported != ADLX_OK || support == nullptr) {
            sensor_error(device, adlx_failure(
                supported, "GetSupportedGPUMetrics", false));
            return;
        }
        adlx_bool has_temperature = false;
        auto result = support->IsSupportedGPUTemperature(&has_temperature);
        device.metrics[temperature] = support_result(
            result, has_temperature, "IsSupportedGPUTemperature");
        bool hotspot = false;
        if (device.metrics[temperature].status == Status::unsupported) {
            result =
                support->IsSupportedGPUHotspotTemperature(&has_temperature);
            auto alternative = support_result(
                result, has_temperature, "IsSupportedGPUHotspotTemperature");
            hotspot = alternative.status == Status::unavailable;
            device.metrics[temperature] = fallback(
                device.metrics[temperature], std::move(alternative));
        }
        adlx_bool has_power = false;
        result = support->IsSupportedGPUPower(&has_power);
        device.metrics[power] =
            support_result(result, has_power, "IsSupportedGPUPower");
        bool board_power = false;
        if (device.metrics[power].status == Status::unsupported) {
            result = support->IsSupportedGPUTotalBoardPower(&has_power);
            auto alternative = support_result(
                result, has_power, "IsSupportedGPUTotalBoardPower");
            board_power = alternative.status == Status::unavailable;
            device.metrics[power] =
                fallback(device.metrics[power], std::move(alternative));
        }

        if (device.metrics[temperature].status != Status::unavailable &&
            device.metrics[power].status != Status::unavailable) {
            return;
        }
        adlx::IADLXGPUMetricsPtr current;
        result = service->GetCurrentGPUMetrics(gpu, &current);
        if (result != ADLX_OK || current == nullptr) {
            const auto error = adlx_failure(
                result, "GetCurrentGPUMetrics", false);
            for (auto metric : {temperature, power}) {
                if (device.metrics[metric].status == Status::unavailable) {
                    device.metrics[metric] = error;
                }
            }
            return;
        }
        if (device.metrics[temperature].status == Status::unavailable) {
            adlx_double value = 0.0;
            result = hotspot ? current->GPUHotspotTemperature(&value)
                             : current->GPUTemperature(&value);
            const std::string operation =
                hotspot ? "GPUHotspotTemperature" : "GPUTemperature";
            device.metrics[temperature] =
                result == ADLX_OK
                    ? reading(value, "ADLX " + operation, false)
                    : adlx_failure(result, operation);
        }
        if (device.metrics[power].status == Status::unavailable) {
            adlx_double value = 0.0;
            result = board_power ? current->GPUTotalBoardPower(&value)
                                 : current->GPUPower(&value);
            const std::string operation =
                board_power ? "GPUTotalBoardPower" : "GPUPower";
            device.metrics[power] =
                result == ADLX_OK ? reading(value, "ADLX " + operation)
                                  : adlx_failure(result, operation);
        }
    }
};

AmdSensors::AmdSensors(std::vector<Device>& devices,
                       std::vector<std::string>& errors)
    : impl_(std::make_unique<Impl>(devices, errors)) {}
AmdSensors::~AmdSensors() = default;

void AmdSensors::sample(std::vector<Device>& devices) {
    for (auto& device : devices) {
        const auto it = impl_->gpus.find(device.luid);
        if (device.vendor_id == 0x1002 && it != impl_->gpus.end()) {
            impl_->sample(device, it->second);
        }
    }
}

}  // namespace probe
