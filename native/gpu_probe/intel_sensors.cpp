#include "intel_loader.h"
#include "vendor_sensors.h"
#include "vendor_status.h"

#include <Windows.h>

#include <algorithm>
#include <iostream>
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
        if (device.vendor_id == 0x8086) {
            sensor_error(device, error);
        }
    }
}

struct IntelAdapter {
    IntelAdapter(ctl_device_adapter_handle_t device,
                 std::chrono::milliseconds maximum_gap)
        : handle(device), telemetry_energy(maximum_gap),
          domain_energy(maximum_gap) {}

    ctl_device_adapter_handle_t handle = nullptr;
    ctl_pwr_handle_t last_power_domain = nullptr;
    EnergyDelta telemetry_energy;
    EnergyDelta domain_energy;
};

}  // namespace

struct IntelSensors::Impl {
    HMODULE module = nullptr;
    ctl_api_handle_t api = nullptr;
    std::map<AdapterId, IntelAdapter> adapters;
    std::set<AdapterId> ambiguous;
    SamplingEpoch& sampling_epoch;
    std::chrono::milliseconds maximum_gap;

    template <typename Function, typename... Args>
    ctl_result_t invoke(const char* name, Args... args) const {
        if (!module) {
            return CTL_RESULT_ERROR_NOT_INITIALIZED;
        }
        const auto function =
            reinterpret_cast<Function>(GetProcAddress(module, name));
        if (!function) {
            // A missing export is an implementation error, not a sensor
            // support flag. Keep the named operation in the diagnostic.
            return CTL_RESULT_ERROR_NOT_IMPLEMENTED;
        }
        return function(args...);
    }

    Impl(std::vector<Device>& devices, std::vector<std::string>& errors,
         SamplingEpoch& epoch, std::chrono::milliseconds interval)
        : sampling_epoch(epoch),
          // Respect configured long intervals, with one second of jitter.
          maximum_gap(interval + std::chrono::seconds(1)) {
        const bool present = std::any_of(
            devices.begin(), devices.end(),
            [](const Device& device) { return device.vendor_id == 0x8086; });
        if (!present) {
            return;
        }
        // Intel documents System32 as the public entry point. DriverStore
        // copies can exist, but ctlSetRuntimePath is reserved for Intel
        // components, not an application-level discovery fallback.
        const auto loaded = load_intel_library();
        module = loaded.module;
        if (!module) {
            vendor_error(devices, intel_loader_failure(loaded.error));
            return;
        }
        ctl_init_args_t args{};
        args.Size = sizeof(args);
        args.Version = 0;
        args.AppVersion = CTL_IMPL_VERSION;
        // Telemetry needs the driver's Level Zero runtime. Never request the
        // IGSC firmware-update flag, an overclock waiver or any setter API.
        args.flags = CTL_INIT_FLAG_USE_LEVEL_ZERO;
        const auto initialized =
            invoke<decltype(&ctlInit)>("ctlInit", &args, &api);
        if (initialized != CTL_RESULT_SUCCESS || !api) {
            vendor_error(devices, intel_failure(
                initialized,
                "ctlInit (installed Intel driver/Level Zero runtime)", false));
            return;
        }
        std::uint32_t count = 0;
        auto result = invoke<decltype(&ctlEnumerateDevices)>(
            "ctlEnumerateDevices", api, &count, nullptr);
        if (result != CTL_RESULT_SUCCESS) {
            vendor_error(devices, intel_failure(
                result, "ctlEnumerateDevices(count)", false));
            return;
        }
        if (count > 256) {
            vendor_error(devices, missing(
                Status::error, "IGCL", "Too many Intel adapters returned."));
            return;
        }
        std::vector<ctl_device_adapter_handle_t> handles(count);
        if (count) {
            result = invoke<decltype(&ctlEnumerateDevices)>(
                "ctlEnumerateDevices", api, &count, handles.data());
            if (result != CTL_RESULT_SUCCESS || count > handles.size()) {
                vendor_error(devices, intel_failure(
                    result, "ctlEnumerateDevices(handles)", false));
                return;
            }
        }
        for (std::uint32_t i = 0; i < count; ++i) {
            LUID luid{};
            ctl_device_adapter_properties_t properties{};
            properties.Size = sizeof(properties);
            properties.Version = 0;
            properties.pDeviceID = &luid;
            properties.device_id_size = sizeof(luid);
            result = invoke<decltype(&ctlGetDeviceProperties)>(
                "ctlGetDeviceProperties", handles[i], &properties);
            if (result != CTL_RESULT_SUCCESS) {
                errors.push_back(intel_failure(
                    result, "ctlGetDeviceProperties(LUID)", false).message);
                continue;
            }
            if (properties.device_type != CTL_DEVICE_TYPE_GRAPHICS ||
                properties.pci_vendor_id != 0x8086) {
                continue;
            }
            if (properties.device_id_size != sizeof(luid)) {
                errors.emplace_back("IGCL returned an invalid LUID size.");
                continue;
            }
            const AdapterId id = {static_cast<std::uint32_t>(luid.HighPart),
                                  luid.LowPart};
            auto* device = match_device(
                devices, id, 0x8086, properties.pci_device_id);
            if (!device) {
                errors.push_back(
                    "IGCL adapter " + id.text() +
                    " has no unique DXGI LUID/PCI identity match.");
                continue;
            }
            if (adapters.count(id) || ambiguous.count(id)) {
                ambiguous.insert(id);
                adapters.erase(id);
                sensor_error(*device, missing(
                    Status::error, "IGCL LUID association",
                    "Multiple Intel devices returned the same LUID; "
                    "no readings will be assigned."));
                continue;
            }
            if (properties.driver_version) {
                device->driver_version =
                    driver_version(properties.driver_version);
            }
            IntelAdapter adapter(handles[i], maximum_gap);
            adapters.emplace(id, std::move(adapter));
        }
        for (auto& device : devices) {
            if (device.vendor_id == 0x8086 && !adapters.count(device.luid) &&
                !ambiguous.count(device.luid)) {
                sensor_error(device, missing(
                    Status::error, "IGCL LUID association",
                    "Intel adapter could not be matched by LUID and PCI "
                    "vendor/device ID. Vendor-only and enumeration-order "
                    "fallbacks are deliberately prohibited."));
            }
        }
    }

    ~Impl() {
        if (api) {
            const auto result = invoke<decltype(&ctlClose)>("ctlClose", api);
            if (result != CTL_RESULT_SUCCESS) {
                std::cerr << "gpu_probe: "
                          << intel_failure(result, "ctlClose", false).message
                          << '\n';
            }
        }
        if (module) {
            FreeLibrary(module);
        }
    }

    Metric temperature_sensors(ctl_device_adapter_handle_t handle) const {
        std::uint32_t count = 0;
        auto result = invoke<decltype(&ctlEnumTemperatureSensors)>(
            "ctlEnumTemperatureSensors", handle, &count, nullptr);
        if (result != CTL_RESULT_SUCCESS) {
            return intel_failure(result, "ctlEnumTemperatureSensors(count)");
        }
        if (count > 256) {
            return missing(Status::error, "IGCL temperature",
                           "Invalid temperature-sensor count.");
        }
        if (!count) {
            return missing(Status::unsupported, "IGCL temperature",
                           "The driver enumerated no temperature sensors.");
        }
        std::vector<ctl_temp_handle_t> sensors(count);
        result = invoke<decltype(&ctlEnumTemperatureSensors)>(
            "ctlEnumTemperatureSensors", handle, &count, sensors.data());
        if (result != CTL_RESULT_SUCCESS || count > sensors.size()) {
            return intel_failure(result, "ctlEnumTemperatureSensors(handles)");
        }
        std::vector<ctl_temp_handle_t> gpu;
        std::vector<ctl_temp_handle_t> global;
        for (std::uint32_t i = 0; i < count; ++i) {
            ctl_temp_properties_t properties{};
            properties.Size = sizeof(properties);
            result = invoke<decltype(&ctlTemperatureGetProperties)>(
                "ctlTemperatureGetProperties", sensors[i], &properties);
            if (result != CTL_RESULT_SUCCESS) {
                return intel_failure(result, "ctlTemperatureGetProperties");
            }
            if (properties.type == CTL_TEMP_SENSORS_GPU) {
                gpu.push_back(sensors[i]);
            } else if (properties.type == CTL_TEMP_SENSORS_GLOBAL) {
                global.push_back(sensors[i]);
            }
        }
        const auto& selected = gpu.empty() ? global : gpu;
        if (selected.empty()) {
            return missing(Status::unsupported, "IGCL temperature",
                           "No GPU or global maximum temperature sensor "
                           "was reported; memory/minimum sensors are not "
                           "substituted for GPU temperature.");
        }
        std::optional<double> maximum;
        const std::string source =
            gpu.empty() ? "IGCL ctlTemperatureGetState (global maximum)"
                        : "IGCL ctlTemperatureGetState (GPU maximum)";
        for (auto sensor : selected) {
            double value = 0.0;
            result = invoke<decltype(&ctlTemperatureGetState)>(
                "ctlTemperatureGetState", sensor, &value);
            if (result != CTL_RESULT_SUCCESS) {
                return intel_failure(result, "ctlTemperatureGetState");
            }
            const auto validated = reading(value, source, false);
            if (validated.status != Status::ok) {
                return validated;
            }
            maximum = maximum ? std::max(*maximum, value) : value;
        }
        return reading(*maximum, source, false);
    }

    Metric domain_power(IntelAdapter& adapter) const {
        std::uint32_t count = 0;
        auto result = invoke<decltype(&ctlEnumPowerDomains)>(
            "ctlEnumPowerDomains", adapter.handle, &count, nullptr);
        if (result != CTL_RESULT_SUCCESS) {
            adapter.domain_energy.reset();
            return intel_failure(result, "ctlEnumPowerDomains(count)");
        }
        if (!count) {
            adapter.domain_energy.reset();
            return missing(Status::unsupported, "IGCL energy counter",
                           "The driver enumerated no power domains.");
        }
        if (count != 1) {
            adapter.domain_energy.reset();
            return missing(
                Status::error, "IGCL energy counter",
                "Multiple power domains have no identifying scope metadata; "
                "refusing to sum overlapping domains or choose by index.");
        }
        ctl_pwr_handle_t domain = nullptr;
        result = invoke<decltype(&ctlEnumPowerDomains)>(
            "ctlEnumPowerDomains", adapter.handle, &count, &domain);
        if (result != CTL_RESULT_SUCCESS || count != 1 || !domain) {
            adapter.domain_energy.reset();
            return intel_failure(result, "ctlEnumPowerDomains(handle)");
        }
        if (adapter.last_power_domain != domain) {
            adapter.domain_energy.reset();
            adapter.last_power_domain = domain;
        }
        ctl_power_energy_counter_t counter{};
        counter.Size = sizeof(counter);
        const auto read_epoch = sampling_epoch.observe();
        result = invoke<decltype(&ctlPowerGetEnergyCounter)>(
            "ctlPowerGetEnergyCounter", domain, &counter);
        const auto completed_epoch = sampling_epoch.observe();
        if (result != CTL_RESULT_SUCCESS) {
            adapter.domain_energy.reset();
            return intel_failure(result, "ctlPowerGetEnergyCounter");
        }
        return adapter.domain_energy.microjoules(
            counter.energy, counter.timestamp,
            "IGCL ctlPowerGetEnergyCounter (uJ/us delta)",
            completed_epoch, EnergyDelta::Clock::now(), read_epoch);
    }

    void sample(Device& device, IntelAdapter& adapter) const {
        ctl_power_telemetry_t telemetry{};
        telemetry.Size = sizeof(telemetry);
        telemetry.Version = 0;
        const auto read_epoch = sampling_epoch.observe();
        const auto result = invoke<decltype(&ctlPowerTelemetryGet)>(
            "ctlPowerTelemetryGet", adapter.handle, &telemetry);
        const auto completed_epoch = sampling_epoch.observe();
        Metric temp;
        Metric watts;
        if (result == CTL_RESULT_SUCCESS) {
            temp = intel_value(
                telemetry.gpuCurrentTemperature,
                CTL_UNITS_TEMPERATURE_CELSIUS,
                "IGCL ctlPowerTelemetryGet GPU temperature", false);
            auto energy = intel_value(
                telemetry.gpuEnergyCounter, CTL_UNITS_ENERGY_JOULES,
                "IGCL ctlPowerTelemetryGet GPU energy");
            auto timestamp = intel_value(
                telemetry.timeStamp, CTL_UNITS_TIME_SECONDS,
                "IGCL ctlPowerTelemetryGet timestamp");
            if (energy.status == Status::ok && timestamp.status == Status::ok) {
                watts = adapter.telemetry_energy.joules(
                    *energy.value, *timestamp.value,
                    "IGCL ctlPowerTelemetryGet (GPU J/s delta)",
                    completed_epoch, EnergyDelta::Clock::now(), read_epoch);
            } else if (energy.status == Status::error) {
                watts = energy;
            } else if (timestamp.status == Status::error) {
                watts = timestamp;
            } else {
                watts = energy.status != Status::ok ? energy : timestamp;
            }
        } else {
            temp = intel_failure(result, "ctlPowerTelemetryGet");
            watts = temp;
        }
        if (temp.status != Status::ok) {
            temp = fallback(std::move(temp),
                            temperature_sensors(adapter.handle));
        }
        if (watts.status != Status::ok &&
            watts.status != Status::unavailable) {
            adapter.telemetry_energy.reset();
            watts = fallback(std::move(watts), domain_power(adapter));
        } else {
            adapter.domain_energy.reset();
        }
        device.metrics[temperature] = std::move(temp);
        device.metrics[power] = std::move(watts);
    }
};

IntelSensors::IntelSensors(std::vector<Device>& devices,
                           std::vector<std::string>& errors,
                           SamplingEpoch& epoch,
                           std::chrono::milliseconds interval)
    : impl_(std::make_unique<Impl>(devices, errors, epoch, interval)) {}
IntelSensors::~IntelSensors() = default;

void IntelSensors::sample(std::vector<Device>& devices) {
    for (auto& device : devices) {
        const auto it = impl_->adapters.find(device.luid);
        if (device.vendor_id == 0x8086 && it != impl_->adapters.end()) {
            impl_->sample(device, it->second);
        }
    }
}

}  // namespace probe
