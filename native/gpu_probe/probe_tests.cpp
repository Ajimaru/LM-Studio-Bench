// Deterministic SDK/protocol tests: no installed GPU or driver is required.
#include "intel_loader.h"
#include "probe_types.h"
#include "vendor_status.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>

namespace {

using namespace probe;

int loader_calls = 0;
std::wstring loader_name;
DWORD loader_flags = 0;
HANDLE loader_file = INVALID_HANDLE_VALUE;

HMODULE WINAPI missing_igcl(LPCWSTR name, HANDLE file, DWORD flags) {
    ++loader_calls;
    loader_name = name;
    loader_file = file;
    loader_flags = flags;
    SetLastError(ERROR_MOD_NOT_FOUND);
    return nullptr;
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_near(double actual, double expected, const char* message) {
    require(std::abs(actual - expected) < 1e-10, message);
}

Snapshot fixture() {
    Snapshot snapshot;
    snapshot.timestamp_unix_ms = 1700000000123;
    Device device;
    device.luid = {0xffffffff, 1};
    device.index = 2;
    device.vendor_id = 0x1002;
    device.device_id = 0x744c;
    device.vendor = "AMD";
    device.name = u8"GPU \"ä\"\\test\n";
    device.vram_total_gb = bytes_to_gib(32ULL * 1024 * 1024 * 1024);
    device.shared_total_gb = bytes_to_gib(24ULL * 1024 * 1024 * 1024);
    device.metrics[temperature] = reading(0.0, "test");
    device.metrics[power] =
        missing(Status::unsupported, "test", "Sensor explicitly unsupported.");
    device.metrics[dedicated] =
        reading(std::numeric_limits<double>::quiet_NaN(), "test");
    device.metrics[shared] =
        missing(Status::unavailable, "test", "Counter baseline not ready.");
    snapshot.devices.push_back(device);
    device.luid.low = 2;
    device.index = 7;
    device.vendor_id = 0x8086;
    device.vendor = "Intel";
    device.name = "Second adapter";
    device.driver_version = "31.0.101.1234";
    device.integrated = true;
    device.pci = pci_address(0, 2, 0);
    device.vram_total_gb = 0.125;
    device.shared_total_gb = 16.0;
    device.metrics[temperature] = reading(-1.5, "test", false);
    device.metrics[power] = reading(12.5, "test");
    device.metrics[dedicated] = reading(0.0, "test");
    device.metrics[shared] = reading(1.25, "test");
    snapshot.devices.push_back(device);
    append_metric_errors(snapshot);
    return snapshot;
}

void conversions_and_identity() {
    expect_near(bytes_to_gib(0), 0.0, "Zero bytes must be valid.");
    expect_near(bytes_to_gib(16ULL << 30), 16.0,
                "VRAM must not truncate at 4 GiB.");
    expect_near(bytes_to_gib(1536ULL << 20), 1.5,
                "Use binary GiB, not decimal GB.");
    require(AdapterId{0xffffffff, 0x123}.text() ==
                "luid:ffffffff:00000123", "LUID must preserve high bits.");
    require(driver_version(0x001f0000006504d2) == "31.0.101.1234",
            "Driver version word conversion failed.");
    require(!physical_adapter(0x1414, false), "Exclude Microsoft renderer.");
    require(!physical_adapter(0x8086, true), "Exclude software adapters.");
    require(physical_adapter(0x1002, false), "Keep AMD hardware.");
    require(consistent_uma({true, true}) == true,
            "Classify verified UMA adapters.");
    require(consistent_uma({false, false}) == false,
            "Preserve verified non-UMA false, not null.");
    require(!consistent_uma({}).has_value(),
            "Missing architecture evidence must remain unknown.");
    require(!consistent_uma({true, false}).has_value(),
            "Never guess for a mixed linked adapter.");
    const auto pci = pci_address(0, 2, 0);
    require(pci.has_value() && pci->bus == 0 &&
                pci->device == 2 && pci->function == 0,
            "PCI zero bus/function are known values, not missing metadata.");
    require(pci_address(255, 31, 7).has_value(),
            "Accept the full PCI bus/device/function ranges.");
    require(!pci_address(256, 0, 0) && !pci_address(0, 32, 0) &&
                !pci_address(0, 0, 8),
            "Reject invalid PCI fields rather than truncate or invent them.");
    require(!pci_address(std::numeric_limits<std::uint32_t>::max(), 0, 0),
            "An unwritten/unknown adapter address must remain null.");
    const auto parsed =
        parse_memory_instance("luid_0xFFFFFFFF_0x00000123_phys_12");
    require(parsed.has_value() && parsed->luid == AdapterId{0xffffffff, 0x123}
                && parsed->physical_index == 12, "PDH LUID parsing failed.");
    require(!parse_memory_instance(
                "pid_123_luid_0x00000000_0x00000123_phys_0"),
            "Never match process-local counters.");
    require(!parse_memory_instance("luid_0x100000000_0x1_phys_0"),
            "Reject LUID overflow.");
    require(!parse_memory_instance("luid_0x1_0x2_phys_0#1"),
            "Reject ambiguous PDH instance suffixes.");
    require(!parse_memory_instance("_Total"), "Never assign aggregate totals.");
    auto snapshot = fixture();
    require(match_device(snapshot.devices, {0xffffffff, 2}, 0x8086) ==
                &snapshot.devices[1], "Match LUID, not enumeration index.");
    require(!match_device(snapshot.devices, {0xffffffff, 2}, 0x1002),
            "Reject vendor mismatch.");
    require(!match_device(snapshot.devices, {0xffffffff, 2}, 0x8086, 0x9999),
            "Reject PCI device mismatch.");
    snapshot.devices.push_back(snapshot.devices[1]);
    require(!match_device(snapshot.devices, {0xffffffff, 2}, 0x8086),
            "Reject duplicate adapter identities.");
}

void statuses_and_values() {
    require(reading(0.0, "test").status == Status::ok,
            "A valid zero must not be missing.");
    require(reading(-1.0, "test").status == Status::error,
            "Reject negative memory/power.");
    require(reading(std::numeric_limits<double>::infinity(), "test").status ==
                Status::error, "Reject infinity.");
    require(adlx_failure(ADLX_NOT_SUPPORTED, "sensor").status ==
                Status::unsupported, "ADLX explicit sensor support status.");
    require(adlx_failure(ADLX_NOT_SUPPORTED, "init", false).status ==
                Status::error, "ADLX init failure must not be unsupported.");
    require(adlx_failure(ADLX_FAIL, "read").status == Status::error,
            "ADLX driver failures must remain errors.");
    require(intel_failure(CTL_RESULT_ERROR_UNSUPPORTED_FEATURE, "sensor")
                .status == Status::unsupported, "Intel support status.");
    for (auto code : {CTL_RESULT_ERROR_NOT_IMPLEMENTED,
                      CTL_RESULT_ERROR_DEVICE_LOST,
                      CTL_RESULT_ERROR_LOAD,
                      CTL_RESULT_ERROR_UNSUPPORTED_VERSION,
                      CTL_RESULT_ERROR_DATA_READ}) {
        require(intel_failure(code, "sensor").status == Status::error,
                "Intel failures must not masquerade as unsupported.");
    }
    require(intel_failure(CTL_RESULT_ERROR_UNSUPPORTED_FEATURE, "init", false)
                .status == Status::error, "Intel init failure is an error.");
    auto error = missing(Status::error, "primary", "Missing implementation.");
    auto absent = missing(Status::unsupported, "fallback", "No sensor.");
    require(fallback(error, absent).status == Status::error,
            "Unsupported fallback must not hide an implementation failure.");
    const auto recovered = fallback(error, reading(0.0, "fallback"));
    require(recovered.status == Status::ok && !recovered.message.empty(),
            "Recovered measurements must retain primary failure diagnostics.");

    ctl_oc_telemetry_item_t item{};
    item.bSupported = false;
    require(intel_value(item, CTL_UNITS_ENERGY_JOULES, "test").status ==
                Status::unsupported, "Honor Intel bSupported.");
    item.bSupported = true;
    item.units = CTL_UNITS_ENERGY_JOULES;
    item.type = CTL_DATA_TYPE_DOUBLE;
    item.value.datadouble = 12.5;
    expect_near(*intel_value(item, CTL_UNITS_ENERGY_JOULES, "test").value,
                12.5, "Intel double decoding.");
    require(intel_value(item, CTL_UNITS_TIME_SECONDS, "test").status ==
                Status::error, "Never reinterpret units.");
    item.type = CTL_DATA_TYPE_UINT64;
    item.value.datau64 = 9000000000ULL;
    expect_near(*intel_value(item, CTL_UNITS_ENERGY_JOULES, "test").value,
                9000000000.0, "Intel 64-bit data decoding.");
    item.type = CTL_DATA_TYPE_STRING_ASCII;
    require(intel_value(item, CTL_UNITS_ENERGY_JOULES, "test").status ==
                Status::error, "Reject unknown/non-numeric data types.");
}

void energy_deltas() {
    EnergyDelta delta;
    require(delta.microjoules(100, 1000, "test").status == Status::unavailable,
            "Initial sample needs a baseline.");
    expect_near(*delta.microjoules(5100, 1200, "test").value, 25.0,
                "Microjoules/microseconds convert to watts without a scale.");
    expect_near(*delta.microjoules(5100, 1300, "test").value, 0.0,
                "Zero delta energy is a valid idle reading.");
    require(delta.microjoules(5200, 1300, "test").status == Status::unavailable,
            "A duplicate timestamp must not divide by zero.");
    require(delta.microjoules(1, 1400, "test").status == Status::unavailable,
            "Counter reset must not underflow.");
    expect_near(*delta.microjoules(101, 1500, "test").value, 1.0,
                "Recover after a counter reset.");
    require(delta.microjoules(102, 1, "test").status == Status::unavailable,
            "A timestamp reset must rebaseline.");
    delta.reset();
    const auto big = std::numeric_limits<std::uint64_t>::max() - 1000;
    delta.microjoules(big, big, "test");
    expect_near(*delta.microjoules(big + 100, big + 4, "test").value, 25.0,
                "Subtract uint64 before float conversion; preserve deltas.");
    delta.reset();
    require(delta.joules(5.5, 100.25, "test").status == Status::unavailable,
            "Telemetry joules also need a baseline.");
    expect_near(*delta.joules(11.75, 100.75, "test").value, 12.5,
                "Telemetry seconds must not be treated as microseconds.");
    require(delta.joules(1.0, 101.0, "test").status == Status::unavailable,
            "Floating-point counter reset must rebaseline.");
    require(delta.joules(std::numeric_limits<double>::quiet_NaN(), 1.0,
                         "test").status == Status::error,
            "Reject non-finite counters.");
}

void expect_energy_gap(const Metric& metric) {
    require(metric.status == Status::unavailable && !metric.value &&
                metric.message.find("gap") != std::string::npos,
            "A gap must discard the average and explain the new baseline.");
}

void intel_loader_policy() {
    loader_calls = 0;
    const auto result = load_intel_library(missing_igcl);
    require(loader_calls == 1 && loader_name == L"ControlLib.dll" &&
                loader_file == nullptr &&
                loader_flags == LOAD_LIBRARY_SEARCH_SYSTEM32,
            "IGCL must use Intel's documented trusted System32 entry point.");
    require(!result.module && result.error == ERROR_MOD_NOT_FOUND,
            "Preserve the loader's actual Win32 error, including 126.");
    const auto diagnostic = intel_loader_failure(result.error);
    require(diagnostic.status == Status::error && !diagnostic.value &&
                diagnostic.message.find("126") != std::string::npos &&
                diagnostic.message.find("does not prove") !=
                    std::string::npos &&
                diagnostic.message.find("DriverStore") != std::string::npos,
            "A System32 failure must not prove runtime or sensor absence.");
    const auto invalid = load_intel_library(nullptr);
    require(!invalid.module && invalid.error == ERROR_INVALID_PARAMETER &&
                loader_calls == 1,
            "An invalid loader must not trigger a different DLL search.");
}

void expect_power(const Metric& metric, double expected, const char* message) {
    require(metric.status == Status::ok && metric.value.has_value(), message);
    expect_near(*metric.value, expected, message);
}

void energy_sampling_gaps() {
    using namespace std::chrono_literals;
    const auto origin = EnergyDelta::Clock::time_point{};
    SamplingEpoch epoch;
    const auto before = epoch.observe(origin);
    EnergyDelta integer(2s);
    EnergyDelta floating(2s);
    integer.microjoules(0, 1'000'000, "test", before, origin);
    floating.joules(0, 100, "test", before, origin);
    epoch.observe(origin + 1s);
    expect_power(integer.microjoules(
                    25'000'000, 2'000'000, "test", before, origin + 1s),
                25.0, "Integer power before suspension.");
    expect_power(floating.joules(
                    25, 101, "test", before, origin + 1s),
                25.0, "Floating power before suspension.");
    const auto resumed = epoch.observe(origin + 20s);
    require(resumed != before, "A missing heartbeat must change the epoch.");
    expect_energy_gap(integer.microjoules(
        1'025'000'000, 21'000'000, "test", resumed, origin + 20s));
    expect_energy_gap(floating.joules(
        1025, 120, "test", resumed, origin + 20s));
    expect_power(integer.microjoules(
                    1'030'000'000, 22'000'000, "test", resumed,
                    origin + 21s),
                5.0, "Integer power must exclude all paused-interval energy.");
    expect_power(floating.joules(
                    1030, 121, "test", resumed, origin + 21s),
                5.0, "Floating power must use only the post-resume pair.");

    EnergyDelta counter_gap(2s);
    counter_gap.microjoules(0, 1'000'000, "test", 0, origin);
    expect_energy_gap(counter_gap.microjoules(
        100'000'000, 21'000'000, "test", 0, origin + 1s));
    expect_power(counter_gap.microjoules(
                    105'000'000, 22'000'000, "test", 0, origin + 2s),
                5.0, "SDK timestamp gaps independently reset integer pairs.");
    EnergyDelta seconds_gap(2s);
    seconds_gap.joules(0, 100, "test", 0, origin);
    expect_energy_gap(seconds_gap.joules(
        100, 120, "test", 0, origin + 1s));
    expect_power(seconds_gap.joules(
                    105, 121, "test", 0, origin + 2s),
                5.0, "SDK timestamp gaps independently reset floating pairs.");

    EnergyDelta host_gap(2s);
    host_gap.joules(0, 100, "test", 0, origin);
    expect_energy_gap(host_gap.joules(
        100, 100.5, "test", 0, origin + 20s));
    expect_power(host_gap.joules(
                    105, 101, "test", 0, origin + 20500ms),
                10.0, "Host sampling gaps must reset even with slow SDK time.");

    // A long configured interval is valid while the watchdog keeps ticking.
    // A suspension inside that interval must still invalidate the next pair.
    SamplingEpoch idle;
    EnergyDelta slow(61s);
    auto current = idle.observe(origin);
    slow.joules(0, 100, "test", current, origin);
    for (int tick = 1; tick <= 600; ++tick) {
        current = idle.observe(origin + tick * 100ms);
    }
    require(current == 0, "Regular idle heartbeats must not signal a gap.");
    expect_power(slow.joules(
                    600, 160, "test", current, origin + 60s),
                10.0, "A legitimate 60-second polling interval must work.");
    current = idle.observe(origin + 75s);
    require(current == 1, "Detect suspension while the watchdog is idle.");
    require(idle.observe(origin + 70s) == current &&
                idle.observe(origin + 75100ms) == current,
            "Out-of-order observers must not regress the heartbeat.");
    expect_energy_gap(slow.joules(
        1200, 220, "test", current, origin + 120s));
    expect_power(slow.joules(
                    1800, 280, "test", current, origin + 180s),
                10.0, "A new slow pair must exclude the suspended window.");

    // A read completed across suspension may contain a pre-pause snapshot.
    // Discard it entirely rather than accepting it as a new baseline.
    EnergyDelta interrupted(61s);
    EnergyDelta interrupted_integer(61s);
    interrupted.joules(0, 100, "test", 0, origin, 0);
    interrupted_integer.microjoules(
        0, 100'000'000, "test", 0, origin, 0);
    expect_energy_gap(interrupted.joules(
        600, 160, "test", 1, origin + 62s, 0));
    expect_energy_gap(interrupted_integer.microjoules(
        600'000'000, 160'000'000, "test", 1, origin + 62s, 0));
    const auto new_baseline = interrupted.joules(
        1200, 220, "test", 1, origin + 120s, 1);
    const auto new_integer = interrupted_integer.microjoules(
        1'200'000'000, 220'000'000, "test", 1, origin + 120s, 1);
    require(new_baseline.status == Status::unavailable && !new_baseline.value &&
                new_integer.status == Status::unavailable && !new_integer.value,
            "Only a wholly post-resume read can establish the baseline.");
    expect_power(interrupted.joules(
                     1800, 280, "test", 1, origin + 180s, 1),
                 10.0, "Discard the interrupted floating counter sample.");
    expect_power(interrupted_integer.microjoules(
                     1'800'000'000, 280'000'000, "test", 1, origin + 180s, 1),
                 10.0, "Discard the interrupted integer counter sample.");
}

class CommaDecimal : public std::numpunct<char> {
protected:
    char do_decimal_point() const override { return ','; }
};

void serialization() {
    const auto original = std::locale();
    std::locale::global(std::locale(original, new CommaDecimal));
    const auto json = to_json(fixture());
    std::locale::global(original);
    require(json.find("\"gtt_gb\":1.25") != std::string::npos,
            "JSON decimals must be locale independent.");
    require(json.find("\\\"") != std::string::npos &&
                json.find("\\n") != std::string::npos,
            "Escape JSON control characters and quotes.");
    require(json.find("\"vram_gb\":null") != std::string::npos &&
                json.find("\"vram_gb\":\"error\"") != std::string::npos,
            "Invalid readings must be null with error status.");
    require(json.find("\"integrated\":null") != std::string::npos &&
                json.find("\"integrated\":true") != std::string::npos,
            "Preserve unknown and verified UMA classification.");
    require(json.find("\"timestamp_unix_ms\":1700000000123") !=
                std::string::npos,
            "Preserve integer epoch milliseconds without 32-bit truncation.");
    require(json.find("\"pci_bus\":null") != std::string::npos &&
                json.find("\"pci_bus\":0,\"pci_device\":2,\"pci_function\":0")
                    != std::string::npos,
            "PCI addresses distinguish unknown from valid zero fields.");
    require(json.find("\"pci_domain\"") == std::string::npos,
            "Never manufacture a PCI domain.");
    require(json_string(std::string(1, '\x01')) == "\"\\u0001\"",
            "Escape arbitrary JSON control characters.");
    auto invalid = fixture();
    invalid.devices[0].integrated = false;
    require(to_json(invalid).find("\"integrated\":false") != std::string::npos,
            "False is a known architecture value, not missing metadata.");
    invalid.devices[0].metrics[temperature] = {
        std::numeric_limits<double>::infinity(), Status::ok, "", "test"};
    require(to_json(invalid).find("\"temp_celsius\":\"error\"") !=
                std::string::npos,
            "Serializer must reject even an invalid caller-supplied metric.");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--fixture") {
            std::cout << probe::to_json(fixture()) << '\n';
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--energy-gap") {
            energy_sampling_gaps();
            std::cout << "Synthetic energy-gap checks passed.\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--intel-loader") {
            intel_loader_policy();
            std::cout << "Trusted Intel loader checks passed.\n";
            return 0;
        }
        conversions_and_identity();
        statuses_and_values();
        energy_deltas();
        energy_sampling_gaps();
        intel_loader_policy();
        serialization();
        std::cout << "All deterministic native GPU probe tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Native GPU probe test failed: " << error.what() << '\n';
        return 1;
    }
}
