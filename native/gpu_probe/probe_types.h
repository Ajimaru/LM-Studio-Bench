// Protocol values and hardware-independent telemetry validation.
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace probe {

enum class Status { ok, unsupported, unavailable, error };
enum MetricIndex : std::size_t { temperature, power, dedicated, shared };
inline constexpr std::array<const char*, 4> metric_names = {
    "temp_celsius", "power_watts", "vram_gb", "gtt_gb"};

struct Metric {
    std::optional<double> value;
    Status status = Status::unavailable;
    std::string message = "No sample has been collected.";
    std::string source;
};

struct AdapterId {
    std::uint32_t high = 0;
    std::uint32_t low = 0;

    bool operator==(const AdapterId& other) const noexcept;
    bool operator<(const AdapterId& other) const noexcept;
    std::string text() const;
};

struct PciAddress {
    std::uint32_t bus;
    std::uint32_t device;
    std::uint32_t function;
};

struct Device {
    AdapterId luid;
    unsigned int index = 0;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::string vendor;
    std::string name;
    std::optional<double> vram_total_gb;
    std::optional<double> shared_total_gb;
    std::optional<std::string> driver_version;
    // D3D12 UMA architecture, not a name, vendor or capacity heuristic.
    std::optional<bool> integrated;
    // D3DKMT exposes bus/device/function, but not a PCI domain.
    std::optional<PciAddress> pci;
    std::array<Metric, 4> metrics;
};

struct Snapshot {
    std::uint64_t sequence = 0;
    // Acquisition start, not output/read time: conservative across suspension.
    std::int64_t timestamp_unix_ms = 0;
    std::vector<Device> devices;
    std::vector<std::string> errors;
};

struct MemoryInstance {
    AdapterId luid;
    unsigned int physical_index = 0;
};

// Every non-ok value is null; non-finite readings become explicit errors.
Metric missing(Status status, std::string source, std::string message);
Metric reading(double value, std::string source, bool nonnegative = true);
Metric fallback(Metric primary, Metric alternative);
double bytes_to_gib(std::uint64_t bytes) noexcept;
const char* status_name(Status status) noexcept;
std::string hex_code(std::uint32_t code);
std::string driver_version(std::uint64_t version);
std::string vendor_name(std::uint32_t vendor_id);
bool physical_adapter(std::uint32_t vendor_id, bool software) noexcept;
// Linked adapters must agree across every physical node; otherwise unknown.
std::optional<bool> consistent_uma(const std::vector<bool>& nodes);
std::optional<PciAddress> pci_address(std::uint32_t bus,
                                     std::uint32_t device,
                                     std::uint32_t function) noexcept;
std::int64_t unix_timestamp_ms() noexcept;

// Parse only adapter-wide counters, never pid_* process-local instances.
std::optional<MemoryInstance> parse_memory_instance(std::string_view name);

// Refuse absent, duplicate, vendor-mismatched or device-mismatched identities.
Device* match_device(std::vector<Device>& devices, AdapterId luid,
                     std::uint32_t vendor_id,
                     std::optional<std::uint32_t> device_id = std::nullopt);

// JSON serialization is locale-independent and never emits NaN/Infinity.
std::string json_string(std::string_view value);
std::string to_json(const Snapshot& snapshot);
void append_metric_errors(Snapshot& snapshot);

// Shared by the watchdog heartbeat and sensor reads, including idle periods.
class SamplingEpoch {
public:
    using Clock = std::chrono::steady_clock;
    std::uint64_t observe(Clock::time_point now = Clock::now());

private:
    std::mutex mutex_;
    std::optional<Clock::time_point> previous_;
    std::uint64_t epoch_ = 0;
};

class EnergyDelta {
public:
    using Clock = SamplingEpoch::Clock;
    explicit EnergyDelta(
        std::chrono::milliseconds maximum_gap = std::chrono::seconds(2));
    // IGCL energy counters use microjoules and microseconds: their ratio is W.
    Metric microjoules(std::uint64_t energy, std::uint64_t timestamp,
                      const std::string& source, std::uint64_t epoch = 0,
                      Clock::time_point observed_at = Clock::now(),
                      std::optional<std::uint64_t> read_epoch = std::nullopt);
    // ctlPowerTelemetryGet declares joules and seconds in its unit fields.
    Metric joules(double energy, double timestamp, const std::string& source,
                  std::uint64_t epoch = 0,
                  Clock::time_point observed_at = Clock::now(),
                  std::optional<std::uint64_t> read_epoch = std::nullopt);
    void reset() noexcept;

private:
    bool begin_sample(std::uint64_t epoch, Clock::time_point observed_at);
    double maximum_gap_seconds_;
    std::optional<std::uint64_t> previous_epoch_;
    std::optional<Clock::time_point> previous_observation_;
    std::optional<std::array<std::uint64_t, 2>> integer_previous_;
    std::optional<std::array<double, 2>> floating_previous_;
};

}  // namespace probe
