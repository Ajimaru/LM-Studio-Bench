#include "probe_types.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <locale>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace probe {
namespace {

void number(std::ostream& out, std::optional<double> value) {
    if (value && std::isfinite(*value)) {
        out << std::setprecision(17) << *value;
    } else {
        out << "null";
    }
}

Metric baseline(const std::string& source, const char* message) {
    return missing(Status::unavailable, source, message);
}

}  // namespace

bool AdapterId::operator==(const AdapterId& other) const noexcept {
    return high == other.high && low == other.low;
}

bool AdapterId::operator<(const AdapterId& other) const noexcept {
    return std::tie(high, low) < std::tie(other.high, other.low);
}

std::string AdapterId::text() const {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "luid:" << std::hex << std::setfill('0') << std::setw(8)
        << high << ':' << std::setw(8) << low;
    return out.str();
}

Metric missing(Status status, std::string source, std::string message) {
    if (status == Status::ok) {
        status = Status::error;
        message = "Internal error: an ok metric requires a finite value.";
    }
    return {std::nullopt, status, std::move(message), std::move(source)};
}

Metric reading(double value, std::string source, bool nonnegative) {
    if (!std::isfinite(value) || (nonnegative && value < 0.0)) {
        return missing(Status::error, std::move(source),
                       "The driver returned an invalid numeric reading.");
    }
    return {value, Status::ok, "", std::move(source)};
}

Metric fallback(Metric primary, Metric alternative) {
    if (primary.status == Status::ok) {
        return primary;
    }
    if (alternative.status == Status::ok ||
        alternative.status == Status::unavailable) {
        alternative.message += " Fallback after " + primary.source + ": " +
                               primary.message;
        return alternative;
    }
    // A failed or missing implementation is not evidence of absent hardware.
    if (primary.status == Status::error) {
        primary.message += " Fallback " + alternative.source + ": " +
                           alternative.message;
        return primary;
    }
    alternative.message += " Primary " + primary.source + ": " +
                           primary.message;
    return alternative;
}

double bytes_to_gib(std::uint64_t bytes) noexcept {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

const char* status_name(Status status) noexcept {
    switch (status) {
    case Status::ok: return "ok";
    case Status::unsupported: return "unsupported";
    case Status::unavailable: return "unavailable";
    case Status::error: return "error";
    }
    return "error";
}

std::string hex_code(std::uint32_t code) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "0x" << std::hex << std::setfill('0') << std::setw(8) << code;
    return out.str();
}

std::string driver_version(std::uint64_t version) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << ((version >> 48) & 0xffff) << '.'
        << ((version >> 32) & 0xffff) << '.'
        << ((version >> 16) & 0xffff) << '.' << (version & 0xffff);
    return out.str();
}

std::string vendor_name(std::uint32_t vendor_id) {
    switch (vendor_id) {
    case 0x1002: return "AMD";
    case 0x8086: return "Intel";
    case 0x10de: return "NVIDIA";
    default: return "Unknown";
    }
}

bool physical_adapter(std::uint32_t vendor_id, bool software) noexcept {
    return !software && vendor_id != 0x1414;
}

std::optional<bool> consistent_uma(const std::vector<bool>& nodes) {
    if (nodes.empty() ||
        !std::all_of(nodes.begin(), nodes.end(),
                     [&nodes](bool value) { return value == nodes.front(); })) {
        return std::nullopt;
    }
    return nodes.front();
}

std::optional<PciAddress> pci_address(std::uint32_t bus,
                                     std::uint32_t device,
                                     std::uint32_t function) noexcept {
    if (bus > 255 || device > 31 || function > 7) {
        return std::nullopt;
    }
    return PciAddress{bus, device, function};
}

std::int64_t unix_timestamp_ms() noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

std::optional<MemoryInstance> parse_memory_instance(std::string_view name) {
    static const std::regex pattern(
        R"(^luid_0x([0-9a-f]{1,8})_0x([0-9a-f]{1,8})_phys_([0-9]+)$)",
        std::regex::icase | std::regex::optimize);
    std::match_results<std::string_view::const_iterator> match;
    if (!std::regex_match(name.begin(), name.end(), match, pattern)) {
        return std::nullopt;
    }
    MemoryInstance parsed;
    std::array<std::uint32_t*, 3> outputs = {
        &parsed.luid.high, &parsed.luid.low, &parsed.physical_index};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const std::string part = match[i + 1].str();
        const auto result = std::from_chars(
            part.data(), part.data() + part.size(), *outputs[i],
            i == 2 ? 10 : 16);
        if (result.ec != std::errc{} ||
            result.ptr != part.data() + part.size()) {
            return std::nullopt;
        }
    }
    return parsed;
}

Device* match_device(std::vector<Device>& devices, AdapterId luid,
                     std::uint32_t vendor_id,
                     std::optional<std::uint32_t> device_id) {
    Device* result = nullptr;
    for (auto& device : devices) {
        if (!(device.luid == luid)) {
            continue;
        }
        if (result || device.vendor_id != vendor_id ||
            (device_id && device.device_id != *device_id)) {
            return nullptr;
        }
        result = &device;
    }
    return result;
}

std::string json_string(std::string_view value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out = "\"";
    for (unsigned char c : value) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                out += "\\u00";
                out += digits[c >> 4];
                out += digits[c & 15];
            } else {
                out += static_cast<char>(c);
            }
        }
    }
    return out + '"';
}

std::string to_json(const Snapshot& snapshot) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\"protocol_version\":1,\"sequence\":" << snapshot.sequence
        << ",\"timestamp_unix_ms\":" << snapshot.timestamp_unix_ms
        << ",\"devices\":[";
    bool comma = false;
    for (const auto& device : snapshot.devices) {
        if (comma) {
            out << ',';
        }
        comma = true;
        out << "{\"id\":" << json_string(device.luid.text())
            << ",\"index\":" << device.index
            << ",\"vendor\":" << json_string(device.vendor)
            << ",\"name\":" << json_string(device.name)
            << ",\"vram_total_gb\":";
        number(out, device.vram_total_gb);
        out << ",\"shared_total_gb\":";
        number(out, device.shared_total_gb);
        out << ",\"driver_version\":";
        out << (device.driver_version ? json_string(*device.driver_version)
                                     : "null");
        out << ",\"integrated\":";
        if (device.integrated.has_value()) {
            out << (*device.integrated ? "true" : "false");
        } else {
            out << "null";
        }
        const std::array<const char*, 3> pci_names = {
            "pci_bus", "pci_device", "pci_function"};
        std::array<std::optional<std::uint32_t>, 3> address;
        if (device.pci) {
            address = {
                device.pci->bus, device.pci->device, device.pci->function};
        }
        for (std::size_t i = 0; i < address.size(); ++i) {
            out << ',' << json_string(pci_names[i]) << ':';
            if (address[i].has_value()) {
                out << *address[i];
            } else {
                out << "null";
            }
        }
        // Defensively normalize malformed values before serializing status.
        auto metrics = device.metrics;
        for (auto& metric : metrics) {
            if (metric.status == Status::ok &&
                (!metric.value || !std::isfinite(*metric.value))) {
                metric = missing(Status::error, metric.source,
                                 "Invalid non-finite or absent reading.");
            }
        }
        for (const auto* section :
             {"metrics", "status", "messages", "sources"}) {
            out << ',' << json_string(section) << ":{";
            for (std::size_t i = 0; i < metrics.size(); ++i) {
                if (i) {
                    out << ',';
                }
                const auto& metric = metrics[i];
                out << json_string(metric_names[i]) << ':';
                if (std::string_view(section) == "metrics") {
                    number(out, metric.status == Status::ok ? metric.value
                                                           : std::nullopt);
                } else if (std::string_view(section) == "status") {
                    out << json_string(status_name(metric.status));
                } else if (std::string_view(section) == "messages") {
                    out << json_string(metric.message);
                } else {
                    out << json_string(metric.source);
                }
            }
            out << '}';
        }
        out << '}';
    }
    out << "],\"errors\":[";
    comma = false;
    for (const auto& error : snapshot.errors) {
        if (comma) {
            out << ',';
        }
        comma = true;
        out << json_string(error);
    }
    out << "]}";
    return out.str();
}

void append_metric_errors(Snapshot& snapshot) {
    for (const auto& device : snapshot.devices) {
        for (std::size_t i = 0; i < device.metrics.size(); ++i) {
            if (device.metrics[i].status == Status::error) {
                snapshot.errors.push_back(device.luid.text() + " " +
                    metric_names[i] + ": " + device.metrics[i].message);
            }
        }
    }
    auto& errors = snapshot.errors;
    std::sort(errors.begin(), errors.end());
    errors.erase(std::unique(errors.begin(), errors.end()), errors.end());
}

std::uint64_t SamplingEpoch::observe(Clock::time_point now) {
    std::lock_guard<std::mutex> lock(mutex_);
    // A caller can capture "now" before another observer acquires this lock.
    if (previous_ && now < *previous_) {
        return epoch_;
    }
    if (previous_ && now - *previous_ > std::chrono::seconds(1)) {
        ++epoch_;
    }
    previous_ = now;
    return epoch_;
}

EnergyDelta::EnergyDelta(std::chrono::milliseconds maximum_gap)
    : maximum_gap_seconds_(
          std::chrono::duration<double>(maximum_gap).count()) {
    if (maximum_gap.count() <= 0) {
        throw std::invalid_argument("Energy sampling gap must be positive.");
    }
}

bool EnergyDelta::begin_sample(std::uint64_t epoch,
                               Clock::time_point observed_at) {
    const bool changed = previous_epoch_ && *previous_epoch_ != epoch;
    const bool gap = previous_observation_ &&
        (observed_at < *previous_observation_ ||
         std::chrono::duration<double>(
             observed_at - *previous_observation_).count() >
             maximum_gap_seconds_);
    if (changed || gap) {
        integer_previous_.reset();
        floating_previous_.reset();
    }
    previous_epoch_ = epoch;
    previous_observation_ = observed_at;
    return changed || gap;
}

Metric EnergyDelta::microjoules(
    std::uint64_t energy, std::uint64_t timestamp, const std::string& source,
    std::uint64_t epoch, Clock::time_point observed_at,
    std::optional<std::uint64_t> read_epoch) {
    if (read_epoch && *read_epoch != epoch) {
        reset();
        return baseline(source, "Sampling/suspension gap interrupted the "
                                "sensor read; waiting for a fresh pair.");
    }
    const bool gap = begin_sample(epoch, observed_at);
    const std::array<std::uint64_t, 2> current = {energy, timestamp};
    if (!integer_previous_) {
        integer_previous_ = current;
        return baseline(
            source, gap ? "Sampling/suspension gap; waiting for a new "
                          "energy-counter pair."
                        : "Waiting for the second energy-counter sample.");
    }
    const auto previous = *integer_previous_;
    if (energy < previous[0] || timestamp < previous[1]) {
        integer_previous_ = current;
        return baseline(
            source, "Energy counter or timestamp reset; new baseline.");
    }
    if (timestamp == previous[1]) {
        return baseline(source, "Energy-counter timestamp has not advanced.");
    }
    integer_previous_ = current;
    if (static_cast<double>(timestamp - previous[1]) >
        maximum_gap_seconds_ * 1'000'000.0) {
        return baseline(
            source, "Energy-counter sampling gap; waiting for a new pair.");
    }
    return reading(static_cast<double>(energy - previous[0]) /
                   static_cast<double>(timestamp - previous[1]), source);
}

Metric EnergyDelta::joules(double energy, double timestamp,
                          const std::string& source, std::uint64_t epoch,
                          Clock::time_point observed_at,
                          std::optional<std::uint64_t> read_epoch) {
    if (!std::isfinite(energy) || !std::isfinite(timestamp) ||
        energy < 0.0 || timestamp < 0.0) {
        reset();
        return missing(Status::error, source,
                       "Invalid energy-counter value or timestamp.");
    }
    if (read_epoch && *read_epoch != epoch) {
        reset();
        return baseline(source, "Sampling/suspension gap interrupted the "
                                "sensor read; waiting for a fresh pair.");
    }
    const bool gap = begin_sample(epoch, observed_at);
    const std::array<double, 2> current = {energy, timestamp};
    if (!floating_previous_) {
        floating_previous_ = current;
        return baseline(
            source, gap ? "Sampling/suspension gap; waiting for a new "
                          "energy-counter pair."
                        : "Waiting for the second energy-counter sample.");
    }
    const auto previous = *floating_previous_;
    if (energy < previous[0] || timestamp < previous[1]) {
        floating_previous_ = current;
        return baseline(
            source, "Energy counter or timestamp reset; new baseline.");
    }
    if (timestamp == previous[1]) {
        return baseline(source, "Energy-counter timestamp has not advanced.");
    }
    floating_previous_ = current;
    if (timestamp - previous[1] > maximum_gap_seconds_) {
        return baseline(
            source, "Energy-counter sampling gap; waiting for a new pair.");
    }
    return reading((energy - previous[0]) / (timestamp - previous[1]), source);
}

void EnergyDelta::reset() noexcept {
    previous_epoch_.reset();
    previous_observation_.reset();
    integer_previous_.reset();
    floating_previous_.reset();
}

}  // namespace probe
