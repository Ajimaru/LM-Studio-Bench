#include "probe_windows.h"

#include <Windows.h>
#include <winternl.h>
#include <d3d12.h>
#include <d3dkmthk.h>
#include <dxgi1_2.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <wrl/client.h>

#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace probe {
namespace {

using Microsoft::WRL::ComPtr;

class KernelAdapter {
public:
    explicit KernelAdapter(D3DKMT_HANDLE handle) : handle_(handle) {}
    ~KernelAdapter() { close(); }
    KernelAdapter(const KernelAdapter&) = delete;
    KernelAdapter& operator=(const KernelAdapter&) = delete;

    NTSTATUS close() noexcept {
        if (!handle_) {
            return 0;
        }
        const D3DKMT_CLOSEADAPTER request{std::exchange(handle_, 0)};
        return D3DKMTCloseAdapter(&request);
    }

private:
    D3DKMT_HANDLE handle_;
};

std::optional<PciAddress> adapter_pci(
    LUID luid, AdapterId identity, std::vector<std::string>& errors) {
    const auto fail = [&](const std::string& message) {
        errors.push_back(identity.text() + " PCI address: " + message);
        return std::optional<PciAddress>{};
    };
    D3DKMT_OPENADAPTERFROMLUID opened{};
    opened.AdapterLuid = luid;
    const auto result = D3DKMTOpenAdapterFromLuid(&opened);
    KernelAdapter owner(opened.hAdapter);
    if (result != 0 || !opened.hAdapter) {
        return fail("D3DKMTOpenAdapterFromLuid failed or returned no handle: " +
                    hex_code(static_cast<std::uint32_t>(result)));
    }
    const auto unknown = std::numeric_limits<UINT>::max();
    D3DKMT_ADAPTERADDRESS address{unknown, unknown, unknown};
    D3DKMT_QUERYADAPTERINFO query{};
    query.hAdapter = opened.hAdapter;
    query.Type = KMTQAITYPE_ADAPTERADDRESS;
    query.pPrivateDriverData = &address;
    query.PrivateDriverDataSize = sizeof(address);
    const auto queried = D3DKMTQueryAdapterInfo(&query);
    const auto closed = owner.close();
    if (closed != 0) {
        errors.push_back(identity.text() + " D3DKMTCloseAdapter failed: " +
                         hex_code(static_cast<std::uint32_t>(closed)));
    }
    if (queried != 0) {
        return fail("KMTQAITYPE_ADAPTERADDRESS failed: " +
                    hex_code(static_cast<std::uint32_t>(queried)));
    }
    if (closed != 0) {
        return fail("Adapter-address query cleanup failed; address withheld.");
    }
    const auto validated = pci_address(
        address.BusNumber, address.DeviceNumber, address.FunctionNumber);
    if (!validated) {
        return fail("The driver returned an invalid or unknown PCI "
                    "bus/device/function. No PCI domain is inferred.");
    }
    return validated;
}

std::optional<bool> adapter_uma(IDXGIAdapter1* adapter, AdapterId identity,
                                std::vector<std::string>& errors) {
    const auto fail = [&](const std::string& message) {
        errors.push_back(identity.text() +
                         " optional UMA metadata: " + message);
        return std::optional<bool>{};
    };
    ComPtr<ID3D12Device> device;
    const auto created = D3D12CreateDevice(
        adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(created)) {
        return fail("D3D12CreateDevice failed: " +
                    hex_code(static_cast<std::uint32_t>(created)));
    }
    const auto luid = device->GetAdapterLuid();
    const AdapterId actual = {static_cast<std::uint32_t>(luid.HighPart),
                              luid.LowPart};
    if (!(identity == actual)) {
        return fail("D3D12 returned a different adapter LUID; ignored.");
    }
    const auto count = device->GetNodeCount();
    if (count == 0 || count > 256) {
        return fail("D3D12 returned an invalid physical-node count.");
    }
    std::vector<bool> nodes;
    for (UINT index = 0; index < count; ++index) {
        D3D12_FEATURE_DATA_ARCHITECTURE1 architecture{};
        architecture.NodeIndex = index;
        auto queried = device->CheckFeatureSupport(
            D3D12_FEATURE_ARCHITECTURE1, &architecture, sizeof(architecture));
        if (queried == E_INVALIDARG) {
            // ARCHITECTURE1 requires Windows 10 1703; retain earlier Windows
            // 10 support using the original, equally authoritative UMA flag.
            D3D12_FEATURE_DATA_ARCHITECTURE original{};
            original.NodeIndex = index;
            queried = device->CheckFeatureSupport(
                D3D12_FEATURE_ARCHITECTURE, &original, sizeof(original));
            architecture.UMA = original.UMA;
        }
        if (FAILED(queried)) {
            return fail("D3D12 architecture query failed: " +
                        hex_code(static_cast<std::uint32_t>(queried)));
        }
        nodes.push_back(architecture.UMA != FALSE);
    }
    const auto result = consistent_uma(nodes);
    if (!result.has_value()) {
        return fail("Linked adapter has mixed UMA/non-UMA physical nodes.");
    }
    return result;
}

std::string pdh_message(const char* operation, PDH_STATUS status) {
    return std::string(operation) + " returned PDH status " +
           hex_code(static_cast<std::uint32_t>(status)) + ".";
}

struct Counter {
    PDH_HCOUNTER handle = nullptr;
    std::string source;
    std::string error;
};

struct MemoryTotal {
    std::uint64_t bytes = 0;
    std::set<unsigned int> physical_indices;
    std::string error;
};

void counter_failure(std::vector<Device>& devices, MetricIndex metric,
                     const Counter& counter, Status status,
                     const std::string& message) {
    for (auto& device : devices) {
        device.metrics[metric] = missing(status, counter.source, message);
    }
}

void read_counter(const Counter& counter, MetricIndex metric,
                  std::vector<Device>& devices,
                  std::vector<std::string>& errors) {
    if (!counter.error.empty()) {
        counter_failure(devices, metric, counter, Status::error, counter.error);
        return;
    }
    DWORD bytes = 0;
    DWORD count = 0;
    auto status = PdhGetFormattedCounterArrayW(
        counter.handle, PDH_FMT_LARGE, &bytes, &count, nullptr);
    std::vector<PDH_FMT_COUNTERVALUE_ITEM_W> storage;
    for (int attempt = 0; status == PDH_MORE_DATA && attempt < 3; ++attempt) {
        if (bytes > 16 * 1024 * 1024 || bytes == 0) {
            break;
        }
        storage.resize((bytes + sizeof(storage[0]) - 1) / sizeof(storage[0]));
        status = PdhGetFormattedCounterArrayW(
            counter.handle, PDH_FMT_LARGE, &bytes, &count, storage.data());
    }
    if (status != ERROR_SUCCESS || storage.empty()) {
        counter_failure(devices, metric, counter,
                        status == PDH_NO_DATA ? Status::unavailable
                                              : Status::error,
                        pdh_message("PdhGetFormattedCounterArrayW", status));
        return;
    }
    if (count > storage.size()) {
        counter_failure(devices, metric, counter, Status::error,
                        "PDH returned an invalid counter-array size.");
        return;
    }

    std::map<AdapterId, MemoryTotal> totals;
    for (DWORD i = 0; i < count; ++i) {
        const auto& item = storage[i];
        if (!item.szName) {
            errors.emplace_back("PDH returned an unnamed memory instance.");
            continue;
        }
        const std::string name = utf8(item.szName);
        if (name == "_Total") {
            continue;
        }
        const auto instance = parse_memory_instance(name);
        if (!instance) {
            errors.push_back("Unrecognized PDH adapter-memory identity: " +
                             name + "; it was not assigned to a GPU.");
            continue;
        }
        auto& total = totals[instance->luid];
        if (!total.physical_indices.insert(instance->physical_index).second) {
            total.error = "Duplicate WDDM physical-adapter identity; "
                          "refusing to double-count memory.";
            continue;
        }
        if (item.FmtValue.CStatus != PDH_CSTATUS_VALID_DATA &&
            item.FmtValue.CStatus != PDH_CSTATUS_NEW_DATA) {
            total.error = pdh_message("Memory counter sample",
                                     item.FmtValue.CStatus);
            continue;
        }
        if (item.FmtValue.largeValue < 0) {
            total.error = "PDH returned a negative byte count.";
            continue;
        }
        const auto value =
            static_cast<std::uint64_t>(item.FmtValue.largeValue);
        if (value > std::numeric_limits<std::uint64_t>::max() - total.bytes) {
            total.error = "WDDM physical-adapter memory sum overflowed.";
            continue;
        }
        total.bytes += value;
    }
    for (auto& device : devices) {
        const auto it = totals.find(device.luid);
        if (it == totals.end()) {
            device.metrics[metric] = missing(
                Status::unavailable, counter.source,
                "No adapter-wide WDDM memory instance matches this LUID.");
        } else if (!it->second.error.empty()) {
            device.metrics[metric] = missing(
                Status::error, counter.source, it->second.error);
        } else {
            device.metrics[metric] = reading(
                bytes_to_gib(it->second.bytes), counter.source);
        }
    }
}

}  // namespace

std::string utf8(std::wstring_view text) {
    if (text.empty()) {
        return {};
    }
    if (text.size() > static_cast<std::size_t>(
                          std::numeric_limits<int>::max())) {
        throw std::runtime_error("Windows string exceeds the UTF-8 limit.");
    }
    const auto length = static_cast<int>(text.size());
    const int count = WideCharToMultiByte(
        CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), length,
        nullptr, 0, nullptr, nullptr);
    if (count == 0) {
        throw std::runtime_error("Invalid UTF-16 returned by Windows.");
    }
    std::string result(static_cast<std::size_t>(count), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(),
                            length, result.data(), count, nullptr,
                            nullptr) != count) {
        throw std::runtime_error("Windows UTF-8 conversion failed.");
    }
    return result;
}

std::vector<Device> enumerate_devices(std::vector<std::string>& errors) {
    std::vector<Device> devices;
    ComPtr<IDXGIFactory1> factory;
    const auto created = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(created)) {
        errors.push_back("CreateDXGIFactory1 failed: " +
                         hex_code(static_cast<std::uint32_t>(created)));
        return devices;
    }
    for (UINT index = 0; index < 256; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const auto enumerated = factory->EnumAdapters1(index, &adapter);
        if (enumerated == DXGI_ERROR_NOT_FOUND) {
            return devices;
        }
        if (FAILED(enumerated)) {
            errors.push_back("DXGI adapter " + std::to_string(index) +
                             " enumeration failed: " +
                             hex_code(static_cast<std::uint32_t>(enumerated)));
            return devices;
        }
        DXGI_ADAPTER_DESC1 desc{};
        const auto described = adapter->GetDesc1(&desc);
        if (FAILED(described)) {
            errors.push_back("DXGI GetDesc1 failed for adapter " +
                             std::to_string(index) + ": " +
                             hex_code(static_cast<std::uint32_t>(described)));
            continue;
        }
        const bool software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
        if (!physical_adapter(desc.VendorId, software)) {
            continue;
        }
        Device device;
        device.index = index;
        device.luid = {static_cast<std::uint32_t>(desc.AdapterLuid.HighPart),
                       desc.AdapterLuid.LowPart};
        device.vendor_id = desc.VendorId;
        device.device_id = desc.DeviceId;
        device.vendor = vendor_name(desc.VendorId);
        device.name = utf8(desc.Description);
        device.vram_total_gb = bytes_to_gib(desc.DedicatedVideoMemory);
        device.shared_total_gb = bytes_to_gib(desc.SharedSystemMemory);
        device.integrated = adapter_uma(adapter.Get(), device.luid, errors);
        device.pci = adapter_pci(desc.AdapterLuid, device.luid, errors);
        LARGE_INTEGER version{};
        if (SUCCEEDED(adapter->CheckInterfaceSupport(
                __uuidof(IDXGIDevice), &version))) {
            device.driver_version = driver_version(
                static_cast<std::uint64_t>(version.QuadPart));
        }
        const bool external = device.vendor_id == 0x10de;
        const auto sensor = missing(
            external ? Status::unavailable : Status::error,
            external ? "nvidia-smi (external)" : "vendor SDK",
            external ? "NVIDIA physical sensors are supplied by nvidia-smi "
                       "in the Python runtime, not by this helper."
                     : "No vendor sensor backend has matched this adapter.");
        device.metrics[temperature] = sensor;
        device.metrics[power] = sensor;
        devices.push_back(std::move(device));
    }
    errors.emplace_back("DXGI exceeded the safety limit of 256 adapters.");
    return devices;
}

struct WddmMemory::Impl {
    PDH_HQUERY query = nullptr;
    Counter dedicated_counter;
    Counter shared_counter;

    Impl() {
        dedicated_counter.source = "PDH/WDDM adapter Dedicated Usage";
        shared_counter.source = "PDH/WDDM adapter Shared Usage";
        const auto result = PdhOpenQueryW(nullptr, 0, &query);
        if (result != ERROR_SUCCESS) {
            dedicated_counter.error = pdh_message("PdhOpenQueryW", result);
            shared_counter.error = dedicated_counter.error;
            return;
        }
        add(dedicated_counter, L"\\GPU Adapter Memory(*)\\Dedicated Usage");
        add(shared_counter, L"\\GPU Adapter Memory(*)\\Shared Usage");
    }

    ~Impl() {
        if (query) {
            PdhCloseQuery(query);
        }
    }

    void add(Counter& counter, const wchar_t* path) {
        // English counter names are resolved by PDH on localized Windows.
        const auto result =
            PdhAddEnglishCounterW(query, path, 0, &counter.handle);
        if (result != ERROR_SUCCESS) {
            counter.error = pdh_message("PdhAddEnglishCounterW", result);
        }
    }
};

WddmMemory::WddmMemory() : impl_(std::make_unique<Impl>()) {}
WddmMemory::~WddmMemory() = default;

void WddmMemory::sample(std::vector<Device>& devices,
                        std::vector<std::string>& errors) {
    if (impl_->query) {
        const auto result = PdhCollectQueryData(impl_->query);
        if (result != ERROR_SUCCESS) {
            const auto state = result == PDH_NO_DATA ? Status::unavailable
                                                    : Status::error;
            const auto message = pdh_message("PdhCollectQueryData", result);
            // Initialization errors must not turn into "no data".
            for (auto metric : {dedicated, shared}) {
                const auto& counter = metric == dedicated
                                          ? impl_->dedicated_counter
                                          : impl_->shared_counter;
                counter_failure(devices, metric, counter,
                                counter.error.empty() ? state : Status::error,
                                counter.error.empty() ? message
                                                      : counter.error);
            }
            return;
        }
    }
    read_counter(impl_->dedicated_counter, dedicated, devices, errors);
    read_counter(impl_->shared_counter, shared, devices, errors);
}

}  // namespace probe
