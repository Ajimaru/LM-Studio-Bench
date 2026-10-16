#include "vendor_status.h"

namespace probe {

Metric adlx_failure(ADLX_RESULT result, const std::string& operation,
                    bool sensor_query) {
    const auto status = sensor_query && result == ADLX_NOT_SUPPORTED
                            ? Status::unsupported : Status::error;
    return missing(status, "ADLX",
                   operation + " returned ADLX result " +
                   std::to_string(static_cast<int>(result)) + ".");
}

Metric intel_failure(ctl_result_t result, const std::string& operation,
                     bool sensor_query) {
    const auto status =
        sensor_query && result == CTL_RESULT_ERROR_UNSUPPORTED_FEATURE
            ? Status::unsupported : Status::error;
    return missing(status, "IGCL",
                   operation + " returned IGCL result " +
                   hex_code(static_cast<std::uint32_t>(result)) + ".");
}

Metric intel_loader_failure(std::uint32_t win32_error) {
    return missing(
        Status::error, "IGCL loader",
        "The documented System32 ControlLib.dll load failed; Win32 error " +
            std::to_string(win32_error) +
            ". This does not prove that compatible DriverStore copies are "
            "absent. No CWD/PATH search or unsupported runtime override "
            "was attempted.");
}

Metric intel_value(const ctl_oc_telemetry_item_t& item, ctl_units_t unit,
                   const std::string& operation, bool nonnegative) {
    if (!item.bSupported) {
        return missing(Status::unsupported, operation,
                       "The driver explicitly reports this sensor "
                       "unsupported.");
    }
    if (item.units != unit) {
        return missing(Status::error, operation,
                       "Unexpected telemetry units " +
                       std::to_string(static_cast<int>(item.units)) +
                       "; expected " + std::to_string(static_cast<int>(unit)) +
                       ". The value was not reinterpreted.");
    }
    double value = 0.0;
    switch (item.type) {
    case CTL_DATA_TYPE_INT8: value = item.value.data8; break;
    case CTL_DATA_TYPE_UINT8: value = item.value.datau8; break;
    case CTL_DATA_TYPE_INT16: value = item.value.data16; break;
    case CTL_DATA_TYPE_UINT16: value = item.value.datau16; break;
    case CTL_DATA_TYPE_INT32: value = item.value.data32; break;
    case CTL_DATA_TYPE_UINT32: value = item.value.datau32; break;
    case CTL_DATA_TYPE_INT64:
        value = static_cast<double>(item.value.data64);
        break;
    case CTL_DATA_TYPE_UINT64:
        value = static_cast<double>(item.value.datau64);
        break;
    case CTL_DATA_TYPE_FLOAT: value = item.value.datafloat; break;
    case CTL_DATA_TYPE_DOUBLE: value = item.value.datadouble; break;
    default:
        return missing(Status::error, operation,
                       "Unknown or non-numeric Intel telemetry data type.");
    }
    return reading(value, operation, nonnegative);
}

}  // namespace probe
