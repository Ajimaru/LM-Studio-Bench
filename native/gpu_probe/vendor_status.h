// Validation of SDK statuses and tagged Intel telemetry values.
#pragma once

#include "probe_types.h"

#include <ADLXDefines.h>
#include <igcl_api.h>

namespace probe {

Metric adlx_failure(ADLX_RESULT result, const std::string& operation,
                    bool sensor_query = true);
Metric intel_failure(ctl_result_t result, const std::string& operation,
                     bool sensor_query = true);
Metric intel_loader_failure(std::uint32_t win32_error);
Metric intel_value(const ctl_oc_telemetry_item_t& item, ctl_units_t unit,
                   const std::string& operation, bool nonnegative = true);

}  // namespace probe
