// Bounded, read-only Windows telemetry with a versioned JSON-lines protocol.
#include "probe_types.h"
#include "probe_windows.h"
#include "vendor_sensors.h"

#include <Windows.h>
#include <fcntl.h>
#include <io.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {

using Clock = std::chrono::steady_clock;
std::atomic<HANDLE> console_stop{nullptr};

BOOL WINAPI console_handler(DWORD event) {
    if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT &&
        event != CTRL_CLOSE_EVENT && event != CTRL_SHUTDOWN_EVENT &&
        event != CTRL_LOGOFF_EVENT) {
        return FALSE;
    }
    if (const auto stop = console_stop.load()) {
        SetEvent(stop);
        return TRUE;
    }
    return FALSE;
}

class StopEvent {
public:
    StopEvent() {
        handle_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!handle_) {
            throw std::runtime_error("Cannot create the stop event.");
        }
        console_stop.store(handle_);
        SetConsoleCtrlHandler(console_handler, TRUE);
    }

    ~StopEvent() {
        SetConsoleCtrlHandler(console_handler, FALSE);
        console_stop.store(nullptr);
        CloseHandle(handle_);
    }

    bool wait(DWORD milliseconds = 0) const {
        return WaitForSingleObject(handle_, milliseconds) == WAIT_OBJECT_0;
    }

private:
    HANDLE handle_ = nullptr;
};

class DriverWatchdog {
public:
    explicit DriverWatchdog(probe::SamplingEpoch& epoch)
        : sampling_epoch_(epoch), thread_([this] { monitor(); }) {}

    ~DriverWatchdog() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closing_ = true;
        }
        condition_.notify_all();
        thread_.join();
    }

    void arm() {
        sampling_epoch_.observe();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_ = true;
            last_tick_ = Clock::now();
            deadline_ = last_tick_ + std::chrono::seconds(15);
        }
        condition_.notify_all();
    }

    void disarm() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_ = false;
        }
        condition_.notify_all();
    }

private:
    static DWORD WINAPI log_timeout(void*) {
        constexpr char message[] =
            "gpu_probe: driver/output operation exceeded 15 seconds; "
            "terminating the helper (exit 124).\n";
        DWORD written = 0;
        WriteFile(GetStdHandle(STD_ERROR_HANDLE), message,
                  sizeof(message) - 1, &written, nullptr);
        return 0;
    }

    void monitor() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!closing_) {
            sampling_epoch_.observe();
            if (!active_) {
                condition_.wait_for(
                    lock, std::chrono::milliseconds(100),
                    [this] { return closing_ || active_; });
                continue;
            }
            const auto now = Clock::now();
            const auto gap = now - last_tick_;
            // Whole-process suspension also stops this thread. A long gap
            // between its 100 ms ticks is not evidence of a blocked driver.
            if (gap > std::chrono::seconds(1)) {
                deadline_ += gap;
            }
            last_tick_ = now;
            if (now >= deadline_) {
                lock.unlock();
                // A blocked stderr reader must not deadlock the watchdog.
                const auto logger = CreateThread(
                    nullptr, 0, log_timeout, nullptr, 0, nullptr);
                if (logger) {
                    WaitForSingleObject(logger, 100);
                    CloseHandle(logger);
                }
                // An unresponsive in-process driver cannot be safely unwound.
                // Kill only ourselves; the Python parent can report/restart it.
                TerminateProcess(GetCurrentProcess(), 124);
                std::_Exit(124);
            }
            const auto due = deadline_;
            const auto tick =
                std::min(due, now + std::chrono::milliseconds(100));
            condition_.wait_until(lock, tick, [this, due] {
                return closing_ || !active_ || deadline_ != due;
            });
        }
    }

    std::mutex mutex_;
    std::condition_variable condition_;
    bool closing_ = false;
    bool active_ = false;
    Clock::time_point deadline_;
    Clock::time_point last_tick_;
    probe::SamplingEpoch& sampling_epoch_;
    std::thread thread_;
};

class ArmOnExit {
public:
    explicit ArmOnExit(DriverWatchdog& watchdog) : watchdog_(watchdog) {}
    ~ArmOnExit() { watchdog_.arm(); }

private:
    DriverWatchdog& watchdog_;
};

class StopInput {
public:
    // Poll pipes instead of detaching a thread blocked in stdin. This permits
    // bounded Ctrl+C shutdown even while the parent keeps its pipe open.
    bool poll() {
        const auto input = GetStdHandle(STD_INPUT_HANDLE);
        if (!input || input == INVALID_HANDLE_VALUE) {
            return true;
        }
        const auto type = GetFileType(input);
        if (type == FILE_TYPE_CHAR) {
            DWORD available = 0;
            if (!GetNumberOfConsoleInputEvents(input, &available)) {
                return true;
            }
            for (DWORD i = 0; i < std::min<DWORD>(available, 256); ++i) {
                INPUT_RECORD record{};
                DWORD count = 0;
                if (!ReadConsoleInputW(input, &record, 1, &count)) {
                    return true;
                }
                if (record.EventType != KEY_EVENT ||
                    !record.Event.KeyEvent.bKeyDown) {
                    continue;
                }
                const auto ch = record.Event.KeyEvent.uChar.UnicodeChar;
                if (ch == L'\r') {
                    if (consume('\n')) {
                        return true;
                    }
                } else if (ch == L'\b') {
                    if (!line_.empty()) {
                        line_.pop_back();
                    }
                } else if (ch > 0 && ch < 128 &&
                           consume(static_cast<char>(ch))) {
                    return true;
                }
            }
            return false;
        }
        DWORD available = 256;
        if (type == FILE_TYPE_PIPE) {
            if (!PeekNamedPipe(
                    input, nullptr, 0, nullptr, &available, nullptr)) {
                return true;  // Closed pipe is stdin EOF.
            }
            if (!available) {
                return false;
            }
        } else if (type != FILE_TYPE_DISK) {
            return true;
        }
        char buffer[256];
        DWORD count = 0;
        if (!ReadFile(input, buffer, std::min<DWORD>(available, sizeof(buffer)),
                      &count, nullptr) || !count) {
            return true;
        }
        for (DWORD i = 0; i < count; ++i) {
            if (consume(buffer[i])) {
                return true;
            }
        }
        return false;
    }

private:
    bool consume(char ch) {
        if (ch == '\r') {
            return false;
        }
        if (ch == '\n') {
            const bool stop = !too_long_ && line_ == "stop";
            if (!stop && (!line_.empty() || too_long_)) {
                std::cerr << "gpu_probe: ignored stdin command; use stop.\n";
            }
            line_.clear();
            too_long_ = false;
            return stop;
        }
        if (line_.size() < 4096) {
            line_ += ch;
        } else {
            too_long_ = true;
        }
        return false;
    }

    std::string line_;
    bool too_long_ = false;
};

struct Options {
    bool watch = false;
    bool help = false;
    unsigned int interval_ms = 1000;
};

Options options(int argc, char** argv) {
    Options result;
    bool mode_set = false;
    bool interval_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--help") {
            result.help = true;
        } else if (argument == "--once" || argument == "--watch") {
            if (mode_set) {
                throw std::invalid_argument("Choose --once or --watch once.");
            }
            mode_set = true;
            result.watch = argument == "--watch";
        } else if (argument == "--interval-ms") {
            if (++i == argc || interval_set) {
                throw std::invalid_argument("Supply one --interval-ms value.");
            }
            interval_set = true;
            const std::string_view value(argv[i]);
            const auto parsed = std::from_chars(
                value.data(), value.data() + value.size(), result.interval_ms);
            if (parsed.ec != std::errc{} ||
                parsed.ptr != value.data() + value.size() ||
                result.interval_ms < 100 || result.interval_ms > 60000) {
                throw std::invalid_argument(
                    "--interval-ms must be an integer from 100 to 60000.");
            }
        } else {
            throw std::invalid_argument("Unknown argument: " +
                                        std::string(argument));
        }
    }
    if (interval_set && !result.watch) {
        throw std::invalid_argument("--interval-ms requires --watch.");
    }
    return result;
}

int run(const Options& args) {
    probe::SamplingEpoch sampling_epoch;
    StopEvent stop;
    DriverWatchdog watchdog(sampling_epoch);
    watchdog.arm();
    {
        probe::Snapshot initial;
        initial.devices = probe::enumerate_devices(initial.errors);
        probe::WddmMemory memory;
        probe::AmdSensors amd(initial.devices, initial.errors);
        probe::IntelSensors intel(
            initial.devices, initial.errors, sampling_epoch,
            std::chrono::milliseconds(args.interval_ms));
        ArmOnExit bounded_cleanup(watchdog);
        watchdog.disarm();
        StopInput input;
        std::uint64_t sequence = 0;
        std::vector<std::string> previous_errors;
        auto next = Clock::now();
        for (;;) {
            watchdog.arm();
            auto frame = initial;
            frame.sequence = sequence++;
            frame.timestamp_unix_ms = probe::unix_timestamp_ms();
            memory.sample(frame.devices, frame.errors);
            amd.sample(frame.devices);
            intel.sample(frame.devices);
            probe::append_metric_errors(frame);
            const auto json = probe::to_json(frame);
            if (frame.errors != previous_errors) {
                for (const auto& error : frame.errors) {
                    std::cerr << "gpu_probe: " << error << '\n';
                }
                previous_errors = frame.errors;
            }
            std::cout << json << '\n' << std::flush;
            if (!std::cout) {
                throw std::runtime_error("The protocol output pipe is closed.");
            }
            watchdog.disarm();
            if (!args.watch || stop.wait()) {
                break;
            }
            next += std::chrono::milliseconds(args.interval_ms);
            if (next < Clock::now()) {
                next = Clock::now();
            }
            bool stopping = false;
            do {
                if (input.poll() || stop.wait()) {
                    stopping = true;
                    break;
                }
                const auto left =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        next - Clock::now()).count();
                if (left <= 0) {
                    break;
                }
                if (stop.wait(static_cast<DWORD>(
                        std::min<std::int64_t>(left, 50)))) {
                    stopping = true;
                    break;
                }
            } while (Clock::now() < next);
            if (stopping) {
                break;
            }
        }
    }
    watchdog.disarm();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX |
                 SEM_NOOPENFILEERRORBOX);
    _setmode(_fileno(stdout), _O_BINARY);
    Options args;
    try {
        args = options(argc, argv);
    } catch (const std::invalid_argument& error) {
        std::cerr << "gpu_probe: " << error.what() << '\n';
        return 2;
    }
    if (args.help) {
        std::cerr << "gpu_probe --once\n"
                     "gpu_probe --watch [--interval-ms 1000]\n"
                     "Interval: 100..60000 ms; "
                     "stop via stop\\n, EOF or Ctrl+C.\n"
                     "15 s driver/output watchdog excludes suspension gaps "
                     "(exit 124).\n";
        return 0;
    }
    try {
        return run(args);
    } catch (const std::exception& error) {
        std::cerr << "gpu_probe: " << error.what() << '\n';
        probe::Snapshot failure;
        failure.timestamp_unix_ms = probe::unix_timestamp_ms();
        failure.errors.push_back(std::string("Probe failure: ") + error.what());
        std::cout << probe::to_json(failure) << '\n' << std::flush;
        return 1;
    }
}
