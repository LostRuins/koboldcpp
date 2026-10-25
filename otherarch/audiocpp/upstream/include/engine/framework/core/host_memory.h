#pragma once

#include <cstddef>

namespace engine::core {

// Physical memory the host still has free, or 0 when it cannot be determined
// (no portable query on this platform). Callers must treat 0 as "unknown" and
// fall back to whatever they would have done anyway, never as "no memory".
size_t available_host_memory_bytes();

// High-water marks of the calling process, read once at the end of a run.
// The OS keeps these regardless of whether anyone asks, so reading them costs
// one syscall and adds nothing to the run itself.
//
// footprint_bytes is the platform's own idea of what the process has claimed
// beyond resident pages, which is what an out-of-memory failure is measured
// against on that platform; footprint_source names which one:
//   "windows_commit"       peak commit charge (PeakPagefileUsage). This is the
//                          figure a Windows commit-limit failure compares to.
//   "linux_vsize"          peak virtual size (VmPeak). Over-counts commit: it
//                          includes reservations that are never charged.
//   "macos_phys_footprint" phys_footprint, the "Memory" column of Activity
//                          Monitor (dirty + compressed pages). macOS
//                          overcommits, so untouched allocations are absent.
// A field is 0 when the platform cannot report it, matching
// available_host_memory_bytes(): 0 means unknown, never "nothing".
struct ProcessMemoryPeak {
    size_t rss_bytes = 0;
    size_t footprint_bytes = 0;
    const char * footprint_source = "";
};
ProcessMemoryPeak process_memory_peak();

}  // namespace engine::core
