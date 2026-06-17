import Foundation
import Darwin

@MainActor
final class SystemMetricsProvider: ObservableObject {
    @Published var cpuPercent: Double = 0
    @Published var ramPercent: Double = 0
    @Published var diskPercent: Double = 0
    @Published var networkKBps: Double = 0   // combined in+out KB/s

    private var timer: Timer?
    private var prevCPUTicks: [UInt32] = []   // [user, system, idle, nice]
    private var prevNetBytes: UInt64 = 0
    private var prevNetTime: Date = Date()

    func start() {
        prevCPUTicks = cpuTicks()
        prevNetBytes = totalNetBytes()
        prevNetTime = Date()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in self?.refresh() }
        }
        RunLoop.main.add(timer!, forMode: .common)
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    private func refresh() {
        cpuPercent = readCPUPercent()
        ramPercent = readRAMPercent()
        diskPercent = readDiskPercent()
        networkKBps = readNetworkKBps()
    }

    // MARK: - CPU

    private func cpuTicks() -> [UInt32] {
        var info = host_cpu_load_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<host_cpu_load_info>.stride / MemoryLayout<integer_t>.stride
        )
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return [0, 0, 0, 0] }
        return [info.cpu_ticks.0, info.cpu_ticks.1, info.cpu_ticks.2, info.cpu_ticks.3]
    }

    private func readCPUPercent() -> Double {
        let curr = cpuTicks()
        guard curr.count == 4, prevCPUTicks.count == 4 else {
            prevCPUTicks = curr
            return cpuPercent
        }
        let dUser   = Double(curr[0] &- prevCPUTicks[0])
        let dSystem = Double(curr[1] &- prevCPUTicks[1])
        let dIdle   = Double(curr[2] &- prevCPUTicks[2])
        let dNice   = Double(curr[3] &- prevCPUTicks[3])
        let total   = dUser + dSystem + dIdle + dNice
        prevCPUTicks = curr
        guard total > 0 else { return cpuPercent }
        return min((dUser + dSystem + dNice) / total * 100, 100)
    }

    // MARK: - RAM

    private func readRAMPercent() -> Double {
        var vmStats = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64>.stride / MemoryLayout<integer_t>.stride
        )
        let kr = withUnsafeMutablePointer(to: &vmStats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return ramPercent }
        let pageSize = Double(vm_page_size)
        let physical = Double(ProcessInfo.processInfo.physicalMemory)
        let used = Double(
            vmStats.active_count + vmStats.inactive_count +
            vmStats.wire_count + vmStats.compressor_page_count
        ) * pageSize
        return min(used / physical * 100, 100)
    }

    // MARK: - Disk

    private func readDiskPercent() -> Double {
        let url = URL(fileURLWithPath: NSHomeDirectory())
        guard let values = try? url.resourceValues(
            forKeys: [.volumeTotalCapacityKey, .volumeAvailableCapacityForImportantUsageKey]
        ),
        let total = values.volumeTotalCapacity,
        let available = values.volumeAvailableCapacityForImportantUsage,
        total > 0 else { return diskPercent }
        let used = Int64(total) - available
        return min(Double(used) / Double(total) * 100, 100)
    }

    // MARK: - Network

    private func totalNetBytes() -> UInt64 {
        var total: UInt64 = 0
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddr) == 0 else { return 0 }
        defer { freeifaddrs(ifaddr) }
        var ptr = ifaddr
        while let p = ptr {
            let ifa = p.pointee
            if ifa.ifa_addr.pointee.sa_family == UInt8(AF_LINK) {
                let data = unsafeBitCast(ifa.ifa_data, to: UnsafeMutablePointer<if_data>.self)
                total += UInt64(data.pointee.ifi_ibytes) + UInt64(data.pointee.ifi_obytes)
            }
            ptr = ifa.ifa_next
        }
        return total
    }

    private func readNetworkKBps() -> Double {
        let now = Date()
        let curr = totalNetBytes()
        let elapsed = now.timeIntervalSince(prevNetTime)
        guard elapsed > 0 else { return networkKBps }
        let delta = curr &- prevNetBytes
        prevNetBytes = curr
        prevNetTime = now
        return Double(delta) / elapsed / 1024
    }
}
