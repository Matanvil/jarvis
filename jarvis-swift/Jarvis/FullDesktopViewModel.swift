import Foundation
import Combine

@MainActor
final class FullDesktopViewModel: ObservableObject {

    // MARK: - Jarvis session metrics (updated by AudioController)
    @Published var tokS: Double = 0
    @Published var ttftMs: Int = 0
    @Published var executorModel: String = "—"
    @Published var intentClass: String = "—"
    @Published var totalTokens: Int = 0
    @Published var turnCount: Int = 0

    // MARK: - Active tools (tool name → used this session)
    @Published var activeTools: [String] = []   // ordered by first use

    // MARK: - Live log (last 8 lines from commands.log)
    @Published var logLines: [String] = []

    private let logPath = NSHomeDirectory() + "/.jarvis/logs/commands.log"
    private var logTimer: Timer?

    // MARK: - Lifecycle

    func start() {
        pollLog()
        let t = Timer(timeInterval: 2.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in self?.pollLog() }
        }
        RunLoop.main.add(t, forMode: .common)
        logTimer = t
    }

    func stop() {
        logTimer?.invalidate()
        logTimer = nil
    }

    // MARK: - Called by AudioController

    func updateMetrics(tokS: Double, ttftMs: Int, model: String, intentClass: String, genTokens: Int) {
        self.tokS = tokS
        self.ttftMs = ttftMs
        self.executorModel = model.isEmpty ? "—" : model
        self.intentClass = intentClass.isEmpty ? "—" : intentClass
        self.totalTokens += genTokens
        self.turnCount += 1
    }

    func recordToolUsed(_ tool: String) {
        let cleaned = tool.components(separatedBy: " ").first ?? tool
        guard !activeTools.contains(cleaned) else { return }
        activeTools.append(cleaned)
    }

    func resetSession() {
        tokS = 0
        ttftMs = 0
        executorModel = "—"
        intentClass = "—"
        totalTokens = 0
        turnCount = 0
        activeTools = []
    }

    enum LogSource: String, CaseIterable {
        case commands = "Commands"
        case analytics = "Analytics"
        case errors = "Errors"

        var path: String {
            let base = NSHomeDirectory() + "/.jarvis/logs/"
            switch self {
            case .commands:  return base + "commands.log"
            case .analytics: return base + "analytics.log"
            case .errors:    return base + "errors.log"
            }
        }
    }

    func allLogLines() -> [String] { allLogLines(source: .commands) }

    func allLogLines(source: LogSource) -> [String] {
        guard let content = try? String(contentsOfFile: source.path, encoding: .utf8) else { return [] }
        return content
            .components(separatedBy: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            .reversed()
    }

    // MARK: - Log polling

    private func pollLog() {
        guard let content = try? String(contentsOfFile: logPath, encoding: .utf8) else { return }
        let lines = content
            .components(separatedBy: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        logLines = Array(lines.suffix(8))
    }
}
