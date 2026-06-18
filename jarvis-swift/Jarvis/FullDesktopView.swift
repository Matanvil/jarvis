import SwiftUI

// MARK: - Design tokens
private let bgPrimary    = Color(red: 0.039, green: 0.059, blue: 0.098)
private let bgSurface    = Color(red: 0.051, green: 0.094, blue: 0.157)
private let accentCyan   = Color(red: 0.055, green: 0.647, blue: 0.914)
private let liveCyan     = Color(red: 0.133, green: 0.827, blue: 0.933)
private let textSecond   = Color(red: 0.392, green: 0.447, blue: 0.561)
private let textDim      = Color(red: 0.200, green: 0.267, blue: 0.337)
private let cardBorder   = Color(red: 0.055, green: 0.647, blue: 0.914).opacity(0.12)

// MARK: - Shared card container

struct DesktopCard<Content: View>: View {
    let icon: String
    let label: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Image(systemName: icon)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundColor(accentCyan)
                Text(label)
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .kerning(1.5)
                    .foregroundColor(accentCyan)
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(bgSurface)

            Divider().background(cardBorder)

            content()
        }
        .background(bgSurface)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(cardBorder, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - Left panel

struct ActivityCard: View {
    @ObservedObject var viewModel: HUDViewModel

    private var stepText: String {
        switch viewModel.state {
        case .executing(let step): return step
        case .thinking:            return "Thinking…"
        case .listening:           return "Listening…"
        case .response:            return "Response ready"
        default:                   return "Idle"
        }
    }

    private var progress: Double {
        switch viewModel.state {
        case .executing: return 0.72
        case .thinking:  return 0.4
        default:         return 0
        }
    }

    var body: some View {
        DesktopCard(icon: "arrow.triangle.2.circlepath", label: "WHAT JARVIS IS DOING") {
            VStack(alignment: .leading, spacing: 6) {
                Text(stepText)
                    .font(.system(size: 12))
                    .foregroundColor(liveCyan)
                ProgressView(value: progress)
                    .tint(accentCyan)
                    .scaleEffect(x: 1, y: 0.5)
            }
            .padding(12)
        }
    }
}

struct RecentTurnsCard: View {
    @ObservedObject var viewModel: HUDViewModel

    private var recentTurns: [ConversationTurn] {
        Array(viewModel.turns.suffix(5).reversed())
    }

    var body: some View {
        DesktopCard(icon: "bubble.left.and.bubble.right", label: "LAST 5 RESPONSES") {
            VStack(spacing: 0) {
                ForEach(recentTurns) { turn in
                    HStack {
                        Text(turn.command)
                            .font(.system(size: 11))
                            .foregroundColor(textSecond)
                            .lineLimit(1)
                        Spacer()
                        if turn.response != nil {
                            Image(systemName: "checkmark")
                                .font(.system(size: 9, weight: .semibold))
                                .foregroundColor(liveCyan)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    Divider().background(cardBorder)
                }
            }
        }
    }
}

struct LeftPanel: View {
    @ObservedObject var viewModel: HUDViewModel

    var body: some View {
        VStack(spacing: 8) {
            ActivityCard(viewModel: viewModel)
            RecentTurnsCard(viewModel: viewModel)
            Spacer()
        }
        .padding(8)
        .frame(width: 265)
        .background(bgPrimary)
    }
}

// MARK: - Right panel

struct MetricRow: View {
    let label: String
    let value: String
    var fillFraction: Double? = nil

    var body: some View {
        VStack(spacing: 3) {
            HStack {
                Text(label)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(textSecond)
                Spacer()
                Text(value)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(Color(red: 0.49, green: 0.83, blue: 0.99))
            }
            if let fill = fillFraction {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Color(red: 0.118, green: 0.227, blue: 0.373))
                            .frame(height: 3)
                        RoundedRectangle(cornerRadius: 2)
                            .fill(accentCyan)
                            .frame(width: geo.size.width * CGFloat(min(fill / 100, 1)), height: 3)
                    }
                }
                .frame(height: 3)
            }
        }
    }
}

struct SystemTelemetryCard: View {
    @ObservedObject var metrics: SystemMetricsProvider

    var body: some View {
        DesktopCard(icon: "antenna.radiowaves.left.and.right", label: "SYSTEM TELEMETRY") {
            VStack(spacing: 8) {
                MetricRow(label: "CPU",  value: "\(Int(metrics.cpuPercent))%",  fillFraction: metrics.cpuPercent)
                MetricRow(label: "RAM",  value: "\(Int(metrics.ramPercent))%",  fillFraction: metrics.ramPercent)
                MetricRow(label: "DISK", value: "\(Int(metrics.diskPercent))%", fillFraction: metrics.diskPercent)
                MetricRow(label: "NET",  value: String(format: "%.0f KB/s", metrics.networkKBps))
            }
            .padding(12)
        }
    }
}

private let allTools = ["shell", "web_search", "read_file", "write_file", "run_snippet", "notify", "delegate_to_local"]

struct ActiveToolsCard: View {
    @ObservedObject var fullViewModel: FullDesktopViewModel

    var body: some View {
        DesktopCard(icon: "bolt", label: "ACTIVE TOOLS") {
            VStack(spacing: 0) {
                ForEach(allTools, id: \.self) { tool in
                    let isActive = fullViewModel.activeTools.contains(tool)
                    HStack(spacing: 8) {
                        Circle()
                            .fill(isActive ? liveCyan : Color(red: 0.118, green: 0.227, blue: 0.373))
                            .frame(width: 6, height: 6)
                            .shadow(color: isActive ? liveCyan : .clear, radius: 3)
                        Text(tool)
                            .font(.system(size: 11))
                            .foregroundColor(textSecond)
                        Spacer()
                        Text(isActive ? "ACTIVE" : "STANDBY")
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundColor(isActive ? liveCyan : textDim)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 5)
                    Divider().background(cardBorder)
                }
            }
        }
    }
}

struct ModelRoutingCard: View {
    @ObservedObject var fullViewModel: FullDesktopViewModel

    var body: some View {
        DesktopCard(icon: "arrow.triangle.branch", label: "MODEL ROUTING") {
            VStack(alignment: .leading, spacing: 10) {
                // ── Last reply ────────────────────────────────────────────
                Text("LAST REPLY")
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .foregroundColor(.white.opacity(0.35))
                    .kerning(1)
                VStack(spacing: 5) {
                    MetricRow(label: "tok/s",   value: fullViewModel.lastTokS > 0
                                                       ? String(format: "%.0f", fullViewModel.lastTokS) : "—")
                    MetricRow(label: "ttft",    value: fullViewModel.lastTtftMs > 0
                                                       ? "\(fullViewModel.lastTtftMs)ms" : "—")
                    MetricRow(label: "tokens",  value: fullViewModel.lastGenTokens > 0
                                                       ? "\(fullViewModel.lastGenTokens)" : "—")
                    MetricRow(label: "model",   value: fullViewModel.lastModel)
                    MetricRow(label: "intent",  value: fullViewModel.lastIntentClass)
                }

                Divider().background(Color.white.opacity(0.08))

                // ── Session ───────────────────────────────────────────────
                Text("SESSION")
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .foregroundColor(.white.opacity(0.35))
                    .kerning(1)
                VStack(spacing: 5) {
                    MetricRow(label: "turns",     value: "\(fullViewModel.turnCount)")
                    MetricRow(label: "tokens",    value: "\(fullViewModel.totalTokens)")
                    MetricRow(label: "avg tok/s", value: fullViewModel.avgTokS > 0
                                                         ? String(format: "%.0f", fullViewModel.avgTokS) : "—")
                    MetricRow(label: "avg ttft",  value: fullViewModel.avgTtftMs > 0
                                                         ? "\(fullViewModel.avgTtftMs)ms" : "—")
                }
            }
            .padding(12)
        }
    }
}

// MARK: - Log row renderers

private let logTextPrimary  = Color.white.opacity(0.90)
private let logTextSecond   = Color.white.opacity(0.65)
private let logTextDim      = Color.white.opacity(0.40)
private let logAccent       = Color(red: 0.2, green: 0.75, blue: 1.0)
private let logRowBg        = Color.white.opacity(0.04)
private let logRowBorder    = Color.white.opacity(0.08)

private func extractBetween(_ s: String, after: String, before: String) -> String? {
    guard let a = s.range(of: after) else { return nil }
    let sub = String(s[a.upperBound...])
    guard let b = sub.range(of: before) else { return String(sub.prefix(300)) }
    return String(sub[..<b.lowerBound])
}

struct CommandLogRow: View {
    let raw: String

    private var ts: String {
        extractBetween(raw, after: "", before: " INFO") ?? ""
    }
    private var cmd: String {
        extractBetween(raw, after: "cmd='", before: "' cwd=") ?? raw
    }
    private var durationMs: String? {
        extractBetween(raw, after: "duration_ms=", before: " result=")
    }
    private var tokS: String? {
        guard let v = extractBetween(raw, after: "'tok_s': ", before: ","),
              v != "null" else { return nil }
        return v
    }
    private var model: String? {
        guard let v = extractBetween(raw, after: "'_model': '", before: "'"),
              !v.isEmpty else { return nil }
        return v
    }
    private var intent: String? {
        guard let v = extractBetween(raw, after: "'_intent_class': '", before: "'"),
              v != "null", !v.isEmpty else { return nil }
        return v
    }
    private var speak: String? {
        guard let v = extractBetween(raw, after: "'speak': '", before: "', 'display'"),
              !v.isEmpty else { return nil }
        return v
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(alignment: .top, spacing: 6) {
                Text(ts)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(logTextDim)
                Spacer()
                HStack(spacing: 4) {
                    if let ms = durationMs { pill("\(ms)ms", color: .white.opacity(0.12)) }
                    if let t = tokS        { pill("\(t) tok/s", color: logAccent.opacity(0.3)) }
                    if let i = intent      { pill(i, color: Color.purple.opacity(0.35)) }
                }
            }
            Text(cmd)
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(logTextPrimary)
                .textSelection(.enabled)
            if let m = model {
                Text(m)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(logAccent.opacity(0.8))
            }
            if let s = speak, !s.isEmpty {
                Text(s)
                    .font(.system(size: 11))
                    .foregroundColor(logTextSecond)
                    .lineLimit(2)
                    .textSelection(.enabled)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(logRowBg)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(logRowBorder, lineWidth: 1))
    }

    private func pill(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .semibold, design: .monospaced))
            .foregroundColor(.white.opacity(0.85))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color)
            .clipShape(Capsule())
    }
}

struct AnalyticsLogRow: View {
    let raw: String

    private var parsed: [String: Any]? {
        guard let data = raw.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return obj
    }
    private var ts: String {
        guard let ts = parsed?["ts"] as? Double else { return "" }
        let d = Date(timeIntervalSince1970: ts)
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return f.string(from: d)
    }
    private var data: [String: Any] { parsed?["data"] as? [String: Any] ?? [:] }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(ts)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(logTextDim)
                Spacer()
                if let escalated = data["escalated"] as? Bool, escalated {
                    pill("ESCALATED", color: .orange.opacity(0.45))
                }
            }
            HStack(spacing: 5) {
                if let ms = data["duration_ms"] as? Int  { pill("\(ms)ms", color: .white.opacity(0.12)) }
                if let ttft = data["ttft_ms"] as? Int     { pill("ttft \(ttft)ms", color: .white.opacity(0.12)) }
                if let t = data["tok_s"] as? Double       { pill(String(format: "%.1f tok/s", t), color: logAccent.opacity(0.3)) }
                if let n = data["gen_tokens"] as? Int     { pill("\(n) tokens", color: .white.opacity(0.12)) }
            }
            if let model = data["model"] as? String {
                Text(model)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(logAccent.opacity(0.8))
                    .textSelection(.enabled)
            }
            if let reason = data["escalation_reason"] as? String {
                Text(reason)
                    .font(.system(size: 11))
                    .foregroundColor(Color.orange.opacity(0.9))
                    .lineLimit(2)
                    .textSelection(.enabled)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(logRowBg)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(logRowBorder, lineWidth: 1))
    }

    private func pill(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .semibold, design: .monospaced))
            .foregroundColor(.white.opacity(0.85))
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(color).clipShape(Capsule())
    }
}

struct ErrorLogRow: View {
    let raw: String

    private var isTimestamped: Bool { raw.count > 19 && raw.first?.isNumber == true }
    private var isError: Bool { raw.contains("ERROR") || raw.contains("Error") || raw.contains("Exception") }
    private var isTraceback: Bool { raw.hasPrefix("  File") || raw.hasPrefix("Traceback") }

    var body: some View {
        Text(raw)
            .font(.system(size: 11, design: .monospaced))
            .foregroundColor(rowColor)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 12)
            .padding(.vertical, isTimestamped ? 4 : 1)
    }

    private var rowColor: Color {
        if isError { return Color(red: 1.0, green: 0.35, blue: 0.35) }
        if isTraceback { return Color(red: 1.0, green: 0.65, blue: 0.2) }
        if isTimestamped { return logTextSecond }
        return logTextDim
    }
}

// MARK: - Log Modal

struct LogModalView: View {
    @ObservedObject var fullViewModel: FullDesktopViewModel
    @Binding var isPresented: Bool

    private let pageSize = 100
    @State private var source: FullDesktopViewModel.LogSource = .commands
    @State private var page = 0
    @State private var lines: [String] = []

    private var totalPages: Int { max(1, Int(ceil(Double(lines.count) / Double(pageSize)))) }
    private var pageLines: [String] {
        let start = page * pageSize
        let end = min(start + pageSize, lines.count)
        guard start < lines.count else { return [] }
        return Array(lines[start..<end])
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack(spacing: 12) {
                Text("LOGS")
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundColor(accentCyan)
                Picker("", selection: $source) {
                    ForEach(FullDesktopViewModel.LogSource.allCases, id: \.self) { s in
                        Text(s.rawValue).tag(s)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 260)
                .onChange(of: source) { _, _ in reload() }
                Spacer()
                Text("\(lines.count) entries")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(logTextDim)
                Button(action: { isPresented = false }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 16))
                        .foregroundColor(.white.opacity(0.5))
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(bgPrimary)

            Divider().background(cardBorder)

            // Log rows
            ScrollView {
                LazyVStack(alignment: .leading, spacing: source == .errors ? 0 : 6) {
                    ForEach(Array(pageLines.enumerated()), id: \.offset) { _, line in
                        switch source {
                        case .commands:  CommandLogRow(raw: line)
                        case .analytics: AnalyticsLogRow(raw: line)
                        case .errors:    ErrorLogRow(raw: line)
                        }
                    }
                }
                .padding(source == .errors ? 0 : 10)
            }
            .background(bgSurface)

            Divider().background(cardBorder)

            // Pagination
            HStack(spacing: 16) {
                Button(action: { page = max(0, page - 1) }) {
                    Image(systemName: "chevron.left").font(.system(size: 11, weight: .semibold))
                }
                .buttonStyle(.plain)
                .foregroundColor(page > 0 ? accentCyan : textDim)
                .disabled(page == 0)

                Text("Page \(page + 1) of \(totalPages)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(textSecond)

                Button(action: { page = min(totalPages - 1, page + 1) }) {
                    Image(systemName: "chevron.right").font(.system(size: 11, weight: .semibold))
                }
                .buttonStyle(.plain)
                .foregroundColor(page < totalPages - 1 ? accentCyan : textDim)
                .disabled(page >= totalPages - 1)

                Spacer()

                Button(action: reload) {
                    Label("Refresh", systemImage: "arrow.clockwise").font(.system(size: 10))
                }
                .buttonStyle(.plain)
                .foregroundColor(accentCyan)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(bgPrimary)
        }
        .frame(width: 820, height: 600)
        .background(bgSurface)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(cardBorder, lineWidth: 1))
        .onAppear { reload() }
        .preferredColorScheme(.dark)
    }

    private func reload() {
        lines = fullViewModel.allLogLines(source: source)
        page = 0
    }
}

struct LiveLogCard: View {
    @ObservedObject var fullViewModel: FullDesktopViewModel
    @State private var showModal = false

    var body: some View {
        DesktopCard(icon: "doc.text", label: "LIVE LOG") {
            VStack(alignment: .leading, spacing: 2) {
                ForEach(Array(fullViewModel.logLines.enumerated()), id: \.offset) { i, line in
                    Text(line)
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(i >= fullViewModel.logLines.count - 2 ? textSecond : textDim)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                HStack {
                    Spacer()
                    Text("tap to expand")
                        .font(.system(size: 8, design: .monospaced))
                        .foregroundColor(textDim.opacity(0.5))
                }
            }
            .padding(12)
        }
        .onTapGesture { showModal = true }
        .sheet(isPresented: $showModal) {
            LogModalView(fullViewModel: fullViewModel, isPresented: $showModal)
        }
    }
}

struct RightPanel: View {
    @ObservedObject var fullViewModel: FullDesktopViewModel
    @ObservedObject var metrics: SystemMetricsProvider

    var body: some View {
        ScrollView {
            VStack(spacing: 8) {
                SystemTelemetryCard(metrics: metrics)
                ActiveToolsCard(fullViewModel: fullViewModel)
                ModelRoutingCard(fullViewModel: fullViewModel)
                LiveLogCard(fullViewModel: fullViewModel)
            }
            .padding(8)
        }
        .frame(width: 265)
        .background(bgPrimary)
    }
}

// MARK: - Center column

struct OrbZoneView: View {
    @ObservedObject var viewModel: HUDViewModel
    @ObservedObject var fullViewModel: FullDesktopViewModel
    var onCollapse: () -> Void

    private var orbSize: CGFloat {
        viewModel.turns.isEmpty ? 120 : 100
    }

    private var stabilityLabel: String {
        switch viewModel.state {
        case .executing: return "ACTIVE"
        case .thinking:  return "PROCESSING"
        default:         return "EXCELLENT"
        }
    }

    var body: some View {
        ZStack {
            // Corner bracket decorations
            VStack {
                HStack {
                    cornerBracket(topLeft: true)
                    Spacer()
                    cornerBracket(topLeft: false)
                }
                Spacer()
                HStack {
                    cornerBracket(bottomLeft: true)
                    Spacer()
                    cornerBracket(bottomLeft: false)
                }
            }
            .padding(12)

            // Collapse button
            VStack {
                HStack {
                    Spacer()
                    Button(action: onCollapse) {
                        Image(systemName: "arrow.down.right.and.arrow.up.left")
                            .font(.system(size: 11))
                            .foregroundColor(textSecond)
                            .padding(6)
                            .background(bgSurface)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(cardBorder, lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                    .padding(12)
                }
                Spacer()
            }

            // Orb + flanking labels + stats
            VStack(spacing: 16) {
                HStack(spacing: 40) {
                    VStack {
                        Text("42°C")
                            .font(.system(size: 20, design: .monospaced))
                            .foregroundColor(Color(red: 0.49, green: 0.83, blue: 0.99))
                        Text("CORE TEMP")
                            .font(.system(size: 8, design: .monospaced))
                            .foregroundColor(textDim)
                            .kerning(1)
                    }

                    ArcReactorView(size: orbSize)
                        .animation(.easeInOut(duration: 0.3), value: orbSize)

                    VStack {
                        Text("87%")
                            .font(.system(size: 20, design: .monospaced))
                            .foregroundColor(Color(red: 0.49, green: 0.83, blue: 0.99))
                        Text("ENERGY LEVEL")
                            .font(.system(size: 8, design: .monospaced))
                            .foregroundColor(textDim)
                            .kerning(1)
                    }
                }

                HStack(spacing: 24) {
                    statPill(value: fullViewModel.lastTokS > 0
                                   ? String(format: "%.0f", fullViewModel.lastTokS) : "—", label: "TOK/S")
                    statPill(value: fullViewModel.lastTtftMs > 0
                                   ? "\(fullViewModel.lastTtftMs)ms" : "—", label: "TTFT")
                    statPill(value: fullViewModel.lastModel, label: "MODEL")
                }

                Text("SYSTEM STABILITY — \(stabilityLabel)")
                    .font(.system(size: 10, design: .monospaced))
                    .kerning(2)
                    .foregroundColor(textSecond)
            }
        }
        .frame(maxWidth: .infinity)
        .background(
            RadialGradient(
                colors: [accentCyan.opacity(0.04), bgPrimary],
                center: .center,
                startRadius: 60,
                endRadius: 300
            )
        )
    }

    private func statPill(value: String, label: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(size: 13, design: .monospaced))
                .foregroundColor(liveCyan)
            Text(label)
                .font(.system(size: 8, design: .monospaced))
                .foregroundColor(textDim)
                .kerning(1)
        }
    }

    private func cornerBracket(topLeft: Bool = false, bottomLeft: Bool = false) -> some View {
        let top  = topLeft || (!topLeft && !bottomLeft)
        let left = topLeft || bottomLeft
        return Rectangle()
            .fill(Color.clear)
            .frame(width: 16, height: 16)
            .overlay(
                ZStack {
                    if top && left   { topLeftBracket }
                    if top && !left  { topRightBracket }
                    if !top && left  { bottomLeftBracket }
                    if !top && !left { bottomRightBracket }
                }
            )
    }

    private var topLeftBracket: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                Rectangle().fill(accentCyan.opacity(0.25)).frame(width: 1, height: 16)
                Spacer()
            }
            HStack(spacing: 0) {
                Rectangle().fill(accentCyan.opacity(0.25)).frame(width: 16, height: 1)
                Spacer()
            }
        }
    }

    private var topRightBracket: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                Spacer()
                Rectangle().fill(accentCyan.opacity(0.25)).frame(width: 1, height: 16)
            }
            HStack(spacing: 0) {
                Spacer()
                Rectangle().fill(accentCyan.opacity(0.25)).frame(width: 16, height: 1)
            }
        }
    }

    private var bottomLeftBracket: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                Rectangle().fill(accentCyan.opacity(0.25)).frame(width: 16, height: 1)
                Spacer()
            }
            HStack(spacing: 0) {
                Rectangle().fill(accentCyan.opacity(0.25)).frame(width: 1, height: 16)
                Spacer()
            }
        }
    }

    private var bottomRightBracket: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                Spacer()
                Rectangle().fill(accentCyan.opacity(0.25)).frame(width: 16, height: 1)
            }
            HStack(spacing: 0) {
                Spacer()
                Rectangle().fill(accentCyan.opacity(0.25)).frame(width: 1, height: 16)
            }
        }
    }
}

private let actionChips = ["📎 Attach", "🌐 Web Search", "🧠 Deep Think", "📊 Analyze"]

struct DesktopInputBar: View {
    @ObservedObject var viewModel: HUDViewModel
    var onTextCommand: (String) -> Void
    var onVoice: () -> Void

    @State private var inputText = ""
    @FocusState private var focused: Bool

    private var isDisabled: Bool {
        switch viewModel.state {
        case .listening, .thinking, .executing: return true
        default: return false
        }
    }

    var body: some View {
        VStack(spacing: 8) {
            // Placeholder action chips — not wired, coming soon
            HStack(spacing: 6) {
                ForEach(actionChips, id: \.self) { chip in
                    Text(chip)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(textDim)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 4)
                        .overlay(Capsule().stroke(cardBorder, lineWidth: 1))
                }
                .opacity(0.4)
                Spacer()
            }

            HStack(spacing: 8) {
                Button(action: onVoice) {
                    Image(systemName: "mic.circle.fill")
                        .font(.system(size: 28))
                        .foregroundColor(accentCyan)
                }
                .buttonStyle(.plain)

                TextField("Ask Jarvis anything…", text: $inputText)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13))
                    .foregroundColor(.white)
                    .focused($focused)
                    .disabled(isDisabled)
                    .onSubmit {
                        guard !inputText.trimmingCharacters(in: .whitespaces).isEmpty else { return }
                        onTextCommand(inputText)
                        inputText = ""
                    }
                    .onChange(of: viewModel.state) { _, newState in
                        // Re-focus after each command completes so the user can type immediately.
                        switch newState {
                        case .response, .approved, .denied:
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { focused = true }
                        default: break
                        }
                    }
                    .onAppear { focused = true }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(accentCyan.opacity(0.04))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(cardBorder, lineWidth: 1))
                    .clipShape(RoundedRectangle(cornerRadius: 8))

                Button {
                    guard !inputText.trimmingCharacters(in: .whitespaces).isEmpty else { return }
                    onTextCommand(inputText)
                    inputText = ""
                } label: {
                    Image(systemName: "paperplane.fill")
                        .font(.system(size: 14))
                        .foregroundColor(.white)
                        .padding(10)
                        .background(accentCyan)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .shadow(color: accentCyan.opacity(0.4), radius: 8)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(14)
        .background(bgSurface)
        .overlay(Rectangle().frame(height: 1).foregroundColor(cardBorder), alignment: .top)
    }
}

struct ConversationZoneView: View {
    @ObservedObject var viewModel: HUDViewModel
    @State private var isAtBottom = true

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(viewModel.turns) { turn in
                        TurnRowView(turn: turn, streamingText: viewModel.streamingBuffer)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(turn.id)
                    }
                    // Sentinel: visible ↔ at bottom; drives isAtBottom state.
                    Color.clear.frame(height: 1)
                        .id("convBottom")
                        .onAppear { isAtBottom = true }
                        .onDisappear { isAtBottom = false }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 14)
            }
            // New turn: always snap to bottom and re-enable auto-scroll.
            .onChange(of: viewModel.turns.count) { _, _ in
                isAtBottom = true
                withAnimation { proxy.scrollTo("convBottom", anchor: .bottom) }
            }
            // Streaming tokens: follow only when pinned to bottom.
            .onChange(of: viewModel.streamingBuffer) { _, _ in
                guard isAtBottom else { return }
                proxy.scrollTo("convBottom", anchor: .bottom)
            }
        }
    }
}

struct CenterColumn: View {
    @ObservedObject var viewModel: HUDViewModel
    @ObservedObject var fullViewModel: FullDesktopViewModel
    var onCollapse: () -> Void
    var onTextCommand: (String) -> Void
    var onVoice: () -> Void

    private var hasConversation: Bool { !viewModel.turns.isEmpty }

    var body: some View {
        VStack(spacing: 0) {
            GeometryReader { geo in
                VStack(spacing: 0) {
                    OrbZoneView(
                        viewModel: viewModel,
                        fullViewModel: fullViewModel,
                        onCollapse: onCollapse
                    )
                    .frame(height: hasConversation ? geo.size.height * 0.45 : geo.size.height)
                    .animation(.easeInOut(duration: 0.3), value: hasConversation)

                    if hasConversation {
                        Divider().background(cardBorder)
                        ConversationZoneView(viewModel: viewModel)
                            .frame(height: geo.size.height * 0.55)
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                    }
                }
            }

            DesktopInputBar(
                viewModel: viewModel,
                onTextCommand: onTextCommand,
                onVoice: onVoice
            )
        }
        .background(bgPrimary)
    }
}

// MARK: - Nav sidebar + title bar

struct IconNavSidebar: View {
    var onSettings: () -> Void

    var body: some View {
        VStack(spacing: 6) {
            navIcon(systemName: "bubble.left.and.bubble.right", active: true, action: {})
            navIcon(systemName: "clock", active: false, action: {})
            Spacer()
            navIcon(systemName: "gearshape", active: false, action: onSettings)
        }
        .padding(.vertical, 14)
        .frame(width: 52)
        .background(bgSurface)
        .overlay(Rectangle().frame(width: 1).foregroundColor(cardBorder), alignment: .trailing)
    }

    private func navIcon(systemName: String, active: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 15))
                .foregroundColor(active ? accentCyan : textSecond)
                .frame(width: 36, height: 36)
                .background(active ? accentCyan.opacity(0.15) : Color.clear)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .shadow(color: active ? accentCyan.opacity(0.2) : .clear, radius: 6)
        }
        .buttonStyle(.plain)
    }
}

struct DesktopTitleBar: View {
    var body: some View {
        HStack {
            Text("JARVIS — THE INTELLIGENT ASSISTANT")
                .font(.system(size: 11, design: .monospaced))
                .kerning(1.5)
                .foregroundColor(textSecond)
            Spacer()
            HStack(spacing: 5) {
                Circle()
                    .fill(liveCyan)
                    .frame(width: 6, height: 6)
                    .shadow(color: liveCyan, radius: 3)
                Text("ONLINE")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(liveCyan)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(bgSurface)
        .overlay(Rectangle().frame(height: 1).foregroundColor(cardBorder), alignment: .bottom)
    }
}

// MARK: - Root view

struct FullDesktopView: View {
    @ObservedObject var viewModel: HUDViewModel
    @ObservedObject var fullViewModel: FullDesktopViewModel
    @ObservedObject var metricsProvider: SystemMetricsProvider
    var onCollapse: () -> Void = {}
    var onTextCommand: (String) -> Void = { _ in }
    var onVoice: () -> Void = {}
    var onSettings: () -> Void = {}

    var body: some View {
        VStack(spacing: 0) {
            DesktopTitleBar()

            HStack(spacing: 0) {
                IconNavSidebar(onSettings: onSettings)

                Divider().background(cardBorder)

                LeftPanel(viewModel: viewModel)

                Divider().background(cardBorder)

                CenterColumn(
                    viewModel: viewModel,
                    fullViewModel: fullViewModel,
                    onCollapse: onCollapse,
                    onTextCommand: onTextCommand,
                    onVoice: onVoice
                )

                Divider().background(cardBorder)

                RightPanel(fullViewModel: fullViewModel, metrics: metricsProvider)
            }
        }
        .background(bgPrimary)
        .preferredColorScheme(.dark)
    }
}
