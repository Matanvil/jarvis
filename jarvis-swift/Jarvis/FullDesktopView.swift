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
            VStack(spacing: 6) {
                MetricRow(label: "tok/s",    value: String(format: "%.0f", fullViewModel.tokS))
                MetricRow(label: "ttft",     value: "\(fullViewModel.ttftMs)ms")
                MetricRow(label: "executor", value: fullViewModel.executorModel)
                MetricRow(label: "intent",   value: fullViewModel.intentClass)
                MetricRow(label: "tokens",   value: "\(fullViewModel.totalTokens)")
                MetricRow(label: "turns",    value: "\(fullViewModel.turnCount)")
            }
            .padding(12)
        }
    }
}

struct LiveLogCard: View {
    @ObservedObject var fullViewModel: FullDesktopViewModel

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
            }
            .padding(12)
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
                    statPill(value: String(format: "%.0f", fullViewModel.tokS), label: "TOK/S")
                    statPill(value: "\(fullViewModel.ttftMs)ms", label: "TTFT")
                    statPill(value: fullViewModel.executorModel, label: "MODEL")
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

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(viewModel.turns) { turn in
                        TurnRowView(turn: turn, streamingText: viewModel.streamingBuffer)
                            .id(turn.id)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 14)
            }
            .onChange(of: viewModel.turns.count) { _ in
                if let last = viewModel.turns.last {
                    withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                }
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
