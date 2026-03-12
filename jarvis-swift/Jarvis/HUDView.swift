import SwiftUI

// MARK: - Height reporting

private struct ThreadHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

// MARK: - HUDView

struct HUDView: View {
    @ObservedObject var viewModel: HUDViewModel
    var onDismiss:  () -> Void = {}
    var onMinimize: () -> Void = {}
    var onExpand:   () -> Void = {}
    var onApprove:  () -> Void = {}
    var onDeny:     () -> Void = {}

    @State private var isHovered = false

    private var maxThreadHeight: CGFloat {
        (NSScreen.main?.visibleFrame.height ?? 900) * 0.6 - 36 - 16
        // 36 = status bar, 16 = top/bottom padding
    }

    var body: some View {
        Group {
            if viewModel.state == .minimized {
                ArcReactorView()
                    .onTapGesture(perform: onExpand)
            } else if viewModel.state != .hidden {
                ZStack(alignment: .topTrailing) {
                    VStack(spacing: 0) {
                        threadView
                        statusBar
                    }
                    .frame(maxWidth: .infinity)
                    .background(
                        RoundedRectangle(cornerRadius: 20)
                            .fill(.ultraThinMaterial)
                    )

                    // Hover-reveal minimize/close buttons
                    HStack(spacing: 6) {
                        Button(action: onMinimize) {
                            Image(systemName: "minus.circle.fill")
                                .font(.system(size: 16))
                                .foregroundStyle(
                                    Color(red: 0.22, green: 0.74, blue: 0.97).opacity(0.85)
                                )
                        }
                        .buttonStyle(.plain)

                        Button(action: onDismiss) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 16))
                                .foregroundStyle(.white.opacity(0.6))
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(10)
                    .opacity(isHovered ? 1 : 0)
                    .animation(.easeInOut(duration: 0.15), value: isHovered)
                }
                .onHover { isHovered = $0 }
                .padding(8)
            }
        }
        .animation(.easeInOut(duration: 0.2), value: viewModel.state == .hidden)
    }

    // MARK: - Thread

    @ViewBuilder
    private var threadView: some View {
        if viewModel.turns.isEmpty {
            EmptyView()
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(viewModel.turns) { turn in
                            TurnRowView(turn: turn)
                        }
                        // Invisible anchor for auto-scroll
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 12)
                    .padding(.bottom, 4)
                    .background(
                        GeometryReader { geo in
                            Color.clear.preference(key: ThreadHeightKey.self, value: geo.size.height)
                        }
                    )
                }
                .frame(maxHeight: maxThreadHeight)
                .onChange(of: viewModel.turns.count) { _ in
                    withAnimation { proxy.scrollTo("bottom", anchor: .bottom) }
                }
                .onChange(of: viewModel.turns.last?.steps.count) { _ in
                    // Suppress auto-scroll during approval so buttons stay visible
                    if case .approval = viewModel.state { return }
                    withAnimation { proxy.scrollTo("bottom", anchor: .bottom) }
                }
            }
            .onPreferenceChange(ThreadHeightKey.self) { height in
                let cap = (NSScreen.main?.visibleFrame.height ?? 900) * 0.6
                let newHeight = min(height + 36 + 16, cap)   // thread + status bar + padding
                // PreferenceKey callbacks may arrive off-main; guard before writing @Published.
                DispatchQueue.main.async {
                    viewModel.contentHeight = max(newHeight, 120)
                }
            }
        }
    }

    // MARK: - Status bar

    @ViewBuilder
    private var statusBar: some View {
        HStack(spacing: 8) {
            statusBarContent
            Spacer()
            if viewModel.completedTurnCount > 0 {
                Text("\(viewModel.completedTurnCount) turn\(viewModel.completedTurnCount == 1 ? "" : "s")")
                    .font(.system(size: 10, weight: .regular))
                    .foregroundStyle(.white.opacity(0.3))
            }
        }
        .padding(.horizontal, 16)
        .frame(height: 36)
        .background(Color.white.opacity(0.04))
    }

    @ViewBuilder
    private var statusBarContent: some View {
        switch viewModel.state {
        case .hidden, .minimized:
            EmptyView()

        case .listening:
            Label("Listening…", systemImage: "waveform")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.white.opacity(0.7))

        case .thinking:
            Label("Thinking", systemImage: "ellipsis.bubble")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.white.opacity(0.7))

        case .executing(let step):
            HStack(spacing: 6) {
                Image(systemName: "gearshape")
                    .font(.system(size: 11))
                    .foregroundStyle(Color(red: 1.0, green: 0.65, blue: 0.2))
                Text(step)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Color(red: 1.0, green: 0.65, blue: 0.2))
            }

        case .response:
            Label("Done", systemImage: "checkmark")
                .font(.system(size: 12))
                .foregroundStyle(.white.opacity(0.4))

        case .approval(let description):
            HStack(spacing: 10) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(.orange)
                Text(description)
                    .font(.system(size: 12))
                    .foregroundStyle(.white.opacity(0.85))
                    .lineLimit(1)
                Spacer()
                Button("Deny", action: onDeny)
                    .buttonStyle(HUDActionButtonStyle(tint: .red))
                Button("Allow", action: onApprove)
                    .buttonStyle(HUDActionButtonStyle(tint: .green))
            }

        case .approved:
            Label("Approved", systemImage: "checkmark.circle.fill")
                .font(.system(size: 12))
                .foregroundStyle(.green.opacity(0.8))

        case .denied:
            Label("Denied", systemImage: "xmark.circle.fill")
                .font(.system(size: 12))
                .foregroundStyle(.red.opacity(0.8))
        }
    }
}

// MARK: - TurnRowView

private struct TurnRowView: View {
    let turn: ConversationTurn

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            // User command
            HStack(spacing: 6) {
                Text("▶")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.3))
                Text(turn.command)
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(Color(red: 0.33, green: 0.67, blue: 1.0))
            }

            // Tool steps
            ForEach(Array(turn.steps.enumerated()), id: \.offset) { _, step in
                HStack(spacing: 6) {
                    Text("⚙")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.2))
                    Text(stepLabel(step))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.3))
                        .lineLimit(1)
                }
                .padding(.leading, 16)
            }

            // Response (only when complete)
            if let response = turn.response {
                Text(response)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(Color(red: 0.47, green: 0.8, blue: 0.47))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.leading, 16)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func stepLabel(_ step: Step) -> String {
        // SSE step events carry only `label` (human-readable, e.g. "Running shell command").
        // `tool` is set to the label string; `inputSummary` is nil. Display it as-is.
        let label = step.tool
        return label.count > 70 ? String(label.prefix(70)) + "…" : label
    }
}

// MARK: - Supporting Views (unchanged)

struct HUDStatusRow: View {
    let icon: String
    let label: String
    let spinning: Bool

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(.white)
            Text(label)
                .foregroundStyle(.white)
                .font(.system(size: 15, weight: .medium))
            if spinning {
                ProgressView()
                    .progressViewStyle(.circular)
                    .scaleEffect(0.75)
                    .tint(.white)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 18)
        .padding(.horizontal, 20)
    }
}

struct HUDActionButtonStyle: ButtonStyle {
    let tint: Color
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(.white)
            .font(.system(size: 12, weight: .semibold))
            .padding(.horizontal, 14)
            .padding(.vertical, 5)
            .background(RoundedRectangle(cornerRadius: 8).fill(tint.opacity(0.8)))
            .opacity(configuration.isPressed ? 0.7 : 1)
    }
}
