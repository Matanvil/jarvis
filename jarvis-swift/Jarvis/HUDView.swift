import SwiftUI

struct HUDView: View {
    @ObservedObject var viewModel: HUDViewModel
    var onDismiss:  () -> Void = {}
    var onMinimize: () -> Void = {}
    var onExpand:   () -> Void = {}
    var onApprove:  () -> Void = {}
    var onDeny:     () -> Void = {}

    @State private var isHovered = false

    var body: some View {
        Group {
            if viewModel.state == .minimized {
                // Arc reactor icon — tap to expand
                ArcReactorView()
                    .onTapGesture(perform: onExpand)
            } else if viewModel.state != .hidden {
                ZStack(alignment: .topTrailing) {
                    contentView
                        .frame(maxWidth: .infinity)
                        .background(
                            RoundedRectangle(cornerRadius: 20)
                                .fill(.ultraThinMaterial)
                        )

                    // Hover-reveal button row: minimize (−) then close (✕)
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

    @ViewBuilder
    private var contentView: some View {
        switch viewModel.state {
        case .hidden, .minimized:
            EmptyView()

        case .listening:
            HUDStatusRow(icon: "waveform", label: "Listening…", spinning: true)

        case .thinking:
            HUDStatusRow(icon: "ellipsis.bubble", label: "Thinking…", spinning: true)

        case .executing(let step):
            HUDStatusRow(icon: "gearshape", label: step, spinning: true)

        case .response(let text):
            ScrollView {
                Text(text)
                    .foregroundStyle(.white)
                    .font(.system(size: 14))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
            }
            .frame(maxHeight: 280)

        case .approval(let description):
            VStack(spacing: 12) {
                Text(description)
                    .foregroundStyle(.white)
                    .font(.system(size: 14))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 20)
                    .padding(.top, 16)

                HStack(spacing: 16) {
                    Button("Deny", action: onDeny)
                        .buttonStyle(HUDActionButtonStyle(tint: .red))
                    Button("Allow", action: onApprove)
                        .buttonStyle(HUDActionButtonStyle(tint: .green))
                }
                .padding(.bottom, 16)
            }

        case .approved:
            HUDStatusRow(icon: "checkmark.circle.fill", label: "Approved", spinning: false)

        case .denied:
            HUDStatusRow(icon: "xmark.circle.fill", label: "Denied", spinning: false)
        }
    }
}

// MARK: - Supporting Views

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
        .padding(.vertical, 18)
        .padding(.horizontal, 20)
    }
}

struct HUDActionButtonStyle: ButtonStyle {
    let tint: Color
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(.white)
            .font(.system(size: 14, weight: .semibold))
            .padding(.horizontal, 28)
            .padding(.vertical, 10)
            .background(RoundedRectangle(cornerRadius: 10).fill(tint.opacity(0.8)))
            .opacity(configuration.isPressed ? 0.7 : 1)
    }
}
