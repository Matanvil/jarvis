import AppKit
import SwiftUI

final class FullDesktopWindow: NSWindow {

    init(
        viewModel: HUDViewModel,
        fullViewModel: FullDesktopViewModel,
        metricsProvider: SystemMetricsProvider,
        onCollapse: @escaping () -> Void,
        onTextCommand: @escaping (String) -> Void,
        onVoice: @escaping () -> Void,
        onSettings: @escaping () -> Void,
        onApprove: @escaping () -> Void = {},
        onDeny: @escaping () -> Void = {}
    ) {
        let screen = NSScreen.main?.visibleFrame ?? CGRect(x: 0, y: 0, width: 1440, height: 900)
        let width  = min(screen.width * 0.9, 1440)
        let height = min(screen.height * 0.9, 900)
        let origin = CGPoint(
            x: screen.origin.x + (screen.width - width) / 2,
            y: screen.origin.y + (screen.height - height) / 2
        )
        super.init(
            contentRect: CGRect(origin: origin, size: CGSize(width: width, height: height)),
            styleMask: [.titled, .closable, .resizable, .miniaturizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        titlebarAppearsTransparent = true
        titleVisibility = .hidden
        isReleasedWhenClosed = false
        hidesOnDeactivate = false
        minSize = CGSize(width: 1024, height: 700)
        backgroundColor = NSColor(red: 0.039, green: 0.059, blue: 0.098, alpha: 1)
        collectionBehavior = [.managed, .participatesInCycle]

        let rootView = FullDesktopView(
            viewModel: viewModel,
            fullViewModel: fullViewModel,
            metricsProvider: metricsProvider,
            onCollapse: onCollapse,
            onTextCommand: onTextCommand,
            onVoice: onVoice,
            onSettings: onSettings,
            onApprove: onApprove,
            onDeny: onDeny
        )
        contentView = NSHostingView(rootView: rootView)
    }
}
