import AppKit
import SwiftUI

class HUDWindow: NSPanel {

    private let screenFrame: NSRect

    init(viewModel: HUDViewModel) {
        screenFrame = NSScreen.main?.visibleFrame
            ?? NSScreen.screens.first?.visibleFrame
            ?? NSRect(x: 0, y: 0, width: 1280, height: 800)

        let width: CGFloat = min(max(screenFrame.width * 0.5, 480), 900)
        let x = screenFrame.midX - width / 2
        let y = screenFrame.minY + 80

        super.init(
            contentRect: NSRect(x: x, y: y, width: width, height: 60),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        self.level = .floating
        self.isOpaque = false
        self.backgroundColor = .clear
        self.hasShadow = true
        self.isMovableByWindowBackground = true
        self.isReleasedWhenClosed = false
        self.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
    }

    private var canonicalWidth: CGFloat {
        min(max(screenFrame.width * 0.5, 480), 900)
    }

    /// Resize the window to the given content height, keeping it bottom-centre on screen.
    func resize(toHeight height: CGFloat) {
        // Always recompute width from screenFrame — do NOT use frame.width, which may have been
        // shrunk to near-zero by NSHostingController when the SwiftUI view rendered as hidden (0×0).
        let width = canonicalWidth
        let x = screenFrame.midX - width / 2
        let y = screenFrame.minY + 80
        setFrame(NSRect(x: x, y: y, width: width, height: height), display: true, animate: true)
    }
}
