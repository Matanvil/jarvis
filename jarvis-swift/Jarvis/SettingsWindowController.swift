import AppKit
import SwiftUI

final class SettingsWindowController: NSWindowController {

    static let shared = SettingsWindowController()

    private init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 700, height: 480),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.title = "Jarvis Settings"
        window.center()
        window.isReleasedWhenClosed = false
        super.init(window: window)

        let view = SettingsView(onDismiss: { [weak self] in
            self?.close()
        })
        window.contentView = NSHostingView(rootView: view)
    }

    required init?(coder: NSCoder) { fatalError() }

    func open() {
        showWindow(nil)
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}
