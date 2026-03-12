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
    }

    required init?(coder: NSCoder) { fatalError() }

    func open() {
        // Swap in a fresh SettingsView so the ViewModel reloads config on every open.
        let view = SettingsView(onDismiss: { [weak self] in self?.close() })
        window?.contentView = NSHostingView(rootView: view)
        showWindow(nil)
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}
