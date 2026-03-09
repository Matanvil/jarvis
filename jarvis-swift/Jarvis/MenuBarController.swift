import AppKit

enum CoreStatus {
    case online, offline
}

final class MenuBarController {
    private let statusItem: NSStatusItem
    private var awayModeItem: NSMenuItem?

    init() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        setupButton()
        setupMenu()
        setStatus(.offline)
    }

    private func setupButton() {
        // Initial image set by setStatus(.offline) called in init
    }

    private func setupMenu() {
        let menu = NSMenu()
        let newConvo = NSMenuItem(
            title: "New Conversation",
            action: #selector(resetConversation),
            keyEquivalent: "n"
        )
        newConvo.target = self
        menu.addItem(newConvo)
        menu.addItem(.separator())
        let awayItem = NSMenuItem(
            title: "Away mode",
            action: #selector(toggleAway(_:)),
            keyEquivalent: ""
        )
        awayItem.state = .off
        awayItem.target = self
        self.awayModeItem = awayItem
        menu.addItem(awayItem)
        menu.addItem(NSMenuItem.separator())
        let quit = NSMenuItem(
            title: "Quit Jarvis",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        )
        menu.addItem(quit)
        statusItem.menu = menu
    }

    @objc private func toggleAway(_ sender: NSMenuItem) {
        let willBeAway = sender.state == .off  // toggling to new state
        let body = "{\"away\": \(willBeAway ? "true" : "false")}"
        guard let url = URL(string: "http://127.0.0.1:8765/telegram/away"),
              let data = body.data(using: .utf8) else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = data
        URLSession.shared.dataTask(with: req) { [weak self] _, _, _ in
            DispatchQueue.main.async {
                sender.state = willBeAway ? .on : .off
            }
        }.resume()
    }

    @objc private func resetConversation() {
        guard let url = URL(string: "http://127.0.0.1:8765/reset") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        URLSession.shared.dataTask(with: request).resume()
    }

    func setStatus(_ status: CoreStatus) {
        DispatchQueue.main.async { [weak self] in
            guard let button = self?.statusItem.button else { return }
            let color: NSColor = status == .online ? .systemGreen : .systemRed
            let config = NSImage.SymbolConfiguration(paletteColors: [color])
            let image = NSImage(systemSymbolName: "waveform", accessibilityDescription: "Jarvis")?
                .withSymbolConfiguration(config)
            button.image = image
        }
    }
}
