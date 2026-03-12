import AppKit

enum CoreStatus {
    case online, offline
}

final class MenuBarController {
    private let statusItem: NSStatusItem
    private var awayModeItem: NSMenuItem?
    private let onRestart: () -> Void
    private let onSettings: () -> Void
    private let onNewConversation: () -> Void

    init(
        onRestart: @escaping () -> Void,
        onSettings: @escaping () -> Void,
        onNewConversation: @escaping () -> Void
    ) {
        self.onRestart = onRestart
        self.onSettings = onSettings
        self.onNewConversation = onNewConversation
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
        let settings = NSMenuItem(title: "Settings…", action: #selector(openSettings), keyEquivalent: ",")
        settings.target = self
        menu.addItem(settings)
        menu.addItem(NSMenuItem.separator())
        let newConvo = NSMenuItem(
            title: "New Conversation",
            action: #selector(resetConversation),
            keyEquivalent: "n"
        )
        newConvo.target = self
        menu.addItem(newConvo)
        let restart = NSMenuItem(title: "Restart Server", action: #selector(restartServer), keyEquivalent: "")
        restart.target = self
        menu.addItem(restart)
        menu.addItem(NSMenuItem.separator())
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

    @objc private func openSettings() {
        onSettings()
    }

    @objc private func restartServer() {
        onRestart()
    }

    @objc private func toggleAway(_ sender: NSMenuItem) {
        let willBeAway = sender.state == .off
        let payload: [String: Bool] = ["away": willBeAway]
        guard let url = URL(string: "http://127.0.0.1:8765/telegram/away"),
              let body = try? JSONSerialization.data(withJSONObject: payload) else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        URLSession.shared.dataTask(with: req) { _, response, error in
            guard error == nil,
                  let http = response as? HTTPURLResponse,
                  (200..<300).contains(http.statusCode) else { return }
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
        // Clear the Swift-side conversation thread and save the session
        DispatchQueue.main.async { [weak self] in
            self?.onNewConversation()
        }
    }

    func syncAwayState() {
        guard let url = URL(string: "http://127.0.0.1:8765/telegram/away") else { return }
        URLSession.shared.dataTask(with: url) { [weak self] data, _, _ in
            guard let data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Bool],
                  let away = json["away"] else { return }
            DispatchQueue.main.async {
                self?.awayModeItem?.state = away ? .on : .off
            }
        }.resume()
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
