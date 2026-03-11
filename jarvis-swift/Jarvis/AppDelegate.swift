import AppKit
import Foundation
import SwiftUI

class AppDelegate: NSObject, NSApplicationDelegate {

    private var pythonProcess: Process?
    private var isRestarting = false
    private var isTerminating = false
    private var lastRestartTime: Date = .distantPast
    private var healthTimer: Timer?
    private var hudWindow: HUDWindow?
    private var hudController: NSHostingController<HUDView>?
    private let hudViewModel = HUDViewModel.shared
    private var lastVisibleState: HUDState = .hidden
    private var menuBarController: MenuBarController?
    private var jarvisClient: JarvisClient!
    private var audioController: AudioController!

    // MARK: - Lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        menuBarController = MenuBarController(onRestart: { [weak self] in
            self?.startPythonCore()
        })
        jarvisClient = JarvisClient()
        audioController = AudioController(
            client: jarvisClient,
            viewModel: hudViewModel,
            showHUD: { [weak self] state in self?.showHUD(state) },
            hideHUD: { [weak self] in self?.hideHUD() }
        )
        startPythonCore()
        scheduleHealthPoll()
        setupHUD()
        minimizeHUD()
        audioController.start()
        installToApplicationsIfNeeded()
        checkFullDiskAccess()
    }

    func applicationWillTerminate(_ notification: Notification) {
        isTerminating = true
        healthTimer?.invalidate()
        if let proc = pythonProcess, proc.processIdentifier > 0 {
            proc.terminate()  // SIGTERM — give uvicorn a chance to flush
            let done = DispatchSemaphore(value: 0)
            DispatchQueue.global().async { proc.waitUntilExit(); done.signal() }
            if done.wait(timeout: .now() + 1.5) == .timedOut {
                kill(proc.processIdentifier, SIGKILL)
            }
        }
        pythonProcess = nil
    }

    // MARK: - Python Core

    private func resolveCoreDirectory() -> URL {
        // Walk up from the executable to find the repo root (contains jarvis-core/).
        var dir = Bundle.main.executableURL?.deletingLastPathComponent()
        for _ in 0..<10 {
            guard let current = dir else { break }
            let candidate = current.appendingPathComponent("jarvis-core")
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
            dir = current.deletingLastPathComponent()
        }
        // Hardcoded dev fallback.
        let fallback = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("dev/jarvis/jarvis-core")
        return fallback
    }

    private func killOrphanedServer() {
        let cleanup = Process()
        cleanup.executableURL = URL(fileURLWithPath: "/bin/sh")
        cleanup.arguments = ["-c", "lsof -ti :8765 | xargs kill -9 2>/dev/null || true"]
        try? cleanup.run()
        cleanup.waitUntilExit()
    }

    func startPythonCore() {
        // Always enter on main thread so isRestarting and pythonProcess are accessed safely.
        DispatchQueue.main.async { [weak self] in
            guard let self, !self.isRestarting else {
                NSLog("[Jarvis] startPythonCore called while already restarting — skipping")
                return
            }
            self.isRestarting = true
            self.lastRestartTime = Date()

            // Immediately reflect the offline state in the UI.
            self.menuBarController?.setStatus(.offline)

            // Kill tracked process now, on main thread, before handing off to background.
            let dyingProcess = self.pythonProcess
            if let pid = dyingProcess?.processIdentifier, pid > 0 {
                kill(pid, SIGKILL)
            }
            self.pythonProcess = nil

            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self else { return }
                defer { DispatchQueue.main.async { self.isRestarting = false } }

                // Wait for the tracked process to fully exit before sweeping the port.
                dyingProcess?.waitUntilExit()
                // Kill any orphaned process still holding port 8765.
                self.killOrphanedServer()

                let coreDir = self.resolveCoreDirectory()
                let venvPython = coreDir.appendingPathComponent(".venv/bin/python")

                guard FileManager.default.fileExists(atPath: venvPython.path) else {
                    NSLog("[Jarvis] Python venv not found at %@", venvPython.path)
                    return
                }

                let process = Process()
                process.executableURL = venvPython
                process.arguments = [
                    "-m", "uvicorn", "server:app",
                    "--host", "127.0.0.1",
                    "--port", "8765",
                    "--log-level", "warning",
                ]
                process.currentDirectoryURL = coreDir

                do {
                    try process.run()
                    DispatchQueue.main.async {
                        guard !self.isTerminating else {
                            kill(process.processIdentifier, SIGKILL)
                            return
                        }
                        self.pythonProcess = process
                        NSLog("[Jarvis] Python core started (pid %d)", process.processIdentifier)
                    }
                } catch {
                    NSLog("[Jarvis] Failed to start Python core: %@", error.localizedDescription)
                }
            }
        }
    }

    // MARK: - Health Poll

    private func scheduleHealthPoll() {
        let timer = Timer(timeInterval: 5.0, repeats: true) { [weak self] _ in
            self?.checkHealth()
        }
        // Add to .common so the timer keeps firing while modal sheets are open.
        RunLoop.main.add(timer, forMode: .common)
        healthTimer = timer
    }

    private func checkHealth() {
        guard let url = URL(string: "http://127.0.0.1:8765/health") else { return }

        let request = URLRequest(url: url, timeoutInterval: 5)
        let task = URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            guard let self else { return }

            if error != nil || (response as? HTTPURLResponse)?.statusCode != 200 {
                self.menuBarController?.setStatus(.offline)
                // Cooldown check and restart must both happen on the main thread so
                // lastRestartTime is read safely (it is only written on main).
                DispatchQueue.main.async { [weak self] in
                    guard let self,
                          Date().timeIntervalSince(self.lastRestartTime) > 15 else { return }
                    NSLog("[Jarvis] Python core not responding — restarting")
                    self.startPythonCore()
                }
            } else {
                self.menuBarController?.setStatus(.online)
                self.menuBarController?.syncAwayState()
                Task { await self.audioController.refreshConfig() }
            }
        }
        task.resume()
    }

    // MARK: - Permissions

    private static let knownInstallPaths = [
        "/Applications/Jarvis.app",
        NSHomeDirectory() + "/Applications/Jarvis.app",
    ]

    private func installToApplicationsIfNeeded() {
        let appPath = Bundle.main.bundlePath
        let fm = FileManager.default
        // Skip if already running from a standard Applications folder, or if a copy is already
        // installed there (e.g. running a dev build while the app is installed in /Applications).
        guard !Self.knownInstallPaths.contains(where: { appPath.hasPrefix($0) }),
              !Self.knownInstallPaths.contains(where: { fm.fileExists(atPath: $0) }) else { return }

        let dest = "/Applications/Jarvis.app"
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            let alert = NSAlert()
            alert.messageText = "Install Jarvis to Applications?"
            alert.informativeText = "Jarvis is running from a temporary location. Installing it to /Applications makes it easier to grant permissions and find in Spotlight."
            alert.addButton(withTitle: "Install")
            alert.addButton(withTitle: "Not Now")
            guard alert.runModal() == .alertFirstButtonReturn else { return }

            let fm = FileManager.default
            do {
                if fm.fileExists(atPath: dest) {
                    try fm.removeItem(atPath: dest)
                }
                try fm.copyItem(atPath: appPath, toPath: dest)
                let relaunchAlert = NSAlert()
                relaunchAlert.messageText = "Installed!"
                relaunchAlert.informativeText = "Jarvis has been copied to /Applications. Relaunch from there to apply permissions."
                relaunchAlert.addButton(withTitle: "Relaunch")
                relaunchAlert.addButton(withTitle: "Later")
                if relaunchAlert.runModal() == .alertFirstButtonReturn {
                    NSWorkspace.shared.openApplication(
                        at: URL(fileURLWithPath: dest),
                        configuration: NSWorkspace.OpenConfiguration()
                    )
                    NSApp.terminate(nil)
                }
            } catch {
                NSLog("[Jarvis] Failed to install to /Applications: %@", error.localizedDescription)
                let errAlert = NSAlert()
                errAlert.messageText = "Installation Failed"
                errAlert.informativeText = error.localizedDescription
                errAlert.runModal()
            }
        }
    }

    private func checkFullDiskAccess() {
        // Full Disk Access is required for Jarvis to read/write files anywhere on disk.
        // Only relevant for production installs — dev builds in DerivedData never have FDA
        // so the probe would always fail, causing the dialog to appear on every Xcode build.
        guard Self.knownInstallPaths.contains(where: { Bundle.main.bundlePath.hasPrefix($0) }) else { return }
        // Probe a TCC-protected path — readable only with FDA granted.
        let probe = "/Library/Application Support/com.apple.TCC/TCC.db"
        guard !FileManager.default.isReadableFile(atPath: probe) else { return }

        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            let alert = NSAlert()
            alert.messageText = "Full Disk Access Required"
            alert.informativeText = "Jarvis needs Full Disk Access to read and write files anywhere on your Mac.\n\n1. Click \"Open Settings\"\n2. Click \"+\" and add Jarvis from the Applications folder\n3. Restart Jarvis"
            alert.alertStyle = .warning
            alert.addButton(withTitle: "Open Settings")
            alert.addButton(withTitle: "Later")
            if alert.runModal() == .alertFirstButtonReturn {
                if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles") {
                    NSWorkspace.shared.open(url)
                }
            }
        }
    }

    // MARK: - HUD

    func showHUD(_ state: HUDState) {
        DispatchQueue.main.async {
            self.lastVisibleState = state
            self.hudViewModel.state = state
            self.hudWindow?.resizeForExpanded(toHeight: state.preferredHeight)
            self.hudWindow?.orderFront(nil)
        }
    }

    func hideHUD() {
        DispatchQueue.main.async {
            self.hudViewModel.state = .hidden
            self.hudWindow?.orderOut(nil)
        }
    }

    func minimizeHUD() {
        DispatchQueue.main.async {
            // Remember what we're minimizing from (so expand can restore it).
            // If lastVisibleState is .hidden (fresh launch, icon saved from prior session),
            // expand() will no-op — which is safe since there's nothing to restore.
            if self.hudViewModel.state != .minimized {
                self.lastVisibleState = self.hudViewModel.state
            }
            self.hudViewModel.state = .minimized
            self.hudWindow?.resizeForMinimized()
            self.hudWindow?.orderFront(nil)
        }
    }

    func expandHUD() {
        DispatchQueue.main.async {
            guard self.lastVisibleState != .hidden else { return }
            let state = self.lastVisibleState
            self.hudViewModel.state = state
            self.hudWindow?.resizeForExpanded(toHeight: state.preferredHeight)
            self.hudWindow?.orderFront(nil)
        }
    }

    private func setupHUD() {
        let view = HUDView(
            viewModel: hudViewModel,
            onDismiss:  { [weak self] in self?.hideHUD() },
            onMinimize: { [weak self] in self?.minimizeHUD() },
            onExpand:   { [weak self] in self?.expandHUD() },
            onApprove:  { [weak self] in self?.handleApprove() },
            onDeny:     { [weak self] in self?.handleDeny() }
        )
        let window = HUDWindow(viewModel: hudViewModel)
        let controller = NSHostingController(rootView: view)
        // Prevent NSHostingController from resizing the window to match SwiftUI's intrinsic
        // content size (which collapses to ~0 when state=.hidden). Width must be controlled
        // entirely by HUDWindow.resizeForExpanded(toHeight:).
        if #available(macOS 13.0, *) {
            controller.sizingOptions = []
        }
        window.contentViewController = controller
        // NSHostingView has a default opaque background — clear it so the window is fully transparent.
        controller.view.wantsLayer = true
        controller.view.layer?.backgroundColor = .clear
        hudController = controller
        hudWindow = window
    }

    private func handleApprove() {
        Task { @MainActor in audioController.submitApproval(approved: true) }
    }

    private func handleDeny() {
        Task { @MainActor in audioController.submitApproval(approved: false) }
    }
}
