import AppKit
import Foundation
import SwiftUI

class AppDelegate: NSObject, NSApplicationDelegate {

    private var pythonProcess: Process?
    private var healthTimer: Timer?
    private var hudWindow: HUDWindow?
    private var hudController: NSHostingController<HUDView>?
    private let hudViewModel = HUDViewModel.shared
    private var menuBarController: MenuBarController?
    private var jarvisClient: JarvisClient!
    private var audioController: AudioController!

    // MARK: - Lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        menuBarController = MenuBarController()
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
        audioController.start()
        installToApplicationsIfNeeded()
        checkFullDiskAccess()
    }

    func applicationWillTerminate(_ notification: Notification) {
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

    private func startPythonCore() {
        DispatchQueue.global(qos: .userInitiated).async {
            // Kill our tracked process if any
            if let pid = self.pythonProcess?.processIdentifier, pid > 0 {
                kill(pid, SIGKILL)
            }
            self.pythonProcess = nil

            // Kill any orphaned process still holding port 8765 (e.g. from a previous run)
            // waitUntilExit() is safe here — we're on a background queue
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
                    self.pythonProcess = process
                    NSLog("[Jarvis] Python core started (pid %d)", process.processIdentifier)
                }
            } catch {
                NSLog("[Jarvis] Failed to start Python core: %@", error.localizedDescription)
            }
        }
    }

    // MARK: - Health Poll

    private func scheduleHealthPoll() {
        healthTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            self?.checkHealth()
        }
    }

    private func checkHealth() {
        guard let url = URL(string: "http://127.0.0.1:8765/health") else { return }

        let request = URLRequest(url: url, timeoutInterval: 5)
        let task = URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            guard let self else { return }

            if error != nil || (response as? HTTPURLResponse)?.statusCode != 200 {
                self.menuBarController?.setStatus(.offline)
                DispatchQueue.global(qos: .utility).async {
                    NSLog("[Jarvis] Python core not responding — restarting")
                    self.startPythonCore()
                }
            } else {
                self.menuBarController?.setStatus(.online)
                self.menuBarController?.syncAwayState()
            }
        }
        task.resume()
    }

    // MARK: - Permissions

    private func installToApplicationsIfNeeded() {
        let appPath = Bundle.main.bundlePath
        guard !appPath.hasPrefix("/Applications/") else { return }

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
            self.hudViewModel.state = state
            self.hudWindow?.resize(toHeight: state.preferredHeight)
            self.hudWindow?.orderFront(nil)
        }
    }

    func hideHUD() {
        DispatchQueue.main.async {
            self.hudViewModel.state = .hidden
            self.hudWindow?.orderOut(nil)
        }
    }

    private func setupHUD() {
        let view = HUDView(
            viewModel: hudViewModel,
            onDismiss: { [weak self] in self?.hideHUD() },
            onApprove: { [weak self] in self?.handleApprove() },
            onDeny: { [weak self] in self?.handleDeny() }
        )
        let window = HUDWindow(viewModel: hudViewModel)
        let controller = NSHostingController(rootView: view)
        // Prevent NSHostingController from resizing the window to match SwiftUI's intrinsic
        // content size (which collapses to ~0 when state=.hidden). Width must be controlled
        // entirely by HUDWindow.resize(toHeight:).
        if #available(macOS 13.0, *) {
            controller.sizingOptions = []
        }
        window.contentViewController = controller
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
