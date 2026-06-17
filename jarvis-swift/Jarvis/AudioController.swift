import AppKit
import AVFoundation
import Speech

private let _synthesizer = AVSpeechSynthesizer()

@MainActor
final class AudioController: NSObject, SFSpeechRecognizerDelegate {

    // MARK: - Hotkey config
    // Modifier-only hotkey: hold Control+Option to trigger.
    // Uses flagsChanged events (no key code needed).
    private let hotkeyModifiers: NSEvent.ModifierFlags = [.control, .option]   // ⌃⌥
    private var modifierComboActive = false

    // MARK: - Dependencies (injected)
    private let client: JarvisClient
    private let viewModel: HUDViewModel
    private let fullDesktopViewModel: FullDesktopViewModel
    private let showHUD: (HUDState) -> Void   // calls AppDelegate.showHUD (updates state + orderFront)
    private let hideHUD: () -> Void           // calls AppDelegate.hideHUD (sets .hidden + orderOut)

    // MARK: - State
    private var hotkeyMonitor: Any?
    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private var silenceTimer: Timer?
    private var streamTimer: Timer?
    private var pendingCompleteEvent: [String: Any]?
    private var isListening = false
    private var pendingToolUseId: String?
    private var pendingApprovalCategory: String?
    private var currentCommandId: String?
    private var lastCommandText: String?
    private var stepVoiceEnabled: Bool = false
    private var lastInputWasText = false

    // MARK: - Jarvis vocabulary for contextual biasing
    private static let jarvisVocabulary: [String] = [
        "jarvis", "ollama", "qwen", "haiku", "sonnet", "claude", "anthropic",
        "kubectl", "terraform", "homebrew", "xcodebuild",
        "pytest", "uvicorn", "fastapi",
        "shell run", "file write", "file read", "web search",
        "delegate to local", "delegate to claude code",
    ]

    // MARK: - Init

    init(
        client: JarvisClient,
        viewModel: HUDViewModel,
        fullDesktopViewModel: FullDesktopViewModel,
        showHUD: @escaping (HUDState) -> Void,
        hideHUD: @escaping () -> Void
    ) {
        self.client = client
        self.viewModel = viewModel
        self.fullDesktopViewModel = fullDesktopViewModel
        self.showHUD = showHUD
        self.hideHUD = hideHUD
        super.init()
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
        speechRecognizer?.delegate = self
    }

    // MARK: - Lifecycle

    func triggerVoiceInput() {
        if isListening {
            stopAndSend()
        } else {
            requestAuthAndListen()
        }
    }

    func start() {
        NSLog("[Jarvis] AudioController.start() — registering flagsChanged monitor")
        hotkeyMonitor = NSEvent.addGlobalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
            self?.handleFlagsChanged(event)
        }
        if hotkeyMonitor == nil {
            NSLog("[Jarvis] WARNING: addGlobalMonitorForEvents returned nil — missing Accessibility permission?")
        } else {
            NSLog("[Jarvis] Global flagsChanged monitor registered OK")
        }
    }

    func stop() {
        if let monitor = hotkeyMonitor {
            NSEvent.removeMonitor(monitor)
            hotkeyMonitor = nil
        }
        silenceTimer?.invalidate()
        silenceTimer = nil
        recognitionTask?.cancel()
        recognitionTask = nil
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        isListening = false
    }

    // MARK: - Hotkey

    private func handleFlagsChanged(_ event: NSEvent) {
        // Only consider ctrl and option — ignore shift/command/caps
        let relevant = event.modifierFlags.intersection([.control, .option, .shift, .command])
        let bothDown = relevant == hotkeyModifiers

        NSLog("[Jarvis] flagsChanged — relevant: %lu, bothDown: %d", relevant.rawValue, bothDown ? 1 : 0)

        guard bothDown != modifierComboActive else { return }
        modifierComboActive = bothDown

        if bothDown {
            NSLog("[Jarvis] Hotkey triggered (⌃⌥)")
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                if self.isListening {
                    self.stopAndSend()
                } else {
                    self.requestAuthAndListen()
                }
            }
        }
    }

    // MARK: - STT

    private func requestAuthAndListen() {
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            guard status == .authorized else {
                NSLog("[Jarvis] Speech recognition not authorized: %@", String(describing: status))
                return
            }
            DispatchQueue.main.async { self?.beginRecording() }
        }
    }

    private func beginRecording() {
        guard !audioEngine.isRunning else { return }

        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest else { return }
        recognitionRequest.shouldReportPartialResults = true
        recognitionRequest.contextualStrings = Self.jarvisVocabulary

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
        }

        do {
            audioEngine.prepare()
            try audioEngine.start()
        } catch {
            NSLog("[Jarvis] Audio engine failed: %@", error.localizedDescription)
            audioEngine.inputNode.removeTap(onBus: 0)
            return
        }

        isListening = true
        showHUD(.listening)
        scheduleSilenceTimeout()

        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            DispatchQueue.main.async {
                guard let self else { return }
                if let result {
                    if result.isFinal {
                        self.finalize(transcription: result.bestTranscription.formattedString)
                    } else {
                        // Still speaking — reset the silence timer
                        self.scheduleSilenceTimeout()
                    }
                } else if error != nil {
                    self.finalize(transcription: nil)
                }
            }
        }
    }

    private func stopAndSend() {
        stopRecording()
        // finalize() is invoked when the recognition task fires isFinal
    }

    private func stopRecording() {
        silenceTimer?.invalidate()
        silenceTimer = nil
        isListening = false
        // Stop audio hardware and signal end of input — do NOT cancel the task,
        // let it fire isFinal naturally so finalize() receives the transcription.
        guard audioEngine.isRunning else { return }
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionRequest = nil
    }

    // MARK: - Stream display timer (15 fps, drains tokenQueue → streamingBuffer)

    private func startStreamTimerIfNeeded() {
        guard streamTimer == nil else { return }
        streamTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 15.0, repeats: true) { [weak self] _ in
            // Timer fires on the main RunLoop (main thread). Use Task @MainActor to
            // access @MainActor-isolated properties with a static guarantee.
            Task { @MainActor [weak self] in
                guard let self else { return }
                // Flush the full queue immediately once complete is pending — no point
                // in throttling after the model has finished generating.
                let drainSize = self.pendingCompleteEvent != nil
                    ? self.viewModel.tokenQueue.count
                    : 20
                let hasMore = self.viewModel.drainTokenQueue(chars: drainSize)
                self.showHUD(.response(text: self.viewModel.streamingBuffer))
                if !hasMore {
                    if let event = self.pendingCompleteEvent {
                        self.pendingCompleteEvent = nil
                        self.stopStreamTimer()
                        self.viewModel.clearStreamingBuffer()
                        self.finalizeComplete(event: event)
                    } else {
                        self.stopStreamTimer()
                    }
                }
            }
        }
    }

    private func stopStreamTimer() {
        streamTimer?.invalidate()
        streamTimer = nil
    }

    private func scheduleSilenceTimeout() {
        silenceTimer?.invalidate()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: false) { [weak self] _ in
            self?.stopAndSend()
        }
    }

    // MARK: - Transcription routing

    private func finalize(transcription: String?) {
        guard let text = transcription, !text.isEmpty else {
            hideHUD()
            return
        }

        // If HUD is waiting for voice approval, route to approval classifier
        if case .approval = viewModel.state {
            routeToApprovalClassification(text: text)
            return
        }

        lastInputWasText = false
        sendCommand(text: text)
    }

    func submitTextCommand(text: String) {
        lastInputWasText = true
        sendCommand(text: text)
    }

    // MARK: - Command

    private func sendCommand(text: String) {
        lastCommandText = text
        viewModel.startTurn(command: text)
        showHUD(.thinking)
        Task { [weak self] in
            guard let self else { return }
            do {
                let commandId = try await client.startCommand(text: text, cwd: nil)
                await listenToEvents(commandId: commandId)
            } catch {
                NSLog("[Jarvis] startCommand failed: %@", error.localizedDescription)
                let msg = "I'm having trouble connecting. Please check that Jarvis is running."
                viewModel.finalizeTurn(response: msg)
                showHUD(.response(text: msg))
                speak(msg)
            }
        }
    }

    private func handleCommandResponse(_ response: CommandResponse) {
        if response.requiresApproval, let toolUseId = response.toolUseId {
            pendingToolUseId = toolUseId
            pendingApprovalCategory = response.approvalCategory
            showHUD(.approval(description: response.approvalDescription ?? response.text))
        } else {
            let fallback = "I'm experiencing an error. Please try again or restart Jarvis."
            let displayText = response.text.isEmpty ? fallback : response.text
            let speakText = response.speak ?? (response.text.isEmpty ? fallback : response.text)
            showHUD(.response(text: displayText))
            if !lastInputWasText { speak(speakText) }
        }
    }

    // MARK: - Approval

    func submitApproval(approved: Bool) {
        guard let toolUseId = pendingToolUseId else { return }
        let category = pendingApprovalCategory
        let originalCommand = lastCommandText
        let commandId = currentCommandId ?? ""
        pendingToolUseId = nil
        pendingApprovalCategory = nil

        if !approved {
            viewModel.finalizeTurn(response: "Denied.")
            showHUD(.denied)
            // Tell the server to drop the paused run; fire-and-forget.
            Task { _ = try? await client.sendApproval(commandId: commandId, toolUseId: toolUseId, approved: false, category: nil) }
            Task {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                await MainActor.run { self.hideHUD() }
            }
            return
        }

        viewModel.finalizeTurn(response: "Approved…")
        showHUD(.thinking)
        Task { [weak self] in
            guard let self else { return }
            let outcome = (try? await client.sendApproval(
                commandId: commandId, toolUseId: toolUseId, approved: true, category: category
            )) ?? .cancelled

            switch outcome {
            case .resumed(let response):
                // Server continued the run in place — render its final result (which may
                // itself be another approval prompt if a later step is also gated).
                await MainActor.run {
                    self.viewModel.finalizeTurn(response: response.text)
                    self.handleCommandResponse(response)
                }
            case .reissue:
                // No paused state (e.g. server restarted) — replay the original command.
                // lastInputWasText is inherited: a text-initiated command re-issues silently.
                guard let text = originalCommand else { await MainActor.run { self.hideHUD() }; return }
                await MainActor.run { self.viewModel.startTurn(command: text) }
                do {
                    let newId = try await client.startCommand(text: text, cwd: nil)
                    await self.listenToEvents(commandId: newId)
                } catch {
                    NSLog("[Jarvis] re-issue after approval failed: %@", error.localizedDescription)
                    let msg = "I'm having trouble connecting. Please check that Jarvis is running."
                    await MainActor.run {
                        self.viewModel.finalizeTurn(response: msg)
                        self.showHUD(.response(text: msg))
                        self.speak(msg)
                    }
                }
            case .cancelled:
                await MainActor.run { self.hideHUD() }
            }
        }
    }

    // MARK: - Config

    func refreshConfig() async {
        guard let url = URL(string: "http://127.0.0.1:8765/config") else { return }
        var request = URLRequest(url: url)
        ServerAuth.apply(to: &request)
        guard let (data, _) = try? await URLSession.shared.data(for: request),
              let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let narration = config["narration"] as? [String: Any]
        else { return }
        stepVoiceEnabled = narration["step_voice"] as? Bool ?? false
    }

    // MARK: - SSE

    private func listenToEvents(commandId: String) async {
        currentCommandId = commandId
        let stepVoice = stepVoiceEnabled
        guard let url = URL(string: "http://127.0.0.1:8765/events/\(commandId)") else { return }
        var request = URLRequest(url: url)
        request.timeoutInterval = 600
        ServerAuth.apply(to: &request)

        do {
            let (stream, _) = try await URLSession.shared.bytes(for: request)
            var buffer = ""
            for try await byte in stream {
                buffer.append(Character(UnicodeScalar(byte)))
                if buffer.hasSuffix("\n\n") {
                    let lines = buffer.components(separatedBy: "\n")
                    for line in lines {
                        if line.hasPrefix("data: ") {
                            let jsonStr = String(line.dropFirst(6))
                            if let data = jsonStr.data(using: .utf8),
                               let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                                handleSSEEvent(event, stepVoice: stepVoice)
                                await Task.yield()
                            }
                        }
                    }
                    buffer = ""
                }
            }
        } catch {
            NSLog("[Jarvis] SSE stream error for %@: %@", commandId, error.localizedDescription)
            let msg = "Lost connection to Jarvis."
            viewModel.finalizeTurn(response: msg)
            showHUD(.response(text: msg))
        }
    }

    @MainActor
    private func handleSSEEvent(_ event: [String: Any], stepVoice: Bool) {
        guard let type = event["type"] as? String else { return }
        switch type {
        case "step":
            let label = event["label"] as? String ?? "Working…"
            let milestone = event["milestone"] as? Bool ?? false
            viewModel.appendStep(Step(tool: label, inputSummary: nil, milestone: milestone))
            showHUD(.executing(step: label))
            if stepVoice, milestone {
                speak(label)
            }
            let toolName = event["tool"] as? String ?? label
            fullDesktopViewModel.recordToolUsed(toolName)
        case "clear":
            stopStreamTimer()
            viewModel.clearStreamingBuffer()
            showHUD(.executing(step: "Working…"))
        case "token":
            let token = event["text"] as? String ?? ""
            if !token.isEmpty {
                viewModel.appendToken(token)
                startStreamTimerIfNeeded()
            }
        case "complete":
            if streamTimer != nil {
                // Timer is still draining — defer finalization until queue is empty.
                pendingCompleteEvent = event
            } else {
                viewModel.clearStreamingBuffer()
                finalizeComplete(event: event)
            }
        case "error":
            // Cancel any pending stream state so it doesn't bleed into the next command.
            stopStreamTimer()
            pendingCompleteEvent = nil
            viewModel.clearStreamingBuffer()
            let msg = event["message"] as? String ?? "Something went wrong."
            viewModel.finalizeTurn(response: "Error: \(msg)")
            showHUD(.response(text: msg))
            speak(msg)
        case "compacted":
            let msg = event["message"] as? String ?? "Context compacted."
            showHUD(.response(text: msg))
            Task { @MainActor [weak self] in
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                guard let self else { return }
                self.showHUD(.hidden)
            }
        default:
            break
        }
    }

    private func finalizeComplete(event: [String: Any]) {
        let tokS        = event["tok_s"] as? Double ?? 0
        let ttftMs      = event["ttft_ms"] as? Int ?? 0
        let model       = event["model"] as? String ?? ""
        let intentClass = event["intent_class"] as? String ?? ""
        let genTokens   = event["gen_tokens"] as? Int ?? 0
        fullDesktopViewModel.updateMetrics(
            tokS: tokS,
            ttftMs: ttftMs,
            model: model,
            intentClass: intentClass,
            genTokens: genTokens
        )
        if let data = try? JSONSerialization.data(withJSONObject: event),
           let response = try? JSONDecoder().decode(CommandResponse.self, from: data) {
            viewModel.finalizeTurn(response: response.text)
            handleCommandResponse(response)
        } else {
            let text = event["display"] as? String ?? event["speak"] as? String ?? "Done."
            viewModel.finalizeTurn(response: text)
            showHUD(.response(text: text))
        }
    }

    private func routeToApprovalClassification(text: String) {
        Task { [weak self] in
            guard let self else { return }
            let result = try? await client.classifyApproval(text: text)
            await MainActor.run {
                if let approved = result {
                    self.submitApproval(approved: approved)
                }
                // nil → unclear, stay in .approval state, user must tap button
            }
        }
    }

    // MARK: - TTS

    private func speak(_ text: String) {
        guard !text.isEmpty else { return }
        _synthesizer.stopSpeaking(at: .immediate)
        let utterance = AVSpeechUtterance(string: text)
        // British English "Daniel" — natural, professional male voice.
        // Try premium first, fall back to compact, then any en-GB voice.
        utterance.voice =
            AVSpeechSynthesisVoice(identifier: "com.apple.ttsbundle.Daniel-premium") ??
            AVSpeechSynthesisVoice(identifier: "com.apple.voice.compact.en-GB.Daniel") ??
            AVSpeechSynthesisVoice(language: "en-GB")
        utterance.rate = 0.53
        utterance.pitchMultiplier = 1.0 // neutral pitch for Daniel's natural tone
        _synthesizer.speak(utterance)
    }
}
