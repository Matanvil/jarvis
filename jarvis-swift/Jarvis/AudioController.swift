import AppKit
import AVFoundation
import Speech

private let _synthesizer = AVSpeechSynthesizer()

@MainActor
final class AudioController: NSObject, SFSpeechRecognizerDelegate, AVAudioPlayerDelegate {

    // MARK: - Hotkey config
    // Modifier-only hotkey: hold Control+Option to trigger.
    // Uses flagsChanged events (no key code needed).
    private let hotkeyModifiers: NSEvent.ModifierFlags = [.control, .option]   // ⌃⌥
    private var modifierComboActive = false

    // MARK: - Dependencies (injected)
    private let client: JarvisClient
    private let viewModel: HUDViewModel
    private let showHUD: (HUDState) -> Void   // calls AppDelegate.showHUD (updates state + orderFront)
    private let hideHUD: () -> Void           // calls AppDelegate.hideHUD (sets .hidden + orderOut)

    // MARK: - State
    private var hotkeyMonitor: Any?
    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private var silenceTimer: Timer?
    private var isListening = false
    private var pendingToolUseId: String?
    private var pendingApprovalCategory: String?
    private var lastCommandText: String?
    private var stepVoiceEnabled: Bool = false

    // MARK: - Piper TTS state
    private var _audioPlayer: AVAudioPlayer?       // strong ref prevents dealloc during playback
    private var _ttsTask: URLSessionDataTask?       // current in-flight /tts request
    private var _currentTempURL: URL?              // temp WAV file; deleted after playback

    // MARK: - Init

    init(
        client: JarvisClient,
        viewModel: HUDViewModel,
        showHUD: @escaping (HUDState) -> Void,
        hideHUD: @escaping () -> Void
    ) {
        self.client = client
        self.viewModel = viewModel
        self.showHUD = showHUD
        self.hideHUD = hideHUD
        super.init()
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
        speechRecognizer?.delegate = self
    }

    // MARK: - Lifecycle

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
            speak(speakText)
        }
    }

    // MARK: - Approval

    func submitApproval(approved: Bool) {
        guard let toolUseId = pendingToolUseId else { return }
        let category = pendingApprovalCategory
        let originalCommand = lastCommandText
        pendingToolUseId = nil
        pendingApprovalCategory = nil

        if !approved {
            viewModel.finalizeTurn(response: "Denied.")
            showHUD(.denied)
            Task {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                await MainActor.run { self.hideHUD() }
            }
            return
        }

        viewModel.finalizeTurn(response: "Approved — re-running…")
        showHUD(.thinking)
        Task { [weak self] in
            guard let self else { return }
            let shouldReissue = (try? await client.sendApproval(
                toolUseId: toolUseId, approved: true, category: category
            )) ?? false

            if shouldReissue, let text = originalCommand {
                // Re-issue the original command — guardrails now trust the category for this session.
                // Create a fresh turn for the re-issued command.
                await MainActor.run { self.viewModel.startTurn(command: text) }
                do {
                    let commandId = try await client.startCommand(text: text, cwd: nil)
                    await self.listenToEvents(commandId: commandId)
                } catch {
                    NSLog("[Jarvis] re-issue after approval failed: %@", error.localizedDescription)
                    let msg = "I'm having trouble connecting. Please check that Jarvis is running."
                    self.viewModel.finalizeTurn(response: msg)
                    self.showHUD(.response(text: msg))
                    self.speak(msg)
                }
            } else {
                self.hideHUD()
            }
        }
    }

    // MARK: - Config

    func refreshConfig() async {
        guard let url = URL(string: "http://127.0.0.1:8765/config"),
              let (data, _) = try? await URLSession.shared.data(from: url),
              let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let narration = config["narration"] as? [String: Any]
        else { return }
        stepVoiceEnabled = narration["step_voice"] as? Bool ?? false
    }

    // MARK: - SSE

    private func listenToEvents(commandId: String) async {
        let stepVoice = stepVoiceEnabled
        guard let url = URL(string: "http://127.0.0.1:8765/events/\(commandId)") else { return }
        var request = URLRequest(url: url)
        request.timeoutInterval = 180

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
        case "complete":
            if let data = try? JSONSerialization.data(withJSONObject: event),
               let response = try? JSONDecoder().decode(CommandResponse.self, from: data) {
                viewModel.finalizeTurn(response: response.text)
                handleCommandResponse(response)
            } else {
                let text = event["display"] as? String ?? event["speak"] as? String ?? "Done."
                viewModel.finalizeTurn(response: text)
                showHUD(.response(text: text))
            }
        case "error":
            let msg = event["message"] as? String ?? "Something went wrong."
            viewModel.finalizeTurn(response: "Error: \(msg)")
            showHUD(.response(text: msg))
            speak(msg)
        default:
            break
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

        // 1. Cancel and clean up prior state (order matters — delegate nil before stop)
        _ttsTask?.cancel()
        _ttsTask = nil
        _audioPlayer?.delegate = nil
        _audioPlayer?.stop()
        _audioPlayer = nil
        if let url = _currentTempURL {
            try? FileManager.default.removeItem(at: url)
        }
        _currentTempURL = nil

        // 2. POST to Piper TTS endpoint
        guard let url = URL(string: "http://127.0.0.1:8765/tts") else {
            speakWithDaniel(text)
            return
        }
        var request = URLRequest(url: url, timeoutInterval: 10)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: ["text": text])

        let task = URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                guard let self else { return }

                // 3. Validate response
                guard error == nil,
                      let http = response as? HTTPURLResponse,
                      http.statusCode == 200,
                      let contentType = http.value(forHTTPHeaderField: "Content-Type"),
                      contentType.contains("audio/wav"),
                      let data = data else {
                    NSLog("[Jarvis] Piper TTS failed (network/status/content-type) — falling back to Daniel")
                    self.speakWithDaniel(text)
                    return
                }

                // 4. Write WAV to temp file
                let tmpURL = URL(fileURLWithPath: NSTemporaryDirectory())
                    .appendingPathComponent("jarvis_tts_\(UUID().uuidString).wav")
                do {
                    try data.write(to: tmpURL)
                } catch {
                    NSLog("[Jarvis] TTS temp file write failed: %@ — falling back to Daniel",
                          error.localizedDescription)
                    self.speakWithDaniel(text)
                    return
                }
                self._currentTempURL = tmpURL

                // 5. Init AVAudioPlayer and play
                do {
                    let player = try AVAudioPlayer(contentsOf: tmpURL)
                    player.delegate = self
                    self._audioPlayer = player
                    player.play()
                } catch {
                    NSLog("[Jarvis] AVAudioPlayer init failed: %@ — falling back to Daniel",
                          error.localizedDescription)
                    try? FileManager.default.removeItem(at: tmpURL)
                    self._currentTempURL = nil
                    self.speakWithDaniel(text)
                }
            }
        }
        _ttsTask = task
        task.resume()
    }

    // AVAudioPlayerDelegate — fires on both success and failure (flag=false).
    // Must be nonisolated: AVFoundation dispatches callbacks on an arbitrary thread,
    // and AudioController is @MainActor. Direct property mutation without the nonisolated
    // marker causes a Swift 5.10 warning and a Swift 6 strict-concurrency error.
    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            if let url = _currentTempURL {
                try? FileManager.default.removeItem(at: url)
                _currentTempURL = nil
            }
            _audioPlayer = nil
        }
    }

    private func speakWithDaniel(_ text: String) {
        _synthesizer.stopSpeaking(at: .immediate)
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice =
            AVSpeechSynthesisVoice(identifier: "com.apple.ttsbundle.Daniel-premium") ??
            AVSpeechSynthesisVoice(identifier: "com.apple.voice.compact.en-GB.Daniel") ??
            AVSpeechSynthesisVoice(language: "en-GB")
        utterance.rate = 0.53
        utterance.pitchMultiplier = 1.0
        _synthesizer.speak(utterance)
    }
}
