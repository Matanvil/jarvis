import Foundation
import Combine

@MainActor
final class SettingsViewModel: ObservableObject {

    // MARK: - General
    @Published var anthropicApiKey: String = ""
    @Published var braveApiKey: String = ""
    @Published var voice: String = "Daniel"
    @Published var wakeWord: String = "hey jarvis"
    @Published var alwaysOn: Bool = false

    // MARK: - AI & Routing
    @Published var routingMode: String = "haiku_first"
    @Published var ollamaHost: String = "http://localhost:11434"
    @Published var ollamaModel: String = "llama3.1:8b"
    @Published var ollamaTimeout: Int = 30

    // MARK: - Telegram
    @Published var telegramBotToken: String = ""
    @Published var telegramUserId: String = ""

    // MARK: - Narration
    @Published var narrationMode: String = "milestones"
    @Published var narrationStepVoice: Bool = false

    // MARK: - Guardrails (true = auto_allow, false = require_approval)
    @Published var grReadFiles: Bool = true
    @Published var grCreateFiles: Bool = true
    @Published var grEditFiles: Bool = true
    @Published var grModifyFilesystem: Bool = false
    @Published var grDeleteFiles: Bool = false
    @Published var grRunShell: Bool = true
    @Published var grRunCode: Bool = true
    @Published var grWebSearch: Bool = true
    @Published var grOpenApps: Bool = true
    @Published var grSendMessages: Bool = false
    @Published var grModifySystem: Bool = false

    // MARK: - Advanced
    @Published var serverPort: Int = 8765
    @Published var maxStepsClaude: Int = 10
    @Published var maxStepsOllama: Int = 5
    @Published var maxTotalSteps: Int = 20
    @Published var stallDetection: Bool = true
    @Published var haikuModel: String = "claude-haiku-4-5-20251001"
    @Published var sonnetModel: String = "claude-sonnet-4-6"

    // MARK: - State
    @Published var isLoading: Bool = false
    @Published var isSaving: Bool = false
    @Published var saveError: String? = nil
    @Published var needsRestart: Bool = false

    // Snapshot of restart-sensitive values at load time — used to detect changes
    private var loadedRestartValues: [String: String] = [:]

    private let baseURL = "http://127.0.0.1:8765"

    // MARK: - Load

    func load() async {
        isLoading = true
        saveError = nil
        defer { isLoading = false }
        guard let url = URL(string: "\(baseURL)/config") else { return }
        guard let (data, _) = try? await URLSession.shared.data(from: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            saveError = "Could not load config — is Jarvis running?"
            return
        }
        applyJSON(json)
        snapshotRestartValues()
    }

    private func applyJSON(_ json: [String: Any]) {
        // General — sensitive fields always start blank
        voice     = json["voice"]      as? String ?? "Daniel"
        wakeWord  = json["wake_word"]  as? String ?? "hey jarvis"
        alwaysOn  = json["always_on"]  as? Bool   ?? false
        // anthropicApiKey, braveApiKey stay blank (server redacts them)

        // AI & Routing
        if let ollama = json["ollama"] as? [String: Any] {
            routingMode   = ollama["routing_mode"]    as? String ?? "haiku_first"
            ollamaHost    = ollama["host"]            as? String ?? "http://localhost:11434"
            ollamaModel   = ollama["model"]           as? String ?? "llama3.1:8b"
            ollamaTimeout = ollama["timeout_seconds"] as? Int    ?? 30
        }

        // Telegram — bot_token always starts blank (server redacts it)
        if let tg = json["telegram"] as? [String: Any] {
            telegramUserId = tg["allowed_user_id"].map { "\($0)" } ?? ""
            // telegramBotToken stays blank
        }

        // Narration
        if let narration = json["narration"] as? [String: Any] {
            narrationMode      = narration["mode"]       as? String ?? "milestones"
            narrationStepVoice = narration["step_voice"] as? Bool   ?? false
        }

        // Guardrails
        if let gr = json["guardrails"] as? [String: Any] {
            grReadFiles        = (gr["read_files"]            as? String) == "auto_allow"
            grCreateFiles      = (gr["create_files"]          as? String) == "auto_allow"
            grEditFiles        = (gr["edit_files"]            as? String) == "auto_allow"
            grModifyFilesystem = (gr["modify_filesystem"]     as? String) == "auto_allow"
            grDeleteFiles      = (gr["delete_files"]          as? String) == "auto_allow"
            grRunShell         = (gr["run_shell"]             as? String) == "auto_allow"
            grRunCode          = (gr["run_code_with_effects"] as? String) == "auto_allow"
            grWebSearch        = (gr["web_search"]            as? String) == "auto_allow"
            grOpenApps         = (gr["open_apps"]             as? String) == "auto_allow"
            grSendMessages     = (gr["send_messages"]         as? String) == "auto_allow"
            grModifySystem     = (gr["modify_system"]         as? String) == "auto_allow"
        }

        // Advanced
        serverPort = json["server_port"] as? Int ?? 8765
        if let reasoning = json["reasoning"] as? [String: Any] {
            maxStepsClaude = reasoning["max_steps_claude"] as? Int  ?? 10
            maxStepsOllama = reasoning["max_steps_ollama"] as? Int  ?? 5
            maxTotalSteps  = reasoning["max_total_steps"]  as? Int  ?? 20
            stallDetection = reasoning["stall_detection"]  as? Bool ?? true
        }
        if let models = json["models"] as? [String: Any] {
            haikuModel  = models["haiku"]  as? String ?? "claude-haiku-4-5-20251001"
            sonnetModel = models["sonnet"] as? String ?? "claude-sonnet-4-6"
        }
    }

    private func snapshotRestartValues() {
        loadedRestartValues = [
            "server_port":         "\(serverPort)",
            "ollama.routing_mode": routingMode,
            "ollama.host":         ollamaHost,
            "models.haiku":        haikuModel,
            "models.sonnet":       sonnetModel,
        ]
    }

    // MARK: - Save

    func save() async {
        isSaving = true
        saveError = nil
        defer { isSaving = false }

        let payload = buildPayload()
        guard let url = URL(string: "\(baseURL)/config"),
              let body = try? JSONSerialization.data(withJSONObject: payload) else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body

        guard let (_, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200 else {
            saveError = "Failed to save — is Jarvis running?"
            return
        }

        // Detect if any restart-required field changed
        let currentRestartValues: [String: String] = [
            "server_port":         "\(serverPort)",
            "ollama.routing_mode": routingMode,
            "ollama.host":         ollamaHost,
            "models.haiku":        haikuModel,
            "models.sonnet":       sonnetModel,
        ]
        needsRestart = currentRestartValues != loadedRestartValues
        snapshotRestartValues()
    }

    private func buildPayload() -> [String: Any] {
        var tg: [String: Any] = [:]
        if let uid = Int(telegramUserId) { tg["allowed_user_id"] = uid }

        var payload: [String: Any] = [
            "voice":       voice,
            "wake_word":   wakeWord,
            "always_on":   alwaysOn,
            "server_port": serverPort,
            "ollama": [
                "routing_mode":    routingMode,
                "host":            ollamaHost,
                "model":           ollamaModel,
                "timeout_seconds": ollamaTimeout,
            ],
            "telegram": tg,
            "narration": [
                "mode":       narrationMode,
                "step_voice": narrationStepVoice,
            ],
            "guardrails": [
                "read_files":            grReadFiles        ? "auto_allow" : "require_approval",
                "create_files":          grCreateFiles      ? "auto_allow" : "require_approval",
                "edit_files":            grEditFiles        ? "auto_allow" : "require_approval",
                "modify_filesystem":     grModifyFilesystem ? "auto_allow" : "require_approval",
                "delete_files":          grDeleteFiles      ? "auto_allow" : "require_approval",
                "run_shell":             grRunShell         ? "auto_allow" : "require_approval",
                "run_code_with_effects": grRunCode          ? "auto_allow" : "require_approval",
                "web_search":            grWebSearch        ? "auto_allow" : "require_approval",
                "open_apps":             grOpenApps         ? "auto_allow" : "require_approval",
                "send_messages":         grSendMessages     ? "auto_allow" : "require_approval",
                "modify_system":         grModifySystem     ? "auto_allow" : "require_approval",
            ],
            "reasoning": [
                "max_steps_claude": maxStepsClaude,
                "max_steps_ollama": maxStepsOllama,
                "max_total_steps":  maxTotalSteps,
                "stall_detection":  stallDetection,
            ],
            "models": [
                "haiku":  haikuModel,
                "sonnet": sonnetModel,
            ],
        ]
        // Only include sensitive fields if the user typed a new value
        if !anthropicApiKey.isEmpty { payload["anthropic_api_key"] = anthropicApiKey }
        if !braveApiKey.isEmpty     { payload["brave_api_key"]     = braveApiKey     }
        if !telegramBotToken.isEmpty {
            var tgUpdated = payload["telegram"] as? [String: Any] ?? [:]
            tgUpdated["bot_token"] = telegramBotToken
            payload["telegram"] = tgUpdated
        }
        return payload
    }
}
