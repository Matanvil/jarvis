import SwiftUI

// MARK: - Info Badge (reusable ? tooltip component)

struct InfoBadge: View {
    let text: String
    @State private var showPopover = false

    var body: some View {
        Button {
            showPopover.toggle()
        } label: {
            ZStack {
                Circle().fill(Color(NSColor.controlBackgroundColor))
                Text("?")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundColor(.secondary)
            }
            .frame(width: 14, height: 14)
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showPopover, arrowEdge: .trailing) {
            Text(text)
                .font(.system(size: 12))
                .padding(10)
                .frame(maxWidth: 220)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: - Sidebar

private enum SettingsSection: String, CaseIterable, Identifiable {
    case general    = "General"
    case ai         = "AI & Routing"
    case telegram   = "Telegram"
    case narration  = "Narration"
    case guardrails = "Guardrails"
    case advanced   = "Advanced"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .general:    return "gearshape"
        case .ai:         return "cpu"
        case .telegram:   return "paperplane"
        case .narration:  return "waveform"
        case .guardrails: return "lock.shield"
        case .advanced:   return "wrench.and.screwdriver"
        }
    }
}

// MARK: - Main View

struct SettingsView: View {
    @StateObject private var vm = SettingsViewModel()
    @State private var selectedSection: SettingsSection = .general
    let onDismiss: () -> Void

    var body: some View {
        HStack(spacing: 0) {
            // Sidebar
            VStack(alignment: .leading, spacing: 2) {
                ForEach(SettingsSection.allCases) { section in
                    sidebarItem(section)
                }
            }
            .padding(.vertical, 12)
            .frame(width: 148)
            .background(Color(NSColor.controlBackgroundColor).opacity(0.5))

            Divider()

            // Content
            VStack(spacing: 0) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        sectionContent(selectedSection)
                            .padding(20)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                Divider()

                // Footer
                VStack(spacing: 6) {
                    Text(vm.saveError ?? "")
                        .font(.system(size: 11))
                        .foregroundColor(.red)
                        .frame(height: 14)
                    HStack {
                        Spacer()
                        Button("Cancel") { onDismiss() }
                            .keyboardShortcut(.cancelAction)
                        Button(vm.isSaving ? "Saving…" : "Save") {
                            Task { await handleSave() }
                        }
                        .keyboardShortcut(.defaultAction)
                        .disabled(vm.isSaving || vm.isLoading)
                    }
                }
                .padding(12)
            }
        }
        .frame(width: 700, height: 480)
        .task { await vm.load() }
    }

    // MARK: - Sidebar item

    @ViewBuilder
    private func sidebarItem(_ section: SettingsSection) -> some View {
        let isSelected = selectedSection == section
        Button {
            selectedSection = section
        } label: {
            HStack(spacing: 8) {
                Image(systemName: section.icon)
                    .frame(width: 16)
                Text(section.rawValue)
                    .font(.system(size: 12))
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(
                isSelected
                    ? Color.accentColor.opacity(0.15)
                    : Color.clear
            )
            .overlay(
                Rectangle()
                    .frame(width: 3)
                    .foregroundColor(isSelected ? .accentColor : .clear),
                alignment: .leading
            )
            .foregroundColor(isSelected ? .accentColor : .secondary)
        }
        .buttonStyle(.plain)
    }

    // MARK: - Section content router

    @ViewBuilder
    private func sectionContent(_ section: SettingsSection) -> some View {
        switch section {
        case .general:    GeneralSection(vm: vm)
        case .ai:         AISection(vm: vm)
        case .telegram:   TelegramSection(vm: vm)
        case .narration:  NarrationSection(vm: vm)
        case .guardrails: GuardrailsSection(vm: vm)
        case .advanced:   AdvancedSection(vm: vm)
        }
    }

    // MARK: - Save + restart prompt

    private func handleSave() async {
        await vm.save()
        guard vm.saveError == nil else { return }
        if vm.needsRestart {
            showRestartAlert()
        } else {
            onDismiss()
        }
    }

    private func showRestartAlert() {
        let alert = NSAlert()
        alert.messageText = "Restart Required"
        alert.informativeText = "Some changes (routing mode, local executor host, or model IDs) require a server restart to take effect."
        alert.addButton(withTitle: "Restart Now")
        alert.addButton(withTitle: "Later")
        let response = alert.runModal()
        if response == .alertFirstButtonReturn {
            (NSApp.delegate as? AppDelegate)?.restartPythonCore()
        }
        onDismiss()
    }
}

// MARK: - Shared row helper

private struct SettingRow<Control: View>: View {
    let label: String
    let tooltip: String?
    let control: Control
    let restartRequired: Bool

    init(_ label: String, tooltip: String? = nil, restartRequired: Bool = false, @ViewBuilder control: () -> Control) {
        self.label = label
        self.tooltip = tooltip
        self.restartRequired = restartRequired
        self.control = control()
    }

    var body: some View {
        HStack {
            HStack(spacing: 4) {
                Text(label).font(.system(size: 12))
                if restartRequired {
                    Text("↺")
                        .font(.system(size: 9))
                        .foregroundColor(.orange)
                        .padding(.horizontal, 3)
                        .background(Color.orange.opacity(0.15))
                        .cornerRadius(3)
                }
                if let tip = tooltip {
                    InfoBadge(text: tip)
                }
            }
            Spacer()
            control
        }
        .padding(.vertical, 6)
        Divider().opacity(0.4)
    }
}

private struct SectionHeader: View {
    let title: String
    var body: some View {
        Text(title.uppercased())
            .font(.system(size: 10, weight: .semibold))
            .foregroundColor(.secondary)
            .padding(.bottom, 8)
    }
}

// MARK: - General Section

private struct GeneralSection: View {
    @ObservedObject var vm: SettingsViewModel
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "General")
            SettingRow("Anthropic API Key",
                       tooltip: "Your Claude API key from console.anthropic.com. Required for all AI features. Never shared or logged.") {
                SecureField("sk-ant-…", text: $vm.anthropicApiKey)
                    .textFieldStyle(.roundedBorder).frame(width: 180)
            }
            SettingRow("Brave Search API Key",
                       tooltip: "Optional. Enables higher-quality web search results. Without it, Jarvis falls back to DuckDuckGo.") {
                SecureField("optional", text: $vm.braveApiKey)
                    .textFieldStyle(.roundedBorder).frame(width: 180)
            }
            SettingRow("Voice",
                       tooltip: "macOS TTS voice name used for spoken responses. E.g. Daniel, Samantha, Alex.") {
                TextField("Daniel", text: $vm.voice)
                    .textFieldStyle(.roundedBorder).frame(width: 120)
            }
            SettingRow("Wake word",
                       tooltip: "The phrase Jarvis listens for when always-on mode is enabled. Say this to activate without the hotkey.") {
                TextField("hey jarvis", text: $vm.wakeWord)
                    .textFieldStyle(.roundedBorder).frame(width: 140)
            }
            SettingRow("Always-on listening",
                       tooltip: "When ON, Jarvis continuously listens for the wake word. Uses the microphone at all times. OFF means hotkey-only activation.") {
                Toggle("", isOn: $vm.alwaysOn).labelsHidden()
            }
        }
    }
}

// MARK: - AI & Routing Section

private struct AISection: View {
    @ObservedObject var vm: SettingsViewModel

    private struct ModeOption: Identifiable {
        let id: String
        let label: String
        let description: String
    }

    private let routingModes: [ModeOption] = [
        ModeOption(id: "automatic", label: "Automatic",
                   description: "Classifier pre-flights intent. Non-complex → local model. Complex reasoning → Sonnet."),
        ModeOption(id: "local",     label: "Local only",
                   description: "Everything runs on the local model. No cloud API calls."),
        ModeOption(id: "cloud",     label: "Cloud only",
                   description: "Skip classifier, send everything directly to Sonnet."),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "AI & Routing")
            SettingRow("Routing mode",
                       tooltip: "automatic: classifier routes simple tasks to the local model, complex tasks to Sonnet. local: no cloud at all. cloud: always Sonnet.",
                       restartRequired: true) {
                Picker("", selection: $vm.routingMode) {
                    ForEach(routingModes) { mode in
                        Text(mode.label).tag(mode.id)
                    }
                }
                .labelsHidden().frame(width: 140)
            }
            if let selected = routingModes.first(where: { $0.id == vm.routingMode }) {
                Text(selected.description)
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .padding(.bottom, 8)
            }
            SettingRow("Local executor host",
                       tooltip: "URL of your Ollama instance. Default is localhost. Change if running Ollama on a remote machine.",
                       restartRequired: true) {
                TextField("http://localhost:11434", text: $vm.localHost)
                    .textFieldStyle(.roundedBorder).frame(width: 180)
            }
            SettingRow("Local executor model",
                       tooltip: "The local model served by Ollama/MLX used as primary executor. Must be pulled via 'ollama pull'.") {
                TextField("qwen3.6:35b-a3b", text: $vm.localModel)
                    .textFieldStyle(.roundedBorder).frame(width: 180)
            }
            SettingRow("Local executor timeout (s)",
                       tooltip: "How long to wait for the local model before giving up. Default 300s for large models.") {
                Stepper("\(vm.localTimeout)s", value: $vm.localTimeout, in: 30...600, step: 30)
                    .frame(width: 110)
            }
            Text("↺ fields require a server restart to take effect.")
                .font(.system(size: 11))
                .foregroundColor(.secondary)
                .padding(.top, 8)
        }
    }
}

// MARK: - Telegram Section

private struct TelegramSection: View {
    @ObservedObject var vm: SettingsViewModel
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "Telegram")
            SettingRow("Bot token",
                       tooltip: "The token from @BotFather. Create a bot at t.me/BotFather and paste the token here to enable Telegram control.") {
                SecureField("paste token…", text: $vm.telegramBotToken)
                    .textFieldStyle(.roundedBorder).frame(width: 200)
            }
            SettingRow("Allowed user ID",
                       tooltip: "Your Telegram numeric user ID. Only this user can send commands to Jarvis. Get yours from @userinfobot.") {
                TextField("123456789", text: $vm.telegramUserId)
                    .textFieldStyle(.roundedBorder).frame(width: 120)
            }
        }
    }
}

// MARK: - Narration Section

private struct NarrationSection: View {
    @ObservedObject var vm: SettingsViewModel
    private let modes = ["milestones", "all", "silent"]
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "Narration")
            SettingRow("Mode",
                       tooltip: "milestones: speaks only at key steps. all: narrates every tool call. silent: no voice output at all.") {
                Picker("", selection: $vm.narrationMode) {
                    ForEach(modes, id: \.self) { Text($0).tag($0) }
                }
                .labelsHidden().frame(width: 120)
            }
            SettingRow("Narrate step names",
                       tooltip: "When ON, Jarvis announces the name of each tool it's using (e.g. 'searching the web'). Useful for longer tasks.") {
                Toggle("", isOn: $vm.narrationStepVoice).labelsHidden()
            }
        }
    }
}

// MARK: - Guardrails Section

private struct GuardrailsSection: View {
    @ObservedObject var vm: SettingsViewModel

    private struct GuardrailRow {
        let label: String
        let tooltip: String
        let binding: Binding<Bool>
    }

    private var rows: [GuardrailRow] {[
        GuardrailRow(label: "Read files",        tooltip: "Read-only file access. Safe to leave ON.",
                     binding: $vm.grReadFiles),
        GuardrailRow(label: "Create files",      tooltip: "Creating new files.",
                     binding: $vm.grCreateFiles),
        GuardrailRow(label: "Edit files",        tooltip: "Modifying the contents of existing files.",
                     binding: $vm.grEditFiles),
        GuardrailRow(label: "Modify filesystem", tooltip: "Moving, renaming, or creating directories. OFF = ask before restructuring.",
                     binding: $vm.grModifyFilesystem),
        GuardrailRow(label: "Delete files",      tooltip: "Permanent file deletion. Recommended OFF — Jarvis will always ask first.",
                     binding: $vm.grDeleteFiles),
        GuardrailRow(label: "Run shell",         tooltip: "Execute terminal commands. ON = runs freely. OFF = approval required for each command.",
                     binding: $vm.grRunShell),
        GuardrailRow(label: "Run code",          tooltip: "Run code snippets with side effects (writes to disk, network calls).",
                     binding: $vm.grRunCode),
        GuardrailRow(label: "Web search",        tooltip: "Search the web. Safe to leave ON.",
                     binding: $vm.grWebSearch),
        GuardrailRow(label: "Open apps",         tooltip: "Open macOS applications on your behalf.",
                     binding: $vm.grOpenApps),
        GuardrailRow(label: "Send messages",     tooltip: "Send Telegram messages or notifications on your behalf. Recommended OFF.",
                     binding: $vm.grSendMessages),
        GuardrailRow(label: "Modify system",     tooltip: "System-level changes: preferences, permissions, startup items. Recommended OFF.",
                     binding: $vm.grModifySystem),
    ]}

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "Guardrails")
            Text("ON = runs automatically    OFF = asks for your approval first")
                .font(.system(size: 11))
                .foregroundColor(.secondary)
                .padding(.bottom, 10)
            ForEach(rows, id: \.label) { row in
                SettingRow(row.label, tooltip: row.tooltip) {
                    Toggle("", isOn: row.binding).labelsHidden()
                }
            }
        }
    }
}

// MARK: - Advanced Section

private struct AdvancedSection: View {
    @ObservedObject var vm: SettingsViewModel
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "Advanced")
            SettingRow("Server port",
                       tooltip: "The local port the Jarvis Python server listens on. Only change if 8765 conflicts with another service. Requires restart.",
                       restartRequired: true) {
                Stepper("\(vm.serverPort)", value: $vm.serverPort, in: 1024...65535, step: 1)
                    .frame(width: 100)
            }
            SettingRow("Max Claude steps",
                       tooltip: "How many tool calls Claude can make per command before stopping. Higher = more thorough but slower and costlier.") {
                Stepper("\(vm.maxStepsClaude)", value: $vm.maxStepsClaude, in: 1...30)
                    .frame(width: 80)
            }
            SettingRow("Max local steps",
                       tooltip: "How many tool calls the local model can make per command.") {
                Stepper("\(vm.maxStepsLocal)", value: $vm.maxStepsLocal, in: 1...30)
                    .frame(width: 80)
            }
            SettingRow("Stall detection",
                       tooltip: "When ON, Jarvis detects if it's calling the same tool twice with the same input and injects a warning to break the loop.") {
                Toggle("", isOn: $vm.stallDetection).labelsHidden()
            }
            SettingRow("Haiku model ID",
                       tooltip: "The Claude model used for most commands. Only change to pin to a specific version.",
                       restartRequired: true) {
                TextField("", text: $vm.haikuModel)
                    .textFieldStyle(.roundedBorder).frame(width: 200)
            }
            SettingRow("Sonnet model ID",
                       tooltip: "The Claude model used for complex reasoning tasks. Only change to pin to a specific version.",
                       restartRequired: true) {
                TextField("", text: $vm.sonnetModel)
                    .textFieldStyle(.roundedBorder).frame(width: 200)
            }
        }
    }
}
