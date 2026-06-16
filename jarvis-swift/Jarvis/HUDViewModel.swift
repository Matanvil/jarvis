import SwiftUI
import Combine

enum HUDState: Equatable {
    case hidden
    case listening
    case thinking
    case executing(step: String)
    case response(text: String)
    case approval(description: String)
    case approved
    case denied
    case minimized
}

class HUDViewModel: ObservableObject {
    static let shared = HUDViewModel()

    // MARK: - Status bar state (drives sticky bar + window show/hide)
    @Published var state: HUDState = .hidden

    // MARK: - Conversation thread
    @Published var turns: [ConversationTurn] = []

    /// Reported by HUDView via PreferenceKey — actual rendered height of the thread.
    /// Capped at 60% of screen height. AppDelegate observes this to resize the window.
    @Published var contentHeight: CGFloat = 120

    /// Set to true to focus the text input field. Resets itself to false after focus is applied (one-shot).
    @Published var focusTextInput: Bool = false

    /// Accumulates streaming tokens for live display. Cleared when the turn finalizes.
    @Published var streamingBuffer: String = ""

    /// Incoming token fragments queued by the network layer; drained at display rate by AudioController timer.
    private(set) var tokenQueue: String = ""

    /// Start time of the current session (first command's timestamp). Used for save filename.
    private(set) var sessionStart: Date = Date()

    var completedTurnCount: Int {
        turns.filter { $0.response != nil }.count
    }

    private init() {}

    // MARK: - Turn mutation (called by AudioController on main thread)

    /// Begin a new in-progress turn. Called when a command is sent.
    func startTurn(command: String) {
        if turns.isEmpty {
            sessionStart = Date()
        }
        turns.append(ConversationTurn(command: command))
    }

    /// Append a tool step to the most recent in-progress turn.
    func appendStep(_ step: Step) {
        guard !turns.isEmpty else { return }
        turns[turns.count - 1].steps.append(step)
    }

    /// Finalize the most recent turn with the given response text.
    func finalizeTurn(response: String) {
        guard !turns.isEmpty else { return }
        turns[turns.count - 1].response = response
    }

    /// Queue an incoming token fragment. Does NOT update streamingBuffer — the display timer does that.
    func appendToken(_ token: String) {
        tokenQueue += token
    }

    /// Drain up to `chars` characters from the queue into streamingBuffer. Returns true if queue still has data.
    @discardableResult
    func drainTokenQueue(chars: Int = 8) -> Bool {
        guard !tokenQueue.isEmpty else { return false }
        let slice = String(tokenQueue.prefix(chars))
        tokenQueue = String(tokenQueue.dropFirst(chars))
        streamingBuffer += slice
        return !tokenQueue.isEmpty
    }

    /// Clear the streaming buffer and pending queue (called on complete/clear events).
    func clearStreamingBuffer() {
        tokenQueue = ""
        streamingBuffer = ""
    }

    /// Save current session and clear the thread.
    func newConversation() {
        let snapshot = turns
        let start = sessionStart
        DispatchQueue.global(qos: .utility).async {
            ConversationStore.save(turns: snapshot, sessionStart: start)
        }
        turns = []
        sessionStart = Date()
        contentHeight = 120
    }

    /// Synchronous save — called from applicationWillTerminate.
    func saveSessionSync() {
        ConversationStore.save(turns: turns, sessionStart: sessionStart)
    }
}
