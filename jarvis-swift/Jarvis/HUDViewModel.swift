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
