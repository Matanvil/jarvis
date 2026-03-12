import Foundation

/// One complete user→Jarvis exchange in the conversation thread.
struct ConversationTurn: Identifiable {
    let id: UUID
    let command: String
    var steps: [Step]
    var response: String?   // nil while in-progress
    let timestamp: Date

    init(command: String) {
        self.id = UUID()
        self.command = command
        self.steps = []
        self.response = nil
        self.timestamp = Date()
    }
}

// MARK: - Encodable for session persistence

extension ConversationTurn: Encodable {
    private enum CodingKeys: String, CodingKey {
        case command, steps, response, timestamp
        // id is runtime-only (Identifiable); omitted from persistence intentionally
    }
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(command, forKey: .command)
        try c.encode(steps, forKey: .steps)
        try c.encodeIfPresent(response, forKey: .response)
        try c.encode(timestamp, forKey: .timestamp)
    }
}

// Note: Step is already `Codable` in JarvisClient.swift with the correct CodingKeys
// (inputSummary → "input_summary"). No additional conformance needed here.
