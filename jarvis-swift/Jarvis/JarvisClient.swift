import Foundation

// MARK: - Response models

struct Step: Codable {
    let tool: String
    let inputSummary: String?
    let milestone: Bool?

    private enum CodingKeys: String, CodingKey {
        case tool
        case inputSummary = "input_summary"
        case milestone
    }
}

// Nested object returned by server when a tool needs approval
private struct ApprovalRequired: Codable {
    let tool: String?
    let description: String?
    let toolUseId: String?
    let category: String?

    private enum CodingKeys: String, CodingKey {
        case tool, description, category
        case toolUseId = "tool_use_id"
    }
}

struct CommandResponse: Decodable {
    /// Short text spoken aloud (< 150 chars). nil on approval_required.
    let speak: String?
    /// Full text shown in HUD. nil on approval_required.
    let display: String?
    let steps: [Step]
    let requiresApproval: Bool
    let toolUseId: String?
    let approvalDescription: String?
    /// Guardrail category to trust for session on approval (e.g. "run_code_with_effects")
    let approvalCategory: String?

    /// Convenience: best text to show in the HUD
    var text: String { display ?? speak ?? "" }

    private enum CodingKeys: String, CodingKey {
        case speak, display, steps
        case approvalRequired = "approval_required"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        speak   = try? c.decode(String.self, forKey: .speak)
        display = try? c.decode(String.self, forKey: .display)
        steps   = (try? c.decode([Step].self, forKey: .steps)) ?? []
        if let approval = try? c.decode(ApprovalRequired.self, forKey: .approvalRequired) {
            requiresApproval    = true
            toolUseId           = approval.toolUseId
            approvalDescription = approval.description
            approvalCategory    = approval.category
        } else {
            requiresApproval    = false
            toolUseId           = nil
            approvalDescription = nil
            approvalCategory    = nil
        }
    }
}

private struct ApprovalClassifyResponse: Codable {
    let approved: Bool?
}

struct CommandStartResponse: Decodable {
    let commandId: String
    private enum CodingKeys: String, CodingKey {
        case commandId = "command_id"
    }
}

// MARK: - Client

final class JarvisClient {
    private let baseURL = "http://127.0.0.1:8765"

    // Long timeout for commands — delegate_to_claude_code can take 60-120s
    private let commandSession: URLSession = {
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 180
        cfg.timeoutIntervalForResource = 300
        return URLSession(configuration: cfg)
    }()

    // 5s timeout for quick endpoints
    private let quickSession: URLSession = {
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 5
        return URLSession(configuration: cfg)
    }()

    /// POSTs to /command with source="swift" and returns command_id immediately.
    func startCommand(text: String, cwd: String?) async throws -> String {
        var request = URLRequest(url: URL(string: "\(baseURL)/command")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        var body: [String: Any] = ["text": text, "source": "swift"]
        if let cwd { body["cwd"] = cwd }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        // Short timeout — server returns command_id immediately
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 10
        let session = URLSession(configuration: cfg)
        let (data, _) = try await session.data(for: request)
        let resp = try JSONDecoder().decode(CommandStartResponse.self, from: data)
        return resp.commandId
    }

    /// Returns true if caller should re-issue the original command.
    func sendApproval(toolUseId: String, approved: Bool, category: String?) async throws -> Bool {
        var request = URLRequest(url: URL(string: "\(baseURL)/approve")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        var body: [String: Any] = ["tool_use_id": toolUseId, "approved": approved]
        if approved, let category {
            // Trust this guardrail category for the rest of the session so the
            // re-issued command auto-allows without triggering approval again.
            body["trust_session"] = true
            body["category"] = category
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, _) = try await quickSession.data(for: request)
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        return (json?["next_action"] as? String) == "reissue_command"
    }

    /// Returns true (approved), false (denied), or nil (unclear — caller should stay in approval state)
    func classifyApproval(text: String) async throws -> Bool? {
        var request = URLRequest(url: URL(string: "\(baseURL)/approve/classify")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["text": text])
        let (data, _) = try await quickSession.data(for: request)
        return try JSONDecoder().decode(ApprovalClassifyResponse.self, from: data).approved
    }
}
