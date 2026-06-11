import Foundation

/// Shared secret between this app and the local Jarvis Python core.
///
/// The token is generated once per app launch and passed to the Python process via
/// the JARVIS_AUTH_TOKEN environment variable at spawn (see AppDelegate). Every HTTP
/// request to the core must carry it as the `X-Jarvis-Token` header, so other local
/// processes can't drive the server (which can run shell commands and edit files).
enum ServerAuth {
    static let headerField = "X-Jarvis-Token"

    /// Stable for the lifetime of the app process. `static let` initializes lazily once.
    static let token: String = UUID().uuidString

    /// Attach the auth header to an outgoing request.
    static func apply(to request: inout URLRequest) {
        request.setValue(token, forHTTPHeaderField: headerField)
    }
}
