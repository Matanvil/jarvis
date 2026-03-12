import Foundation

struct ConversationStore {

    private static let conversationsDir: URL = {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".jarvis/conversations")
    }()

    /// Save completed turns to disk. In-progress turns (response == nil) are excluded.
    /// Called on background queue or synchronously from applicationWillTerminate.
    static func save(turns: [ConversationTurn], sessionStart: Date) {
        let completedTurns = turns.filter { $0.response != nil }
        guard !completedTurns.isEmpty else { return }

        let dir = conversationsDir
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        } catch {
            NSLog("[Jarvis] ConversationStore: failed to create directory: %@", error.localizedDescription)
            return
        }

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH-mm-ss-SSS"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        let filename = formatter.string(from: sessionStart) + ".json"
        let fileURL = dir.appendingPathComponent(filename)

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = .prettyPrinted
        do {
            let data = try encoder.encode(completedTurns)
            try data.write(to: fileURL, options: .atomic)
        } catch {
            NSLog("[Jarvis] ConversationStore: failed to write session: %@", error.localizedDescription)
            return
        }

        pruneToLimit(20)
    }

    /// Delete oldest files beyond the limit. Always called after save so the new file is kept.
    private static func pruneToLimit(_ limit: Int) {
        let dir = conversationsDir
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: .skipsHiddenFiles
        ) else { return }

        let sorted = files
            .compactMap { url -> (URL, Date)? in
                guard url.pathExtension == "json",
                      let date = try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
                else { return nil }
                return (url, date)
            }
            .sorted { $0.1 > $1.1 }   // newest first

        guard sorted.count > limit else { return }
        for (url, _) in sorted.dropFirst(limit) {
            try? FileManager.default.removeItem(at: url)
        }
    }
}
