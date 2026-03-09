import SwiftUI

@main
struct JarvisApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        // No window — app lives in the menu bar only.
        // Settings scene keeps the SwiftUI lifecycle running.
        Settings { EmptyView() }
    }
}
