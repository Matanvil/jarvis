import SwiftUI

enum HUDState: Equatable {
    case hidden
    case listening
    case thinking
    case executing(step: String)
    case response(text: String)
    case approval(description: String)
    case approved
    case denied

    var preferredHeight: CGFloat {
        switch self {
        case .hidden:                   return 60
        case .listening, .thinking,
             .executing, .approved,
             .denied:                   return 60
        case .response(let text):
            // Approx 20px per line at 14pt, 560px usable width (~80 chars/line), 32px padding
            let lines = max(1, text.count / 80 + text.filter { $0 == "\n" }.count)
            return min(320, CGFloat(lines) * 22 + 64)
        case .approval:                 return 160
        }
    }
}

class HUDViewModel: ObservableObject {
    static let shared = HUDViewModel()
    @Published var state: HUDState = .hidden
    private init() {}
}
