import SwiftUI

struct FullDesktopView: View {
    @ObservedObject var viewModel: HUDViewModel
    @ObservedObject var fullViewModel: FullDesktopViewModel
    @ObservedObject var metricsProvider: SystemMetricsProvider
    var onCollapse: () -> Void = {}

    var body: some View {
        Text("Full Desktop Mode")
            .foregroundColor(.white)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(red: 0.039, green: 0.059, blue: 0.098))
    }
}
