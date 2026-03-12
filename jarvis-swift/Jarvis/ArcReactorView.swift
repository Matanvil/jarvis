import SwiftUI

/// Animated arc reactor icon shown when the HUD is minimized.
/// Energetic variant: fast-spinning rings, vivid blue glow, pulsing core.
struct ArcReactorView: View {

    // Ring rotations — each animates independently on appear
    @State private var ring2Rotation: Double = 0
    @State private var ring3Rotation: Double = 0
    @State private var ring4Rotation: Double = 0
    @State private var arcAngle: Double = 0
    @State private var coreBright: Bool = false
    @State private var glowLarge: Bool = false

    private let size: CGFloat = 72
    private let blue = Color(red: 0.22, green: 0.74, blue: 0.97)
    private let lightBlue = Color(red: 0.49, green: 0.83, blue: 0.99)

    var body: some View {
        ZStack {
            // Outer ambient glow
            Circle()
                .fill(blue.opacity(glowLarge ? 0.28 : 0.12))
                .frame(width: size + 28, height: size + 28)
                .blur(radius: 14)

            // Ring 1 — static outer border with segment dots
            ZStack {
                Circle()
                    .stroke(blue.opacity(0.65), lineWidth: 1.5)
                    .shadow(color: blue.opacity(0.45), radius: 8)

                // 8 segment dots evenly spaced at radius 34
                ForEach(0..<8, id: \.self) { i in
                    let angle = Double(i) * 45.0 * .pi / 180.0
                    Circle()
                        .fill(lightBlue)
                        .frame(width: 5, height: 5)
                        .shadow(color: blue.opacity(0.9), radius: 3)
                        .offset(
                            x: 34 * CGFloat(cos(angle)),
                            y: 34 * CGFloat(sin(angle))
                        )
                }
            }
            .frame(width: size, height: size)

            // Ring 2 — rotates CW with two arc overlays
            ZStack {
                Circle()
                    .stroke(lightBlue.opacity(0.45), lineWidth: 1.5)

                // Fast arc CW
                Circle()
                    .trim(from: 0, to: 0.55)
                    .stroke(blue.opacity(0.95), style: StrokeStyle(lineWidth: 1.5, lineCap: .round))
                    .rotationEffect(.degrees(arcAngle))

                // Counter arc
                Circle()
                    .trim(from: 0, to: 0.40)
                    .stroke(blue.opacity(0.60), style: StrokeStyle(lineWidth: 1.5, lineCap: .round))
                    .rotationEffect(.degrees(-arcAngle * 0.65))
            }
            .frame(width: size - 14, height: size - 14)
            .rotationEffect(.degrees(ring2Rotation))

            // Ring 3 — rotates CCW with a slow arc overlay
            ZStack {
                Circle()
                    .stroke(Color(red: 0.73, green: 0.90, blue: 0.99).opacity(0.35), lineWidth: 1.5)

                Circle()
                    .trim(from: 0, to: 0.50)
                    .stroke(lightBlue.opacity(0.80), style: StrokeStyle(lineWidth: 1.5, lineCap: .round))
                    .rotationEffect(.degrees(ring3Rotation * 1.4))
            }
            .frame(width: size - 28, height: size - 28)
            .rotationEffect(.degrees(-ring3Rotation))

            // Ring 4 — innermost thin ring, slow CW
            Circle()
                .stroke(blue.opacity(0.50), lineWidth: 1.5)
                .frame(width: size - 44, height: size - 44)
                .rotationEffect(.degrees(ring4Rotation))

            // Core — radial gradient with pulsing shadow
            Circle()
                .fill(
                    RadialGradient(
                        gradient: Gradient(colors: [
                            Color.white,
                            Color(red: 0.78, green: 0.95, blue: 1.0),
                            blue,
                            Color(red: 0.04, green: 0.64, blue: 0.91),
                        ]),
                        center: .center,
                        startRadius: 0,
                        endRadius: 10
                    )
                )
                .frame(width: size - 56, height: size - 56)
                .shadow(color: blue.opacity(coreBright ? 1.0 : 0.65), radius: coreBright ? 14 : 7)
                .shadow(color: blue.opacity(coreBright ? 0.85 : 0.45), radius: coreBright ? 26 : 13)
        }
        .frame(width: size, height: size)
        .onAppear { startAnimations() }
    }

    private func startAnimations() {
        withAnimation(.linear(duration: 8).repeatForever(autoreverses: false)) {
            ring2Rotation = 360
        }
        withAnimation(.linear(duration: 6).repeatForever(autoreverses: false)) {
            ring3Rotation = 360
        }
        withAnimation(.linear(duration: 5).repeatForever(autoreverses: false)) {
            ring4Rotation = 360
        }
        withAnimation(.linear(duration: 3).repeatForever(autoreverses: false)) {
            arcAngle = 360
        }
        withAnimation(.easeInOut(duration: 2.5).repeatForever(autoreverses: true)) {
            coreBright = true
            glowLarge  = true
        }
    }
}
