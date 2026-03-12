import AppKit
import SwiftUI

class HUDWindow: NSPanel {

    private let screenFrame: NSRect
    private static let positionKeyFull      = "hudFullPosition"
    private static let positionKeyMinimized = "hudMinimizedPosition"

    private(set) var isMinimized: Bool = false

    init(viewModel: HUDViewModel) {
        screenFrame = NSScreen.main?.visibleFrame
            ?? NSScreen.screens.first?.visibleFrame
            ?? NSRect(x: 0, y: 0, width: 1280, height: 800)

        // Default position: center-top
        let width: CGFloat = min(max(screenFrame.width * 0.5, 480), 900)
        let defaultX = screenFrame.midX - width / 2
        let defaultY = screenFrame.minY + 80

        // Load saved full-HUD position if available
        let savedOrigin = HUDWindow.loadOrigin(forKey: HUDWindow.positionKeyFull)
        let x = savedOrigin?.x ?? defaultX
        let y = savedOrigin?.y ?? defaultY

        super.init(
            contentRect: NSRect(x: x, y: y, width: width, height: 60),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        self.level = .floating
        self.isOpaque = false
        self.backgroundColor = .clear
        self.hasShadow = true
        self.isMovableByWindowBackground = true
        self.isReleasedWhenClosed = false
        self.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        self.delegate = self
    }

    // MARK: - Canonical sizes

    private var canonicalWidth: CGFloat {
        min(max(screenFrame.width * 0.5, 480), 900)
    }

    // MARK: - UserDefaults

    private static func loadOrigin(forKey key: String) -> NSPoint? {
        guard let str = UserDefaults.standard.string(forKey: key) else { return nil }
        let pt = NSPointFromString(str)
        // NSPointFromString returns .zero on failure — treat as missing
        return (pt == .zero) ? nil : pt
    }

    private func savePosition() {
        let key = isMinimized ? Self.positionKeyMinimized : Self.positionKeyFull
        UserDefaults.standard.set(NSStringFromPoint(frame.origin), forKey: key)
    }

    // MARK: - Minimize / expand sizing

    /// Collapse to 72×72 arc reactor icon. Restores saved minimized position if available.
    func resizeForMinimized() {
        isMinimized = true
        isMovableByWindowBackground = false   // tap gestures need priority; drag is handled in mouseDragged

        let size = NSSize(width: 72, height: 72)
        let saved = Self.loadOrigin(forKey: Self.positionKeyMinimized)
        let origin = clampedOrigin(saved ?? frame.origin, size: size)
        setFrame(NSRect(origin: origin, size: size), display: true, animate: true)
        applyContentMask(circle: true, size: size)
        savePosition()
    }

    /// Expand to full-width pill at the given height.
    /// If coming from minimized, anchors to the icon's current position and picks the best
    /// expansion direction so the pill stays fully on screen.
    func resizeForExpanded(toHeight height: CGFloat) {
        isMinimized = false
        isMovableByWindowBackground = true

        let width = canonicalWidth
        let size = NSSize(width: width, height: height)
        let origin = smartExpandOrigin(from: frame.origin, targetSize: size)
        setFrame(NSRect(origin: origin, size: size), display: true, animate: true)
        applyContentMask(circle: false, size: size)
        savePosition()
    }

    /// Mask the content view's layer to match the visible shape —
    /// circle for the minimized arc reactor, pill-shaped rounded rect for the expanded HUD.
    private func applyContentMask(circle: Bool, size: NSSize) {
        guard let layer = contentView?.layer else { return }
        let mask = CAShapeLayer()
        if circle {
            mask.path = CGPath(ellipseIn: CGRect(origin: .zero, size: size), transform: nil)
        } else {
            // The SwiftUI pill has .padding(8) and cornerRadius 20.
            // Inset the mask to match the pill exactly.
            let inset: CGFloat = 8
            let pillRect = CGRect(x: inset, y: inset,
                                  width: size.width - inset * 2,
                                  height: size.height - inset * 2)
            mask.path = CGPath(roundedRect: pillRect, cornerWidth: 20, cornerHeight: 20, transform: nil)
        }
        layer.mask = mask
    }

    // MARK: - Smart screen-aware expansion

    /// Given the current window origin (e.g. the reactor icon's bottom-left) and the target size,
    /// return an origin that keeps the window fully within the visible screen frame.
    private func smartExpandOrigin(from currentOrigin: NSPoint, targetSize: NSSize) -> NSPoint {
        let margin: CGFloat = 16
        let screen = NSScreen.main?.visibleFrame ?? screenFrame

        var x = currentOrigin.x
        var y = currentOrigin.y

        // Horizontal: shift left only if right edge would overflow
        if x + targetSize.width > screen.maxX - margin {
            x = screen.maxX - margin - targetSize.width
        }
        x = max(x, screen.minX + margin)

        // Vertical: shift down only if top edge would overflow
        if y + targetSize.height > screen.maxY - margin {
            y = screen.maxY - margin - targetSize.height
        }
        y = max(y, screen.minY + margin)

        return NSPoint(x: x, y: y)
    }

    /// Clamp origin so a window of `size` stays within the visible screen.
    private func clampedOrigin(_ origin: NSPoint, size: NSSize) -> NSPoint {
        let margin: CGFloat = 16
        let screen = NSScreen.main?.visibleFrame ?? screenFrame
        let x = min(max(origin.x, screen.minX + margin), screen.maxX - margin - size.width)
        let y = min(max(origin.y, screen.minY + margin), screen.maxY - margin - size.height)
        return NSPoint(x: x, y: y)
    }

    // MARK: - Manual drag when minimized
    // isMovableByWindowBackground is false when minimized so SwiftUI tap gestures work.
    // We handle dragging ourselves at the AppKit level.

    override func mouseDragged(with event: NSEvent) {
        guard isMinimized else {
            // Full HUD drag is handled by isMovableByWindowBackground = true; forward to super.
            super.mouseDragged(with: event)
            return
        }
        var origin = frame.origin
        origin.x += event.deltaX
        origin.y -= event.deltaY   // NSEvent deltaY is flipped relative to screen coords
        setFrameOrigin(clampedOrigin(origin, size: frame.size))
    }

    override func mouseUp(with event: NSEvent) {
        if isMinimized {
            savePosition()
        }
    }
}

// MARK: - NSWindowDelegate

extension HUDWindow: NSWindowDelegate {
    /// Save position whenever the user finishes dragging the full HUD.
    func windowDidMove(_ notification: Notification) {
        guard !isMinimized else { return }   // minimized drag is saved in mouseUp
        savePosition()
    }
}
