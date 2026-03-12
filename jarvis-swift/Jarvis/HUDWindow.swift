import AppKit
import SwiftUI

class HUDWindow: NSPanel {

    private let screenFrame: NSRect
    private static let positionKey = "hudPosition"

    private(set) var isMinimized: Bool = false

    init(viewModel: HUDViewModel) {
        screenFrame = NSScreen.main?.visibleFrame
            ?? NSScreen.screens.first?.visibleFrame
            ?? NSRect(x: 0, y: 0, width: 1280, height: 800)

        // Default position: center-top
        let width: CGFloat = min(max(screenFrame.width * 0.5, 480), 900)
        let defaultX = screenFrame.midX - width / 2
        let defaultY = screenFrame.minY + 80

        // Migrate legacy separate position keys → single shared key
        if UserDefaults.standard.object(forKey: Self.positionKey) == nil {
            let legacyStr = UserDefaults.standard.string(forKey: "hudMinimizedPosition")
                ?? UserDefaults.standard.string(forKey: "hudFullPosition")
            if let legacyStr {
                UserDefaults.standard.set(legacyStr, forKey: Self.positionKey)
            }
            UserDefaults.standard.removeObject(forKey: "hudMinimizedPosition")
            UserDefaults.standard.removeObject(forKey: "hudFullPosition")
        }
        let savedOrigin = HUDWindow.loadOrigin(forKey: HUDWindow.positionKey)
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

    // Issue #4: query live screen width so canonicalWidth stays correct after display changes.
    private var canonicalWidth: CGFloat {
        let screenWidth = NSScreen.main?.visibleFrame.width ?? screenFrame.width
        return min(max(screenWidth * 0.5, 480), 900)
    }

    // MARK: - UserDefaults

    private static func loadOrigin(forKey key: String) -> NSPoint? {
        // Issue #1: check key existence before parsing so {0,0} is a valid saved position.
        guard UserDefaults.standard.object(forKey: key) != nil,
              let str = UserDefaults.standard.string(forKey: key) else { return nil }
        return NSPointFromString(str)
    }

    private func savePosition() {
        UserDefaults.standard.set(NSStringFromPoint(frame.origin), forKey: Self.positionKey)
    }

    // MARK: - Minimize / expand sizing

    /// Collapse to 100×100 arc reactor icon (extra 14pt on each side for the ambient glow).
    /// Restores saved minimized position if available.
    func resizeForMinimized() {
        isMinimized = true
        isMovableByWindowBackground = false   // tap gestures need priority; drag is handled in mouseDragged

        let size = NSSize(width: 72, height: 72)
        let saved = Self.loadOrigin(forKey: Self.positionKey)
        let origin = clampedOrigin(saved ?? frame.origin, size: size)
        // Issue #5: animate: false — spec marks collapse/expand animation out of scope.
        setFrame(NSRect(origin: origin, size: size), display: true, animate: false)
        applyContentMask(circle: true, size: size)
        savePosition()
    }

    /// Expand to full-width pill at the given height.
    /// Anchors to the saved icon position so the HUD always expands near where the icon lives.
    func resizeForExpanded(toHeight height: CGFloat) {
        isMinimized = false
        isMovableByWindowBackground = true

        let width = canonicalWidth
        let size = NSSize(width: width, height: height)
        // Icon position is the source of truth — expand from it, don't overwrite it.
        let iconOrigin = Self.loadOrigin(forKey: Self.positionKey)
        let origin = smartExpandOrigin(from: iconOrigin ?? frame.origin, targetSize: size)
        // Issue #5: animate: false — spec marks collapse/expand animation out of scope.
        setFrame(NSRect(origin: origin, size: size), display: true, animate: false)
        applyContentMask(circle: false, size: size)
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
        super.mouseUp(with: event)   // Issue #8: maintain responder chain
    }
}

// MARK: - NSWindowDelegate

extension HUDWindow: NSWindowDelegate {
    // Position is persisted only when the icon is dragged (mouseUp above).
    // HUD drag does not update the icon anchor — the icon always returns to its saved spot.
}
