# jarvis-swift — macOS App

SwiftUI menu bar app with floating HUD, global hotkey, STT, and TTS.

## Requirements

- macOS 13+
- Xcode 15+
- xcodegen (`npm install -g xcodegen`)
- Python core running on localhost:8765

## Build

```bash
xcodegen generate
open Jarvis.xcodeproj
# ⌘B to build
# ⌘R to run
```

## Key Files

| File | Purpose |
|---|---|
| `AppDelegate.swift` | App lifecycle, Python core launch + health polling |
| `AudioController.swift` | Hotkey (⌃Space), STT, TTS, approval prompts |
| `MenuBarController.swift` | Menu bar icon + Quit |
| `HUDWindow.swift` | Floating overlay window |
| `HUDView.swift` | SwiftUI response display |
| `HUDViewModel.swift` | State management |
| `JarvisClient.swift` | HTTP client to Python core |

## Permissions Required

- Microphone (STT)
- Speech Recognition
- Accessibility (global hotkey)

## Architecture

The Swift app is a thin client. It:
1. Launches and monitors the Python core process
2. Captures voice via SFSpeechRecognizer
3. POSTs transcribed text to `localhost:8765/command`
4. Displays the response in the HUD
5. Narrates summaries via `say`
