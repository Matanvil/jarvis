import subprocess

_APPLESCRIPT_TIMEOUT = 15  # seconds


class MacOSTool:
    def open_app(self, app_name: str) -> dict:
        try:
            result = subprocess.run(
                ["open", "-a", app_name],
                capture_output=True, text=True, timeout=_APPLESCRIPT_TIMEOUT
            )
            return {
                "success": result.returncode == 0,
                "error": result.stderr if result.returncode != 0 else None,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"open_app timed out after {_APPLESCRIPT_TIMEOUT}s"}

    def run_applescript(self, script: str) -> dict:
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=_APPLESCRIPT_TIMEOUT
            )
            if result.returncode != 0:
                return {"output": None, "error": result.stderr}
            return {"output": result.stdout, "error": None}
        except subprocess.TimeoutExpired:
            return {"output": None, "error": f"AppleScript timed out after {_APPLESCRIPT_TIMEOUT}s"}

    def notify(self, title: str, body: str) -> dict:
        # Sanitize to prevent AppleScript injection via double-quote breakout
        safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
        safe_body = body.replace("\\", "\\\\").replace('"', '\\"')
        script = f'display notification "{safe_body}" with title "{safe_title}"'
        result = self.run_applescript(script)
        return {
            "success": result["error"] is None,
            "error": result["error"],
        }

    def set_volume(self, level: int) -> dict:
        """level: 0-100"""
        clamped = max(0, min(100, level))
        return self.run_applescript(f"set volume output volume {clamped}")

    def get_frontmost_app(self) -> dict:
        return self.run_applescript(
            'tell application "System Events" to get name of first application process whose frontmost is true'
        )
