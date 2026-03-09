# Security Policy

## Reporting a Vulnerability

Please do **not** open a public GitHub issue for security vulnerabilities.

Instead, report them privately via GitHub's [private vulnerability reporting](../../security/advisories/new) feature, or email the maintainer directly.

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

You'll receive a response within 48 hours.

## Scope

Key areas to review:
- Guardrails bypass (destructive actions without approval)
- Shell injection via tool inputs
- API key exposure via logs or responses
- AppleScript injection via macOS tool
