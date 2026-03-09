# Contributing to Jarvis

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

1. Fork and clone the repo
2. Set up the Python core (see `jarvis-core/README.md`)
3. Set up the Swift app (see `jarvis-swift/README.md`)
4. Copy config: `cp jarvis-core/config.example.json ~/.jarvis/config.json`
5. Add your Anthropic API key to `~/.jarvis/config.json`

## Running Tests

```bash
cd jarvis-core
source .venv/bin/activate
pytest
```

All PRs must pass the full test suite.

## Submitting a PR

1. Create a branch: `git checkout -b feat/your-feature`
2. Write tests first (TDD — see existing test patterns in `jarvis-core/tests/`)
3. Implement the feature
4. Run `pytest` — all tests must pass
5. Open a PR with a clear description of what and why

## Code Style

- Python: standard PEP 8, no formatter enforced yet
- Swift: standard Swift conventions
- Keep functions small and focused
- No hardcoded secrets — all config goes through `~/.jarvis/config.json`

## What to Work On

Check the [Issues](../../issues) tab for open bugs and feature requests. Issues tagged `good first issue` are great starting points.

## Questions?

Open a [Discussion](../../discussions) or file an issue.
