# Release Checklist

Before committing to git or making the project public, verify:

## 1. Secrets Audit

```bash
# Check for .env files (should NOT exist)
ls -la .env*

# Check for API keys in code
grep -r "sk-" --include="*.py" --include="*.md" .
grep -r "Bearer" --include="*.py" --include="*.md" .

# Check for hardcoded tokens
grep -rE "[a-zA-Z0-9]{32,}" --include="*.py" .

# Verify .gitignore is working
git status --ignored
```

**Expected:**
- `.env` does not exist (only `.env.example`)
- No API keys in code (only in user's `.env`)
- `.gitignore` excludes `*.db`, `logs/`, `mcp.json`

## 2. Configuration Files

```bash
# Verify .env.example has placeholders, not real values
cat .env.example

# Should show:
# OBEKTCLAW_LLM_API_KEY=your-api-key-here
# Not: OBEKTCLAW_LLM_API_KEY=sk-actual-key
```

## 3. User Data Cleanup

```bash
# Remove any user-specific data from repo
rm -rf ~/.obektclaw  # If it's inside repo (shouldn't be)
rm -f *.db *.sqlite *.sqlite3
rm -rf logs/
rm -f *.jsonl

# Or set OBEKTCLAW_HOME to /tmp for testing
export OBEKTCLAW_HOME=/tmp/obektclaw-test
python -m obektclaw chat  # Test fresh install
```

## 4. Test Suite

```bash
# All tests must pass
python -m pytest -v

# Expected: 235 passed
```

## 5. Documentation Review

```bash
# Check all docs reference "obektclaw" not "obektclaw-mini"
grep -r "obektclaw-mini" --include="*.md" docs/
grep -r "Obektclaw-mini" --include="*.md" docs/

# Check CLI banners
python -m obektclaw --help  # Should show "obektclaw"
```

**Expected:**
- No "obektclaw-mini" references (upstream is credited as "Hermes Agent")
- CLI shows "obektclaw"

## 6. Import Paths

```bash
# Package is still 'obektclaw' (folder name)
python -c "import obektclaw; print(obektclaw.__version__)"

# But CLI uses 'obektclaw'
python -m obektclaw --help
```

**Note:** The folder is `obektclaw/` for import compatibility, but the CLI is `obektclaw`.

## 7. File Permissions

```bash
# Scripts should be executable (if any)
ls -la *.sh bin/

# Python files should be readable
find . -name "*.py" -exec ls -l {} \; | head
```

## 8. Git Status

```bash
# Clean working directory
git status

# Should show only tracked files
# No untracked .env, *.db, logs/, etc.
```

## 9. Commit Message

Use a clear, descriptive message:

```
Initial release: obektclaw v0.1.0

A minimal, self-improving AI agent implementation (~2,900 lines).

Features:
- Three-layer memory (session + persistent + 12-layer user model)
- Self-improving markdown skills
- 16 built-in tools + MCP bridge
- Learning Loop (retrospects after every turn)
- CLI + Telegram gateways
- 235 tests (all offline)

Based on the Nous Research Hermes Agent concept.
```

## 10. Post-Commit Test

```bash
# Clone fresh and test
cd /tmp
git clone <repo-url> obektclaw-fresh
cd obektclaw-fresh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with test credentials
python -m obektclaw chat
```

## Quick Command

Run all checks:

```bash
./scripts/release-check.sh  # (create this if needed)
```

---

**Golden rule:** If you wouldn't paste it on Twitter, don't commit it.
