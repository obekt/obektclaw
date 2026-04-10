---
name: deployment-checklist
description: Pre-flight checks before pushing a service to a server
---

# Before deploying
- [ ] Tests pass locally
- [ ] Lockfile committed (requirements.txt / package-lock.json / etc.)
- [ ] Secrets are *not* in the repo — only in env vars or a secret store
- [ ] Health-check endpoint exists and returns 200 quickly
- [ ] Logs go to stdout/stderr (not a file inside the container)
- [ ] Migrations are idempotent and reversible
- [ ] Rollback plan is one command, not "redeploy the previous tag from memory"

# After deploying
- [ ] Smoke test the public URL
- [ ] Check error rate for 10 minutes
- [ ] Confirm metrics are flowing
- [ ] Note the deploy in the project's daily log

# How to apply
- When the user says "deploy", "ship", "push to prod", or anything similar,
  load this and walk through the relevant items. Skip steps that don't apply
  to the project (e.g. no migrations) but mention which ones you skipped.
