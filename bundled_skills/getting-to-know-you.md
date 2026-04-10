---
name: getting-to-know-you
description: Onboarding skill — used in early conversations to build the user model
---

# Goal
On the first few conversations with a new user, gently learn enough about them
to populate the user model. Don't run a survey — pick up signals from natural
conversation and only ask a question if it would unblock the current task.

# What to listen for
- What language(s) and tools they reach for first
- How verbose / terse they want responses
- Whether they prefer to see a result first or an explanation first
- The domain they keep coming back to (data eng, web dev, ML, etc.)
- The directories / projects they actually work in

# How to apply
- After the conversation, the Learning Loop will write inferences into the
  user_model. You don't need to write them yourself during the chat.
- If you do learn an explicit preference ("I always use httpx, not requests"),
  call `memory_set_fact` with category="preference" so it persists immediately.
