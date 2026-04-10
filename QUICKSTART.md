# Quick Start Guide

## First Time Setup

### 1. Install and Run

```bash
cd obektclaw
source .venv/bin/activate
python -m obektclaw start
```

### 2. What You'll See

On first run, you'll see a welcome message explaining what obektclaw can do:

```
╔═══════════════════════════════════════════════════════════╗
║              Welcome to obektclaw! 🤖                     ║
╚═══════════════════════════════════════════════════════════╝

I'm a self-improving AI agent. Here's what I can do:

1. REMEMBER: Tell me your preferences and I'll remember them
   → "I always use httpx instead of requests"

2. EXECUTE: I can run commands and manipulate files
   → "List all Python files in this directory"

3. LEARN: I create skills from successful patterns
   → After helping you deploy, I'll save it as a skill

4. CONNECT: Optional integrations
   → Run /setup to configure Telegram or MCP servers
```

### 3. Try These Examples

```
you> What can you do?
you> Remember that I prefer concise answers
you> List Python files in the obektclaw directory
you> What skills do you have?
you> /traits
you> /help
you> /setup
you> /exit
```

## Gateway Auto-Detection

obektclaw automatically detects which gateways are available:

- **CLI** — Always available
- **Telegram** — Starts automatically if `OBEKTCLAW_TG_TOKEN` is set
- **Both** — Run simultaneously so you can chat from either place

```bash
python -m obektclaw start          # Auto-detect (recommended)
python -m obektclaw start cli      # CLI only
python -m obektclaw start tg       # Telegram only
```

*(Legacy `python -m obektclaw chat` and `python -m obektclaw tg` still work.)*

## Commands Reference

### In Chat

| Command | Description |
|---------|-------------|
| `/help` | Show detailed help with examples |
| `/skills` | List all available skills |
| `/memory <query>` | Search your persistent memories |
| `/traits` | Show what obektclaw learned about you |
| `/sessions` | Browse and resume past sessions |
| `/setup` | Configuration wizard |
| `/exit` | Exit the chat |

### CLI Commands

| Command | Description |
|---------|-------------|
| `python -m obektclaw start` | Start obektclaw (auto-detects gateways) |
| `python -m obektclaw setup` | Run setup wizard |
| `python -m obektclaw sessions list` | List recent sessions |
| `python -m obektclaw sessions show <id>` | Show session details |
| `python -m obektclaw sessions export <id>` | Export session (md or json) |
| `python -m obektclaw sessions resume <id>` | Resume a past session |
| `python -m obektclaw skill list` | List skills |
| `python -m obektclaw memory search <q>` | Search memory |
| `python -m obektclaw traits` | Show user model |
| `python -m obektclaw --help` | Show all commands |

## Optional: Telegram Setup

Want to chat with obektclaw on Telegram?

1. **Get a bot token:**
   - Open Telegram and message `@BotFather`
   - Send: `/newbot`
   - Follow the prompts
   - Copy the API token

2. **Configure obektclaw:**
   ```bash
   # Edit .env in the project root
   OBEKTCLAW_TG_TOKEN=your_token_here
   ```

3. **Start obektclaw — Telegram starts automatically:**
   ```bash
   python -m obektclaw start
   ```

4. **Message your bot on Telegram!**

To run Telegram-only mode: `python -m obektclaw start tg`

## Optional: MCP Servers

Connect external tools via MCP (Model Context Protocol):

1. Create `~/.obektclaw/mcp.json`:
   ```json
   {
     "mcpServers": {
       "filesystem": {
         "command": "npx",
         "args": ["@modelcontextprotocol/server-filesystem", "/tmp"]
       }
     }
   }
   ```

2. Obektclaw will auto-load these tools on next start.

## What Makes Obektclaw Special?

### 🧠 Three-Layer Memory
- **Session memory**: Our conversation history
- **Persistent facts**: Long-term knowledge (survives restarts)
- **User model**: 12-layer profile of your preferences and goals

### 🛠️ 16 Built-in Tools
- File operations (read, write, list, grep)
- Command execution (bash, Python)
- Web fetching
- Memory management
- Skill system

### 📚 Self-Improving Skills
- Auto-creates skills from successful patterns
- Improves existing skills after each use
- Markdown-based (you can edit them!)

### 🔍 FTS5 Search
- Full-text search across all memories
- Stemming and fuzzy matching
- Finds relevant context automatically

## Configuration

All config via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OBEKTCLAW_HOME` | `~/.obektclaw` | Where data is stored |
| `OBEKTCLAW_LLM_BASE_URL` | OpenAI API | LLM endpoint |
| `OBEKTCLAW_LLM_API_KEY` | (required) | API key |
| `OBEKTCLAW_LLM_MODEL` | `gpt-4o-mini` | Main model |
| `OBEKTCLAW_TG_TOKEN` | (empty) | Telegram bot token |

## Need Help?

```bash
# General help
python -m obektclaw --help

# Setup wizard
python -m obektclaw setup

# In chat
/help
```

## Testing

Run the test suite:

```bash
python -m pytest
```

All 326 tests should pass.
