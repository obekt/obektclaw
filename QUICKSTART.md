# Quick Start Guide

## First Time Setup

### 1. Install and Run

```bash
cd obektclaw
source .venv/bin/activate
python -m obektclaw chat
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

## Commands Reference

### In Chat (`python -m obektclaw chat`)

| Command | Description |
|---------|-------------|
| `/help` | Show detailed help with examples |
| `/skills` | List all available skills |
| `/memory <query>` | Search your persistent memories |
| `/traits` | Show what Obektclaw learned about you |
| `/setup` | Configuration wizard |
| `/exit` | Exit the chat |

### CLI Commands

| Command | Description |
|---------|-------------|
| `python -m obektclaw chat` | Start interactive chat |
| `python -m obektclaw setup` | Run setup wizard |
| `python -m obektclaw tg` | Start Telegram bot |
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
   
   # Or run setup wizard
   python -m obektclaw setup
   ```

3. **Start the bot:**
   ```bash
   python -m obektclaw tg
   ```

4. **Message your bot on Telegram!**

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

All 235 tests should pass.
