# Hooks and Advanced Patterns

## TaskCompleted Hook

Runs when a task is being marked as completed — either explicitly via `TaskUpdate` or when a teammate finishes its turn with in-progress tasks.

**Use case:** enforce completion criteria (passing tests, lint checks, build verification) before a task can close.

**Exit codes:**
- `0` — criteria met, task marked completed
- `2` — criteria failed, task stays in progress, stderr message fed back to the model

**Hook configuration** (in `settings.json` or `.claude/settings.json`):

```json
{
  "hooks": {
    "TaskCompleted": [
      {
        "command": "bash .claude/hooks/task-completed.sh"
      }
    ]
  }
}
```

**Example hook script:**

```bash
#!/bin/bash
INPUT=$(cat)
TASK_SUBJECT=$(echo "$INPUT" | jq -r '.task_subject')

# Run the test suite
if ! npm test 2>&1; then
  echo "Tests not passing. Fix failing tests before completing: $TASK_SUBJECT" >&2
  exit 2
fi

exit 0
```

**Input fields available:**

| Field | Description |
|-------|-------------|
| `session_id` | Session identifier |
| `hook_event_name` | Always `TaskCompleted` |
| `task_id` | Task identifier |
| `task_subject` | Task title |
| `task_description` | Detailed description (may be absent) |
| `teammate_name` | Name of completing teammate (may be absent) |
| `team_name` | Team name (may be absent) |
| `cwd` | Current working directory |
| `permission_mode` | Permission mode |

## TeammateIdle Hook

Runs when a teammate is about to go idle after finishing its turn.

**Use case:** enforce quality gates before a teammate stops working (lint checks, output file verification, build artifact checks).

**Exit codes:**
- `0` — quality gates passed, teammate goes idle
- `2` — gate failed, teammate continues working with stderr feedback

**Example hook script:**

```bash
#!/bin/bash
INPUT=$(cat)
TEAMMATE=$(echo "$INPUT" | jq -r '.teammate_name')

# Verify build artifact exists
if [ ! -f "./dist/output.js" ]; then
  echo "Build artifact missing. Run the build before stopping." >&2
  exit 2
fi

# Verify no uncommitted changes
if ! git diff --quiet; then
  echo "Uncommitted changes detected. Commit your work before going idle." >&2
  exit 2
fi

exit 0
```

**Input fields available:**

| Field | Description |
|-------|-------------|
| `session_id` | Session identifier |
| `hook_event_name` | Always `TeammateIdle` |
| `teammate_name` | Name of the teammate going idle |
| `team_name` | Team name |
| `cwd` | Current working directory |
| `permission_mode` | Permission mode |

## Display Mode Setup

### tmux Split Panes

Install tmux via your package manager, then launch Claude inside a tmux session:

```bash
tmux new -s claude-team "claude"
```

Or set `teammateMode` in settings:

```json
{ "teammateMode": "tmux" }
```

Each teammate gets its own pane. Click a pane to interact directly with that teammate.

### iTerm2 Split Panes

Install the [`it2` CLI](https://github.com/mkusaka/it2), then enable the Python API in **iTerm2 → Settings → General → Magic → Enable Python API**.

The `"tmux"` setting auto-detects whether to use tmux or iTerm2 based on your terminal.

### In-Process Mode

All teammates run inside the main terminal. No extra setup needed.

- `Shift+Up/Down` — select a teammate
- Type to message the selected teammate

Force for a single session: `claude --teammate-mode in-process`

## Advanced Coordination Patterns

### Plan Approval Gate

For risky tasks, spawn teammates in plan mode. They must get approval before making changes:

```
Task: {
  prompt: "Refactor the database schema in src/db/",
  subagent_type: "general-purpose",
  team_name: "my-team",
  name: "db-refactor",
  mode: "plan"
}
```

When the teammate calls `ExitPlanMode`, the lead receives a `plan_approval_request`. Approve or reject with `SendMessage` type `"plan_approval_response"`.

### Peer Communication

Teammates can send DMs to each other directly — they don't need to go through the lead. When a teammate sends a DM to another teammate, a brief summary is included in the lead's idle notification for visibility.

### Discovering Team Members

Any teammate can read the team config to discover other members:

```
Read: ~/.claude/teams/{team-name}/config.json
```

The config contains a `members` array with each teammate's name, agentId, and agentType.

### Tmux-Based Multi-Agent Swarm (Legacy Pattern)

For manual orchestration outside the built-in team tools, you can use tmux sessions with state files. This approach uses a `.claude/multi-agent-swarm.local.md` file with YAML frontmatter per agent:

```markdown
---
agent_name: auth-implementation
task_number: 3.5
pr_number: 1234
coordinator_session: team-leader
enabled: true
dependencies: ["Task 3.4"]
additional_instructions: "Use JWT tokens, not sessions"
---

# Task: Implement Authentication

Build JWT-based authentication for the REST API.
```

A hook script can send idle notifications to the coordinator via `tmux send-keys`:

```bash
#!/bin/bash
SWARM_STATE_FILE=".claude/multi-agent-swarm.local.md"
[ ! -f "$SWARM_STATE_FILE" ] && exit 0

FRONTMATTER=$(sed -n '/^---$/,/^---$/{ /^---$/d; p; }' "$SWARM_STATE_FILE")
COORDINATOR_SESSION=$(echo "$FRONTMATTER" | grep '^coordinator_session:' | sed 's/coordinator_session: *//')
AGENT_NAME=$(echo "$FRONTMATTER" | grep '^agent_name:' | sed 's/agent_name: *//')
ENABLED=$(echo "$FRONTMATTER" | grep '^enabled:' | sed 's/enabled: *//')

[ "$ENABLED" != "true" ] && exit 0

if tmux has-session -t "$COORDINATOR_SESSION" 2>/dev/null; then
  tmux send-keys -t "$COORDINATOR_SESSION" "Agent ${AGENT_NAME} is idle." Enter
fi
```

This legacy pattern gives you full control over agent lifecycle but requires more manual setup than the built-in TeamCreate/SendMessage tools.
