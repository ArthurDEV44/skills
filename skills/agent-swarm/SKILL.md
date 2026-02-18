---
name: agent-swarm
description: "Best practices for orchestrating Claude Code agent teams (swarm mode). Use when: (1) the user asks to create an agent team, swarm, or multi-agent setup, (2) spawning teammates for parallel work, (3) coordinating multiple Claude Code instances, (4) the user says 'create a team', 'agent team', 'swarm', 'spawn teammates', 'multi-agent', or 'parallel agents'. Covers setup, 4 orchestration patterns (Leader, Swarm, Pipeline, Watchdog), TeamCreate/SendMessage/Task tools, display modes (tmux/iTerm2/in-process), communication (DM/broadcast/shutdown), hooks (TaskCompleted/TeammateIdle), subagent types, and cleanup."
---

# Agent Swarm — Claude Code Agent Teams

## Setup

Enable agent teams by setting the environment variable in `~/.claude/settings.json`:

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

## Architecture

An agent team has 4 components:

| Component | Role |
|-----------|------|
| **Team lead** | Primary session — creates the team, spawns teammates, coordinates work |
| **Teammates** | Separate Claude Code instances, each working on assigned tasks |
| **Task list** | Shared work items teammates claim and complete |
| **Mailbox** | Messaging system for inter-agent communication |

**File locations:**
- Team config: `~/.claude/teams/{team-name}/config.json` (members array with name, agentId, agentType)
- Task list: `~/.claude/tasks/{team-name}/`

Task dependencies resolve automatically — when a teammate finishes a blocking task, dependent tasks unblock without manual intervention.

## Pattern Selection

Pick the right pattern based on task structure:

| Pattern | When | Example |
|---------|------|---------|
| **Leader** | One coordinator distributes independent subtasks | Feature with backend + frontend + tests |
| **Swarm** | N similar tasks, no cross-dependencies | Migrate 10 route files, review 5 modules |
| **Pipeline** | Sequential stages where each depends on the previous | Research → Design → Implement → Review |
| **Watchdog** | Parallel work needs continuous quality oversight | Implementation + live code reviewer |

Combine patterns for complex projects (e.g., Leader + Watchdog, Pipeline + Swarm).

## Team Lifecycle

### 1. Create Team

Use `TeamCreate` to initialize a team with a name and optional description:

```
TeamCreate: team_name="my-feature", description="Build the auth feature"
```

### 2. Create Tasks

Use `TaskCreate` for each work item. Include `subject`, `description`, and `activeForm`:

```
TaskCreate: subject="Implement API endpoints", description="...", activeForm="Implementing API endpoints"
```

Set dependencies with `TaskUpdate`:

```
TaskUpdate: taskId="2", addBlockedBy=["1"]
```

### 3. Spawn Teammates

Use the `Task` tool with `team_name` and `name` to create teammates. Choose `subagent_type` based on what tools the teammate needs:

| Subagent Type | Tools | Best For |
|---------------|-------|----------|
| **general-purpose** | All tools (Edit, Write, Bash, etc.) | Implementation, coding, full-stack work |
| **Explore** | Read-only (Glob, Grep, Read) | Research, codebase analysis, auditing |
| **Plan** | Read-only | Architecture design, planning |
| **Bash** | Bash only | Command execution, git ops, builds |

Assign tasks with `TaskUpdate`:

```
TaskUpdate: taskId="1", owner="backend-dev"
```

### 4. Communication

Use `SendMessage` to communicate with teammates:

- **DM** (`type: "message"`): Send to a specific teammate by name. Default for most communication.
- **Broadcast** (`type: "broadcast"`): Send to ALL teammates. Use sparingly — costs scale linearly with team size. Only for critical team-wide issues.
- **Shutdown request** (`type: "shutdown_request"`): Ask a teammate to gracefully shut down.

Teammates respond to shutdown requests with `type: "shutdown_response"` (approve or reject).

Always refer to teammates **by name**, not by UUID.

### 5. Cleanup

Always follow this sequence:
1. Send shutdown requests to all teammates via `SendMessage`
2. Wait for all shutdown responses
3. Call `TeamDelete` to remove team and task directories
4. If using tmux, verify with `tmux ls` — kill any orphaned sessions

## Display Modes

| Mode | Setting | Requirements | Notes |
|------|---------|--------------|-------|
| **In-process** | `"in-process"` | Any terminal | All teammates in main terminal. Use `Shift+Up/Down` to select. |
| **Auto** (default) | `"auto"` | — | Uses split panes if inside tmux, in-process otherwise |
| **Split panes** | `"tmux"` | tmux or iTerm2 | Each teammate gets its own pane. Click to interact. |

Configure in `~/.claude/settings.json`:

```json
{ "teammateMode": "tmux" }
```

Or per-session: `claude --teammate-mode in-process`

Split panes are **not supported** in VS Code integrated terminal, Windows Terminal, or Ghostty.

## Spawning Rules

1. **Right-size the team** — 2-5 teammates is optimal. Each costs a full context window. Coordination overhead grows with team size.

2. **Assign explicit file ownership** — each teammate owns distinct files. Never let two teammates edit the same file.

```
# GOOD: clear ownership
- "backend": owns src/api/ and src/services/
- "frontend": owns src/components/ and src/pages/

# BAD: overlapping ownership
- "dev-1": works on auth (touches src/api/auth.ts AND src/components/Login.tsx)
- "dev-2": works on UI (also touches src/components/Login.tsx)
```

3. **Give rich spawn context** — teammates don't inherit the lead's conversation. Include in the spawn prompt: exact file paths, tech stack, acceptance criteria, and project conventions.

4. **Specify model per teammate** — use `model: "sonnet"` for routine tasks (migrations, tests, formatting), `model: "opus"` for complex reasoning (architecture, security).

5. **Use plan approval for risky tasks** — spawn teammates with `mode: "plan"` to require plan approval before making changes.

6. **Use delegate mode** — press `Shift+Tab` to prevent the lead from coding and keep it as a pure orchestrator.

## Task Management

- Create **5-6 tasks per teammate** for continuous flow
- Set **task dependencies** with `addBlockedBy` for pipeline stages
- Teammates should check `TaskList` after completing each task to find next work
- Prefer tasks **in ID order** (lowest first) — earlier tasks set context for later ones
- Mark tasks `in_progress` before starting, `completed` when done

## Teammate Idle State

Teammates go idle after every turn — this is **completely normal**. A teammate sending a message then going idle is the standard flow.

- Idle teammates **can receive messages** — sending a message wakes them up
- Do **not** treat idle as an error or finished state
- Do **not** comment on idleness unless it impacts work

## Hooks

Two hooks enable quality gates for teams. See [references/hooks-and-advanced.md](references/hooks-and-advanced.md) for implementation details.

- **TaskCompleted**: runs when a task is marked completed. Exit code 2 prevents completion with feedback.
- **TeammateIdle**: runs when a teammate is about to go idle. Exit code 2 keeps the teammate working.

## Common Pitfalls

- **Lead starts coding instead of delegating** → tell it: "Wait for teammates to complete their tasks", or enable delegate mode (`Shift+Tab`)
- **File conflicts** → always assign explicit file ownership before spawning
- **Teammates lack context** → enrich spawn prompts with file paths, stack info, and criteria
- **Too many teammates** → more agents ≠ faster; 2-5 is optimal
- **Broadcast overuse** → use DMs for most communication; broadcast only for critical team-wide issues
- **Orphaned sessions** → after cleanup, verify with `tmux ls` and kill stale sessions

## Limitations

- **No session resumption** for in-process teammates — `/resume` and `/rewind` don't restore them
- **Task status can lag** — teammates sometimes fail to mark tasks completed, blocking dependents
- **Slow shutdown** — teammates finish current request before shutting down
- **One team per session** — clean up current team before starting a new one
- **No nested teams** — only the lead manages the team
- **Fixed lead** — cannot promote a teammate or transfer leadership
- **Permissions at spawn** — all teammates start with the lead's permission mode; individual modes can change later
- **Split panes require tmux or iTerm2** — not supported in VS Code terminal, Windows Terminal, or Ghostty

## References

- [Prompt Templates](references/prompt-templates.md) — ready-to-use prompts for all 4 patterns and combined workflows
- [Tools Reference](references/tools-reference.md) — detailed tool API for TeamCreate, SendMessage, TaskCreate, TaskUpdate, TaskList, TaskGet, TeamDelete
- [Hooks and Advanced Patterns](references/hooks-and-advanced.md) — TaskCompleted/TeammateIdle hooks, advanced coordination patterns, display mode setup
