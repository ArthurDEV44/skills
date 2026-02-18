# Agent Team Tools Reference

Complete reference for all tools used in agent team orchestration.

## TeamCreate

Creates a new team with a shared task list.

**Parameters:**
| Parameter | Required | Description |
|-----------|----------|-------------|
| `team_name` | Yes | Name for the team (used in file paths) |
| `description` | No | Team purpose/description |
| `agent_type` | No | Type/role of the team lead |

**Creates:**
- Team config at `~/.claude/teams/{team-name}/config.json`
- Task list at `~/.claude/tasks/{team-name}/`

**Config file structure:**
```json
{
  "members": [
    {
      "name": "backend-dev",
      "agentId": "abc123",
      "agentType": "general-purpose"
    }
  ]
}
```

## TeamDelete

Removes team and task directories. Fails if the team still has active members — send shutdown requests and wait for all teammates to shut down first.

**Parameters:** None (uses current session's team context).

## SendMessage

Send messages between teammates. Always use teammate **names**, not UUIDs.

### type: "message" — Direct Message

Send to a single teammate.

```json
{
  "type": "message",
  "recipient": "backend-dev",
  "content": "The API schema has changed, update your endpoints",
  "summary": "API schema update notification"
}
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `type` | Yes | `"message"` |
| `recipient` | Yes | Teammate name |
| `content` | Yes | Message text |
| `summary` | Yes | 5-10 word preview for UI |

### type: "broadcast" — Message All Teammates

Sends the same message to every teammate. **Use sparingly** — costs scale linearly.

```json
{
  "type": "broadcast",
  "content": "Stop all work — blocking bug found in shared module",
  "summary": "Critical blocking issue found"
}
```

Valid broadcast use cases:
- Critical issues requiring immediate team-wide attention
- Major announcements affecting every teammate equally

Use DMs for everything else.

### type: "shutdown_request"

Ask a teammate to gracefully shut down.

```json
{
  "type": "shutdown_request",
  "recipient": "backend-dev",
  "content": "All tasks complete, wrapping up"
}
```

### type: "shutdown_response"

Teammate responds to a shutdown request. Extract `requestId` from the incoming JSON.

**Approve:**
```json
{
  "type": "shutdown_response",
  "request_id": "abc-123",
  "approve": true
}
```

**Reject:**
```json
{
  "type": "shutdown_response",
  "request_id": "abc-123",
  "approve": false,
  "content": "Still working on task #3, need more time"
}
```

### type: "plan_approval_response"

Approve or reject a teammate's plan when they use `ExitPlanMode`.

**Approve:**
```json
{
  "type": "plan_approval_response",
  "request_id": "abc-123",
  "recipient": "architect",
  "approve": true
}
```

**Reject with feedback:**
```json
{
  "type": "plan_approval_response",
  "request_id": "abc-123",
  "recipient": "architect",
  "approve": false,
  "content": "Add error handling for the API calls"
}
```

## TaskCreate

Create a work item in the shared task list.

```json
{
  "subject": "Implement user authentication",
  "description": "Add JWT auth endpoints to src/api/auth.ts with login, register, refresh, and logout. Use bcrypt for passwords. Write tests in tests/auth.test.ts.",
  "activeForm": "Implementing user authentication"
}
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `subject` | Yes | Imperative title (e.g., "Run tests") |
| `description` | Yes | Detailed requirements and context |
| `activeForm` | Recommended | Present continuous form shown in spinner (e.g., "Running tests") |

Tasks are created with status `pending` and no owner.

## TaskUpdate

Update task status, ownership, or dependencies.

**Assign to teammate:**
```json
{ "taskId": "1", "owner": "backend-dev" }
```

**Start work:**
```json
{ "taskId": "1", "status": "in_progress" }
```

**Mark complete:**
```json
{ "taskId": "1", "status": "completed" }
```

**Set dependencies:**
```json
{ "taskId": "2", "addBlockedBy": ["1"] }
```

**Delete task:**
```json
{ "taskId": "1", "status": "deleted" }
```

| Parameter | Description |
|-----------|-------------|
| `taskId` | Required. Task ID to update. |
| `status` | `pending` → `in_progress` → `completed`, or `deleted` |
| `owner` | Teammate name to assign |
| `addBlockedBy` | Task IDs that must complete first |
| `addBlocks` | Task IDs that depend on this one |
| `subject` | Update title |
| `description` | Update description |
| `activeForm` | Update spinner text |

## TaskList

List all tasks in the shared task list. Returns summary with id, subject, status, owner, and blockedBy for each task.

Use after completing a task to find next available work. Prefer tasks in ID order.

## TaskGet

Fetch full details for a single task by ID. Returns subject, description, status, blocks, and blockedBy.

Use before starting work to verify requirements and check that blockedBy is empty.

## Spawning Teammates with Task Tool

Use the `Task` tool to spawn teammates into the team:

```json
{
  "prompt": "Implement the REST API endpoints for user management in src/api/users.ts. Use Express with TypeScript. Write tests in tests/users.test.ts. Follow the existing patterns in src/api/auth.ts.",
  "description": "Implement user API",
  "subagent_type": "general-purpose",
  "team_name": "my-feature",
  "name": "backend-dev",
  "mode": "plan",
  "model": "sonnet"
}
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | Yes | Detailed task instructions with full context |
| `description` | Yes | Short 3-5 word summary |
| `subagent_type` | Yes | Agent type (determines available tools) |
| `team_name` | Yes | Must match the TeamCreate name |
| `name` | Yes | Human-readable teammate name |
| `mode` | No | `"plan"` to require plan approval, `"bypassPermissions"` for autonomous work |
| `model` | No | `"sonnet"` for routine work, `"opus"` for complex reasoning |
