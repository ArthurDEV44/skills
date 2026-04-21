---
name: api2cli
description: "Generate or discover a CLI + AgentSkill for any REST API. Use when: user says 'create a CLI for X', 'wrap this API', 'search if a CLI exists', 'install an existing CLI', 'make a skill for X', 'publish my CLI', 'publish to npm', 'push to github'. Handles registry search, install, discovery, scaffolding, resource implementation, building, linking, skill generation, npm publishing, and GitHub publishing. Do NOT use for non-REST-API CLI tools or general coding tasks."
---

# api2cli

Turn any REST API into a standardized, agent-ready CLI.

Always use `bunx api2cli` to run commands. Always use `--json` when calling generated CLIs programmatically.

## Prerequisites

```bash
bun --version || curl -fsSL https://bun.sh/install | bash
```

## Workflow

Follow all steps in order — do not skip any.

### 1. Search the registry first

Before generating anything, check if a CLI already exists:

```bash
bunx api2cli search <query>
```

If the registry already has the CLI you need, install it instead of rebuilding it:

```bash
bunx api2cli install <name>
```

See [references/commands.md](references/commands.md) for filters like `--type`, `--category`, and `--json`.

### 2. Discover the API

Find the API docs or OpenAPI spec. Identify: base URL, auth type, auth header, all resources and endpoints.

### 3. Create the scaffold

```bash
bunx api2cli create <app> --base-url <url> --auth-type bearer
```

See [references/create.md](references/create.md) for all flags and what gets generated.

### 4. Implement resources

Create `~/.cli/<app>-cli/src/resources/<resource>.ts` for each API resource. Register in `src/index.ts`.

See [references/resource-patterns.md](references/resource-patterns.md) for the CRUD template and library API.

### 5. Build, link, and test

```bash
bunx api2cli bundle <app>
bunx api2cli link <app>
<app>-cli --help
<app>-cli <resource> list --json
```

`api2cli link` adds `~/.local/bin` to PATH automatically. No `export PATH` needed.

### 6. Finalize skill and README

**This step is mandatory.** The skill must become a real operating guide for another agent, not just a command list — no placeholders, no TODOs.

1. **Introspect the CLI**: run `<app>-cli --help`, then `<resource> --help` and `<resource> <action> --help` for every resource and action
2. **Update the description**: list all resource names and add domain-specific trigger phrases
3. **Set the category**: replace `{{CATEGORY}}` with the correct value (e.g. `social-media`, `email`, `devtools`)
4. **Write the task-oriented sections**: replace `{{WHEN_TO_USE_HELP}}`, `{{CAPABILITIES_HELP}}`, and `{{USE_CASES_HELP}}` with domain-specific guidance
5. **Build resource tables**: for each resource, create a command table with every action and its real flags (from `--help` output)
6. **Add Quick Reference + Output Format**: include `--help` commands and document the JSON envelope
7. **Remove all placeholders**: no `{{...}}` or `<!-- TODO -->` should remain
8. **Validate**: run at least one command to confirm the skill's examples are accurate

See [references/skill-generation.md](references/skill-generation.md) for the full introspection workflow, format, quality checklist, and symlink instructions.

To also link skills for OpenClaw:

```bash
bunx api2cli link <app> --openclaw
```

See [references/openclaw.md](references/openclaw.md) for the one-prompt setup, ClawHub publishing, API key auto-detection, and custom `--skills-path` usage.

Also available on ClawHub: `bunx clawhub install api2cli`

### 7. Publish (when user asks)

Before any publish target, run these pre-flight checks:

1. **Verify `gh` CLI is authenticated**: run `gh auth status`. If not logged in, ask the user to run `gh auth login` first. **Stop and wait.**
2. **Check if the CLI is already on GitHub**: run `git remote get-url origin` in the CLI directory.
   - If no remote exists → the CLI is not on GitHub yet. **Automatically run the GitHub publish flow first** (see below) before proceeding to npm or registry publish.
   - If a remote exists → already on GitHub, continue.

#### To GitHub

Push the CLI to a public GitHub repo.

See [references/publish-to-github.md](references/publish-to-github.md) for pre-flight checks, repo creation, and push workflow.

#### To npm

Requires the CLI to be on GitHub first (for `repository` field in package.json). If not on GitHub, run the GitHub publish flow above first.

Publish the CLI to the npm registry so users can `bun add -g <name>` or `bunx <name>`.

See [references/publish-to-npm.md](references/publish-to-npm.md) for auth, package.json validation, build, verify, and publish workflow. Also see [references/package-checklist.md](references/package-checklist.md) for the field-by-field package.json reference.

#### To ClawHub

Publish the generated skill to ClawHub so OpenClaw users can discover and install it.

1. **Auth**: run `bunx clawhub login`. If not authenticated, ask the user to complete login first. **Stop and wait.**
2. **Push the skill**: run `bunx clawhub publish <cli-dir>/skills/<app>-cli --slug <app>-cli`
3. After publish, users can install with:
   ```bash
   bunx clawhub install <app>-cli
   ```

See [references/openclaw.md](references/openclaw.md) for the full OpenClaw setup workflow.

#### To Sundial Hub

Publish the generated skill to the Sundial Hub so any agent (Claude Code, Cursor, Codex, etc.) can install it.

1. **Auth**: run `bunx sundial-hub auth status`. If not authenticated, ask the user to run `bunx sundial-hub auth login` first. **Stop and wait.**
2. **Push the skill**: run `bunx sundial-hub push <cli-dir>/skills/<app>-cli --visibility public --categories coding`
3. After publish, users can install with:
   ```bash
   bunx sundial-hub add <sundial-username>/<app>-cli
   ```

#### To api2cli.dev registry

Requires the CLI to be on GitHub first. If not on GitHub, run the GitHub publish flow above first.

Register on api2cli.dev so others can install with `bunx api2cli install <name>`.

See [references/publish.md](references/publish.md) for the registry publish workflow.

## Conventions

```
<app>-cli <resource> <action> [flags]
<app>-cli auth set|show|remove|test
```

`--json` returns: `{ "ok": true, "data": [...], "meta": { "total": 42 } }`

Other flags: `--format <text|json|csv|yaml>`, `--verbose`, `--no-color`, `--no-header`

Exit codes: 0 = success, 1 = API error, 2 = usage error

Tokens: `~/.config/tokens/<app>-cli.txt` (chmod 600)

## Hard Rules

- ALWAYS search the registry (`bunx api2cli search`) before generating a new CLI.
- ALWAYS use `bunx api2cli` to run commands — NEVER install api2cli globally.
- ALWAYS use `--json` when calling generated CLIs programmatically.
- ALWAYS run the finalize step (step 6) — a skill with placeholders or TODOs is incomplete.
- NEVER skip the `bundle` + `link` cycle before testing.
- NEVER publish to npm or registry without confirming the CLI is on GitHub first.
- NEVER guess commands or flags — introspect with `--help` to get the source of truth.

## Error Handling

| Scenario | Action |
|----------|--------|
| `bun --version` fails | Install bun: `curl -fsSL https://bun.sh/install \| bash` |
| `bunx api2cli search` returns no results | Proceed to step 2 (discover + create). |
| `bunx api2cli create` fails | Check that `--base-url` is a valid URL and `--auth-type` is one of `bearer`, `basic`, `api-key`, `none`. |
| `bunx api2cli bundle` fails with TypeScript errors | Fix the resource file indicated in the error output, then re-run `bundle`. |
| `bunx api2cli link` fails | Verify `~/.local/bin` exists. Run `mkdir -p ~/.local/bin` if missing. |
| `gh auth status` not authenticated | Ask the user to run `gh auth login`. Stop and wait. |
| `npm whoami` not authenticated | Ask the user to run `npm login`. Stop and wait. |
| Skill still has `{{...}}` placeholders after step 6 | Re-run introspection (`--help` for every resource/action) and replace all remaining placeholders. |

## Done When

- [ ] Registry searched — existing CLI reused or new one scaffolded
- [ ] All API resources implemented in `src/resources/` and registered in `src/index.ts`
- [ ] `bunx api2cli bundle` succeeds without errors
- [ ] `bunx api2cli link` succeeds and `<app>-cli --help` works
- [ ] At least one resource command tested (e.g., `<app>-cli <resource> list --json`)
- [ ] Skill file finalized — zero `{{...}}` placeholders, all resource tables populated
- [ ] Published (only if user requested)
