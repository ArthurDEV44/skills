# Skills

Collection of AI coding agent skills by [ArthurDEV44](https://github.com/ArthurDEV44).

## Available Skills

### Rust Ecosystem

| Skill | Description |
|-------|-------------|
| [rust-best-practices](skills/rust-best-practices) | Idiomatic patterns, API design, Cargo tooling, Clippy, performance, documentation |
| [rust-traits](skills/rust-traits) | Trait system best practices, patterns, and idiomatic usage |
| [rust-ownership](skills/rust-ownership) | Ownership, borrowing, lifetimes, subtyping, and variance |
| [rust-async](skills/rust-async) | Async/await patterns, futures, streams, and runtime usage |
| [rust-arc](skills/rust-arc) | Thread-safe shared ownership with Arc and reference counting |
| [rust-concurrency](skills/rust-concurrency) | Threads, message passing, shared state, Send/Sync, atomics |
| [rust-tokio](skills/rust-tokio) | Tokio runtime: task spawning, channels, I/O, select!, shutdown |
| [rust-axum](skills/rust-axum) | Axum 0.8 web framework: routing, extractors, handlers, middleware, SSE |
| [rust-seaorm](skills/rust-seaorm) | SeaORM async ORM: entities, CRUD, relations, transactions |
| [rust-crypto](skills/rust-crypto) | RustCrypto: AES-GCM, ChaCha20, HMAC, SHA-2, HKDF, Argon2, Ed25519, X25519 |
| [clerk-rs-sdk](skills/clerk-rs-sdk) | clerk-rs Rust SDK: typed API calls, JWT validation, framework middleware |
| [async-stripe](skills/async-stripe) | async-stripe Rust crate: OpenAPI codegen, typed Stripe API bindings |

### C

| Skill | Description |
|-------|-------------|
| [c-best-practices](skills/c-best-practices) | Modern C (C11/C17/C23): memory safety, pointers, structs, error handling, strings, preprocessor, threads, atomics, sanitizers |

### Go

| Skill | Description |
|-------|-------------|
| [go-best-practices](skills/go-best-practices) | Go idioms and modern patterns: error handling, interfaces, concurrency, generics, testing, project structure |

### Python

| Skill | Description |
|-------|-------------|
| [python-best-practices](skills/python-best-practices) | Idiomatic Python 3.10+: type hints, dataclasses, error handling, generators, context managers, decorators, enums, pattern matching, pytest, async/await, packaging |

### Frontend & Web

| Skill | Description |
|-------|-------------|
| [angular-best-practices](skills/angular-best-practices) | Angular v20+: standalone components, signals, DI, routing, forms, HttpClient, SSR, zoneless |
| [primeng](skills/primeng) | PrimeNG v20: 80+ Angular UI components, theming, design tokens, Table, Dialog, Forms |
| [next-best-practices](skills/next-best-practices) | Next.js App Router (v14/v15/v16+): file conventions, RSC boundaries, data fetching, caching, metadata, error handling, Server Actions, middleware, image/font optimization, bundling |
| [clerk-best-practices](skills/clerk-best-practices) | Clerk auth for Next.js: setup, middleware, Server Components, Server Actions, API routes, organizations, RBAC, webhooks, caching, custom UI, testing |
| [tailwind-best-practices](skills/tailwind-best-practices) | Tailwind CSS v4: CSS-first @theme, @utility, @custom-variant, responsive, dark mode, animations, 3D transforms, v3-to-v4 migration, Next.js integration |
| [fumadocs](skills/fumadocs) | Fumadocs documentation framework for Next.js: MDX, layouts, UI components, search, OpenAPI, i18n |
| [drizzle-orm](skills/drizzle-orm) | Drizzle ORM: PostgreSQL schemas, queries, relations, migrations, Neon serverless |
| [neon-best-practices](skills/neon-best-practices) | Neon serverless Postgres for Next.js: driver setup, connection pooling, branching, autoscaling, RLS, Neon Auth |
| [tanstack-query](skills/tanstack-query) | TanStack Query v5: data fetching, caching, mutations, and SSR |
| [tanstack-form](skills/tanstack-form) | TanStack Form v1: type-safe forms, validation, arrays, SSR |
| [tanstack-table](skills/tanstack-table) | TanStack Table v8: headless tables, sorting, filtering, pagination, selection |
| [tanstack-store](skills/tanstack-store) | TanStack Store: framework-agnostic reactive state with Store, Derived, Effect |
| [coss-ui](skills/coss-ui) | Component library built on Base UI and Tailwind CSS for React |

### 3D Graphics & Shaders

| Skill | Description |
|-------|-------------|
| [react-three-fiber](skills/react-three-fiber) | React Three Fiber: Canvas, hooks, events, drei, performance, models, portals |
| [web-3d-shaders](skills/web-3d-shaders) | Shaders & real-time 3D on the Web: GLSL, Three.js, R3F, WebGPU |
| [tsl-webgpu](skills/tsl-webgpu) | TSL (Three.js Shading Language) and WebGPU: shaders, compute, node materials |
| [volumetric-lighting](skills/volumetric-lighting) | Volumetric lighting with post-processing raymarching for R3F/Three.js |
| [caustics-r3f](skills/caustics-r3f) | Real-time caustic light effects: GLSL shaders, render targets, refraction for R3F |
| [post-processing-shaders](skills/post-processing-shaders) | Creative post-processing: pixelation, cell patterns, optical illusions for R3F |
| [painterly-kuwahara-shader](skills/painterly-kuwahara-shader) | Painterly Kuwahara filter: watercolor, oil paint, gouache effects for R3F |
| [moebius-post-processing](skills/moebius-post-processing) | Moebius-style NPR: hand-drawn outlines, crosshatched shadows for R3F |
| [retro-dithering-crt](skills/retro-dithering-crt) | Retro dithering, color quantization, CRT effects: Bayer, scanlines for R3F |

### GPU / CUDA

| Skill | Description |
|-------|-------------|
| [cuda-best-practices](skills/cuda-best-practices) | NVIDIA CUDA C/C++ GPU programming: kernels, memory hierarchy, coalescing, streams, profiling, Thrust/CUB |

### AI & Data

| Skill | Description |
|-------|-------------|
| [rag-pgvector](skills/rag-pgvector) | RAG pipelines with pgvector: hybrid search, RRF fusion, chunking, reranking, citations |

### Tools & Workflows

| Skill | Description |
|-------|-------------|
| [agent-swarm](skills/agent-swarm) | Agent teams orchestration: TeamCreate, SendMessage, tasks, 4 patterns, tmux, hooks, cleanup |
| [mcp-server-dev](skills/mcp-server-dev) | Build MCP servers in TypeScript: tools, resources, prompts, transports |
| [skill-creator](skills/skill-creator) | Interactive guide for creating and updating Claude Code skills |

## Install

Install all skills:

```bash
npx skills add https://github.com/ArthurDEV44/skills
```

Install a specific skill:

```bash
npx skills add https://github.com/ArthurDEV44/skills --skill rust-traits
```

## License

MIT
