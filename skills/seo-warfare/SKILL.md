---
model: opus
name: seo-warfare
description: >
  SEO audit, optimization, and generation pipeline for traditional search and AI search
  (Google AI Overviews, ChatGPT Search, Perplexity). Orchestrates agent-websearch, agent-explore,
  and agent-docs in a multi-phase pipeline. Use when the user says "seo", "seo audit",
  "seo-warfare", "structured data", "schema markup", "core web vitals", "GEO", "AEO",
  or asks to optimize for search engines, fix SEO issues, or generate schema markup.
  Do NOT use for general web development tasks that have no SEO component.
argument-hint: "[url-or-domain?] [mode?]"
---

# seo-warfare — Search Engine Domination Pipeline

## The 2026 SEO Landscape

The search ecosystem has fundamentally shifted. You must optimize for TWO audiences simultaneously:

1. **Traditional search crawlers** (Googlebot, Bingbot) — still drive 60%+ of organic traffic
2. **AI answer engines** (Google AI Overviews, ChatGPT Search, Perplexity, Claude) — growing at
   800% YoY referral traffic, appearing on 47%+ of informational queries

Sites that optimize for only one audience will lose. Your job is to win both wars.

## Runtime Output Format

Before each phase, print a progress header:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Phase N/8] PHASE_NAME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Between phases, print: `───────────────────────────────`

## Phase 0: CLASSIFY (before any agent)

Parse `$ARGUMENTS` and route before loading references or spawning agents.

**0a. Parse arguments:**

- If `$ARGUMENTS` contains a URL or domain → use as target (store as `target_url`)
- If `$ARGUMENTS` contains a mode keyword (e.g., "audit", "quick-meta", "geo", "vitals") → route directly to the matching mode
- If no arguments → scan the codebase local directory and ask the user for the target domain

**0b. Determine mode** by matching user request against the Workflow Modes table below:

- Exact match on trigger phrase → route immediately to that mode's phases
- Multiple possible matches → ask the user to disambiguate
- No match / generic "SEO" request → default to Full Audit
- Quick modes (Quick Meta, Quick Sitemap, Quick Robots, Quick Schema, Quick llms.txt, Quick IndexNow) → skip to Quick Generation Commands section directly, do NOT load full pipeline

**0c. For Quick modes:** Do NOT read reference files until the Quick section instructs it. Return the generated artifact and stop.

**0d. For pipeline modes:** Proceed to Phase 1.

## Workflow Modes

The skill operates in different modes depending on what the user needs:

| Mode | Trigger | Phases |
|------|---------|--------|
| **Full Audit** | "seo audit", "full seo review" | All 8 phases |
| **Technical Fix** | "fix seo", "technical seo" | Phases 1-3 only |
| **Structured Data** | "schema markup", "structured data", "json-ld" | Phase 1 + 4 only |
| **AI Optimization** | "GEO", "AEO", "AI search", "llms.txt" | Phase 1 + 5 only |
| **Performance** | "core web vitals", "lighthouse", "page speed" | Phase 1 + 6 only |
| **Content Strategy** | "content strategy", "topic clusters", "pillar pages" | Phase 1 + 7 only |
| **Competitive Intel** | "competitor analysis", "keyword gap", "backlink gap" | Phase 3 only |
| **Quick Meta** | "meta tags", "generate meta tags for" | Generate meta tags template only |
| **Quick Sitemap** | "sitemap", "generate sitemap" | Generate sitemap only |
| **Quick Robots** | "robots.txt" | Generate robots.txt only |

## Execution Flow

```
User Request + $ARGUMENTS
     |
     v
+---------------+
|  Phase 0:     |
|  CLASSIFY     |  <- Parse args, detect mode, route or ask
+-------+-------+
        |
        v
+---------------+
|  Phase 1:     |
|  INTAKE &     |  <- Detect project stack, read existing SEO state
|  DISCOVERY    |
+-------+-------+
        |
        v
+-------+-------+
|  Phase 2:     |
|  TECHNICAL    |  <- Crawlability, indexing, robots.txt, sitemap, canonicals
|  SEO AUDIT    |
+-------+-------+
        |
        v
+-------+----------------------------+
|            PARALLEL                 |
|  +------------+  +--------------+  |
|  | Phase 3a:  |  | Phase 3b:    |  |
|  | COMPETITIVE|  | FRAMEWORK    |  |
|  | INTEL      |  | DOCS         |  |
|  | (websearch)|  | (agent-docs) |  |
|  +-----+------+  +------+------+  |
|        |                |          |
+--------+----------------+----------+
         |
    [compress to <500 words]
         |
         v
+--------+--------+
|  Phase 4:       |
|  STRUCTURED     |  <- Generate/fix all JSON-LD schema markup
|  DATA           |
+---------+-------+
          |
          v
+---------+-------+
|  Phase 5:       |
|  AI SEARCH      |  <- GEO optimization, llms.txt, E-E-A-T
|  OPTIMIZATION   |
+---------+-------+
          |
          v
+---------+-------+
|  Phase 6:       |
|  CORE WEB       |  <- LCP, CLS, INP, Lighthouse optimization
|  VITALS         |
+---------+-------+
          |
          v
+---------+-------+
|  Phase 7:       |
|  CONTENT        |  <- Topic clusters, featured snippets, PAA
|  STRATEGY       |
+---------+-------+
          |
          v
+---------+-------+
|  Phase 8:       |
|  REPORT &       |  <- Priority-ranked action items, implementation plan
|  ACTION PLAN    |
+---------+-------+
```

## Phase Details

### Phase 1: INTAKE & DISCOVERY

**Goal:** Understand the project, its stack, and current SEO state.

1. **Detect project type** via Glob for manifest files:
   - `package.json` → Node/React/Next.js/Nuxt/etc.
   - `Cargo.toml` → Rust (likely Axum/Actix serving HTML or API)
   - `pyproject.toml` / `requirements.txt` → Python (Django/Flask/FastAPI)
   - `composer.json` → PHP (Laravel/WordPress)
   - `Gemfile` → Ruby (Rails)

2. **Detect framework** from dependencies — this determines SSR/SSG strategy:
   - Next.js → App Router (RSC) or Pages Router
   - Nuxt → Nuxt 3 with Nitro
   - Astro → Static-first with islands
   - Gatsby → Static with GraphQL
   - SvelteKit → SSR/SSG hybrid
   - Plain HTML → direct optimization

3. **Scan existing SEO artifacts** via Glob and Read:
   - `robots.txt` — check for blocking issues
   - `sitemap.xml` or `sitemap-index.xml`
   - `public/llms.txt` or root `llms.txt`
   - Any `<meta>` tags in layout/head components
   - Any existing JSON-LD structured data
   - `.env` files for SEO-related API keys (Search Console, IndexNow)

4. **Check for SEO libraries** in dependencies:
   - `next-seo`, `@nuxtjs/seo`, `astro-seo`, `react-helmet`, `@unhead/vue`
   - Schema generators: `schema-dts`, `next-seo`

5. **Output:** Project profile in this exact format:

```yaml
project_profile:
  target: "{target_url or local project}"
  stack: "{language/runtime}"
  framework: "{framework name + version}"
  rendering: "{SSR | SSG | CSR | hybrid | static HTML}"
  seo_library: "{library name or 'none'}"
  existing_artifacts:
    robots_txt: "{present | missing | blocking-issues}"
    sitemap: "{present | missing | outdated}"
    llms_txt: "{present | missing}"
    structured_data: "{types found or 'none'}"
    meta_tags: "{present in layout | missing | partial}"
  gaps: ["{gap_1}", "{gap_2}", "..."]
```

**GATE:** Project profile is complete — stack, framework, and rendering strategy are all identified. If no web-serving framework detected (pure API, CLI tool), stop and inform the user this project may not benefit from SEO optimization.

---

### Phase 2: TECHNICAL SEO AUDIT

**Goal:** Identify all technical SEO issues that block crawling, indexing, or ranking.

Spawn `agent-explore` to scan the codebase for these issues:

```
Agent(
  description: "Technical SEO audit of codebase",
  prompt: "Scan this codebase for technical SEO issues. Check: (1) robots.txt — exists, allows Googlebot/Bingbot, AI crawler policy for GPTBot/ClaudeBot/PerplexityBot, references sitemap; (2) XML sitemap — exists, has accurate lastmod dates; (3) Canonical URLs on all pages; (4) No noindex on important pages; (5) Server-side rendering of critical content; (6) Meta tags in initial HTML (not JS-injected); (7) Navigation uses <a href> not onClick; (8) Clean URL structure, consistent trailing slash policy; (9) No duplicate content (www/non-www, http/https); (10) Hreflang for multi-language. Return a structured checklist with PASS/FAIL/MISSING for each item and the file:line where issues are found.",
  subagent_type: "agent-explore"
)
```

**Crawlability checklist:**
- [ ] `robots.txt` exists and allows Googlebot/Bingbot
- [ ] AI crawler policy defined (GPTBot, ClaudeBot, PerplexityBot)
- [ ] XML sitemap exists with accurate `<lastmod>` dates
- [ ] Sitemap referenced in `robots.txt`
- [ ] No orphan pages (every page reachable within 3 clicks)
- [ ] Canonical URLs set on all pages
- [ ] No `noindex` on important pages
- [ ] Hreflang set for multi-language sites

**Rendering checklist:**
- [ ] Critical content rendered server-side (not client-only)
- [ ] Meta tags in initial HTML response (not injected by JS)
- [ ] `<a href>` links used for navigation (not `onClick` handlers)
- [ ] Hydration mismatches checked in production build

**URL structure checklist:**
- [ ] Clean, descriptive URLs (no query parameters for content pages)
- [ ] Consistent trailing slash policy
- [ ] No duplicate content (www/non-www, http/https resolved)
- [ ] 301 redirects for any moved pages

Consult `references/technical-seo.md` for the complete audit checklist and fix patterns.

**GATE:** At least 10 of the 16 checklist items have been evaluated as PASS/FAIL/MISSING. If agent-explore fails, fall back to direct Glob/Grep scanning of the codebase.

---

### Phase 3: COMPETITIVE INTELLIGENCE + FRAMEWORK DOCS (parallel)

Spawn BOTH agents in a **SINGLE message** with multiple Agent tool calls for true parallelism:

```
// Phase 3a — ALWAYS runs
Agent(
  description: "SEO competitive analysis for {target}",
  prompt: "Research the top 5 competitors for '{target_url}' or '{target_domain}'. For each competitor, find: (1) What structured data/schema markup they use, (2) Their content strategy (topic clusters, content types, publishing frequency), (3) SERP features they capture (featured snippets, PAA, AI Overviews), (4) Their domain authority and backlink profile strength. Return a structured comparison table.",
  subagent_type: "agent-websearch"
)

// Phase 3b — ONLY if a framework-specific SEO library was detected in Phase 1
Agent(
  description: "Fetch docs for {seo_library}",
  prompt: "Look up documentation for {seo_library} (version {version}). Focus on: meta tag configuration, JSON-LD generation, sitemap generation, and any SEO-specific APIs. Return code examples for the most common use cases.",
  subagent_type: "agent-docs"
)
```

**After both agents complete — COMPRESS before passing downstream:**

Synthesize Phase 3a + 3b outputs into a structured summary of **< 500 words** containing:
- Top 3 competitors and their key SEO strengths
- SERP feature opportunities (what competitors have that the target doesn't)
- Framework-specific SEO API patterns to use in Phases 4-6
- Any contradictions between competitor practices and Google guidelines

This compressed summary is the ONLY context passed to Phases 4-7. Raw agent output is NOT forwarded.

**GATE:** agent-websearch returned at least 2 competitor profiles. If agent-websearch fails, base competitive context on codebase analysis only and note "Competitive data unavailable" in the Phase 8 report.

---

### Phase 4: STRUCTURED DATA GENERATION

**Goal:** Generate complete, valid JSON-LD for all applicable schema types.

Consult `references/structured-data.md` for:
- Complete JSON-LD templates for every Schema.org type
- Priority tier system (which schemas to implement first)
- Rich results eligibility requirements
- Common validation errors and fixes

**Implementation priority:**

| Priority | Schema Type | Impact |
|----------|------------|--------|
| P0 | Organization / LocalBusiness | Brand knowledge panel |
| P0 | BreadcrumbList | Navigation rich results |
| P0 | WebSite (with SearchAction) | Sitelinks search box |
| P1 | Article / BlogPosting | Discover eligibility, AI citation |
| P1 | FAQPage | Featured snippets, AI extraction |
| P1 | Product + Offer + Review | Shopping rich results, 18-25% CTR lift |
| P2 | HowTo | Step-by-step rich results |
| P2 | VideoObject | Video carousel |
| P2 | SoftwareApplication | App rich results |
| P3 | Event | Event rich results |
| P3 | Recipe | Recipe carousel |
| P3 | Course | Course rich results |

Generate schemas based on the actual page content. Never generate fake or misleading schema data.

**GATE:** All generated JSON-LD must be syntactically valid JSON. Verify by parsing each block inline. If using a framework SEO library (from Phase 1), generate code using that library's API (from Phase 3b docs) rather than raw `<script>` tags.

---

### Phase 5: AI SEARCH OPTIMIZATION (GEO/AEO)

**Goal:** Optimize content structure for citation in AI-generated answers.

This is the most strategically important phase for 2026. Consult `references/ai-search-optimization.md` for:
- Content structure patterns that AI systems prefer
- E-E-A-T signal implementation
- llms.txt generation
- Platform-specific optimization (Google AI Overviews vs ChatGPT vs Perplexity)
- The 12 tactics for AI Overview citation

**Key actions:**
1. Restructure headings as natural-language questions
2. Add direct-answer openings to each section (2-3 sentence summaries)
3. Include specific, verifiable numbers with inline citations
4. Generate `llms.txt` pointing to priority content
5. Ensure FAQPage schema accompanies all FAQ sections
6. Add author schema with credentials and social proof

**GATE:** At least 4 of the 6 key actions have been addressed. llms.txt has been generated if it was missing.

---

### Phase 6: CORE WEB VITALS OPTIMIZATION

**Goal:** Achieve green scores on all Core Web Vitals (LCP < 2.5s, CLS < 0.1, INP < 200ms).

Consult `references/core-web-vitals.md` for:
- LCP optimization techniques (preload, critical CSS, font optimization)
- CLS prevention patterns (explicit dimensions, aspect-ratio, font metrics)
- INP optimization code patterns (scheduler.yield, Web Workers, DOM batching)
- Image optimization (AVIF/WebP, responsive images, lazy loading strategy)
- Performance budget enforcement

**Framework-specific optimizations:**
- **Next.js**: Use `next/image` with `priority` for LCP, `next/font` for font optimization,
  App Router server components for zero client JS
- **Astro**: Islands architecture, zero JS by default, `astro:assets` for images
- **Nuxt**: `useHead` for meta tags, `NuxtImg` for optimized images
- **SvelteKit**: Streaming SSR, code splitting, `enhanced:img`

**GATE:** All code changes use framework-appropriate APIs (from project profile). No script or resource added that increases page weight by more than 5KB without justification.

---

### Phase 7: CONTENT STRATEGY

**Goal:** Design a content architecture that establishes topical authority.

Consult `references/content-strategy.md` for:
- Topic cluster methodology
- Pillar page templates
- Featured snippet targeting patterns
- People Also Ask optimization
- Content freshness signals
- Long-tail keyword strategies
- Internal linking architecture

Using the compressed competitive intelligence from Phase 3, identify:
1. Content gaps — topics competitors cover that the target doesn't
2. Featured snippet opportunities — queries where competitors hold snippets that could be captured
3. Topic cluster map — 1 pillar + 5-10 supporting pages per core topic
4. Internal linking recommendations — specific pages to cross-link

**GATE:** At least one topic cluster has been defined with a pillar page and 3+ supporting pages. Content gaps are mapped against competitive data.

---

### Phase 8: REPORT & ACTION PLAN

**Goal:** Deliver a priority-ranked, actionable implementation plan.

**Report structure:**

```markdown
# SEO Audit Report — [Project Name]
**Date:** [current date]
**Target:** [target_url or domain]
**Audited by:** Claude SEO Warfare Pipeline

## Executive Summary
[3-5 sentence overview of current state and biggest opportunities]

## Critical Issues (Fix Immediately)
| # | Issue | Impact | Fix | File(s) |
|---|-------|--------|-----|---------|
| 1 | [issue] | [HIGH/MEDIUM/LOW] | [specific fix] | [file:line] |

## Structured Data Status
| Schema Type | Status | Action |
|-------------|--------|--------|
| Organization | [Missing/Present/Incomplete] | [action] |

## Core Web Vitals Assessment
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| LCP | [value] | < 2.5s | [PASS/FAIL] |
| CLS | [value] | < 0.1 | [PASS/FAIL] |
| INP | [value] | < 200ms | [PASS/FAIL] |

## AI Search Readiness Score
[Score 0-100 based on GEO optimization checklist]
**Confidence:** {high|medium|low} — {basis}

## Content Strategy Recommendations
[Topic cluster map, content gaps, featured snippet opportunities]

## Competitive Gap Analysis
[Keyword gaps, backlink gaps, SERP feature gaps]
[Note: "Competitive data unavailable" if Phase 3a failed]

## Implementation Roadmap
### Week 1 (Critical)
- [ ] [task]
### Week 2-3 (High Priority)
- [ ] [task]
### Month 2 (Medium Priority)
- [ ] [task]
### Ongoing
- [ ] [task]
```

## Quick Generation Commands

For targeted outputs without full audit:

### Quick Meta Tags
When user asks to "generate meta tags" for a page, output the complete meta tag template
from `references/meta-tags-templates.md`, customized for the page content.

### Quick Robots.txt
Generate a robots.txt with AI crawler management. See `references/technical-seo.md`.

### Quick Sitemap
Generate XML sitemap structure. See `references/technical-seo.md`.

### Quick Schema
Generate specific JSON-LD for a page. See `references/structured-data.md`.

### Quick llms.txt
Generate llms.txt file. See `references/ai-search-optimization.md`.

### Quick IndexNow
Set up IndexNow protocol. See `references/technical-seo.md`.

## Hard Rules

1. **Never generate fake schema data** — all structured data must reflect actual page content
2. **Never keyword-stuff** — entity salience > keyword density in 2026
3. **Never recommend black-hat techniques** — no link schemes, cloaking, hidden text, or PBNs
4. **Always validate JSON-LD** — output must pass Google Rich Results Test
5. **Always cite sources** — link to Google documentation for recommendations
6. **Framework-aware** — generate code that works with the detected framework
7. **Mobile-first** — every recommendation must work on mobile
8. **Performance-conscious** — never add scripts/resources that degrade Core Web Vitals
9. **AI-dual-optimized** — every content recommendation must serve both traditional SEO and GEO
10. **Competitive context** — recommendations should be calibrated against what competitors are doing
11. **Phase 0 ALWAYS runs** — classify and route before loading references or spawning agents
12. **Phase 1 completes before Phase 2** — project profile is required context for all downstream phases
13. **Phase 3a + 3b spawn in a SINGLE message** — never sequentially
14. **Compress Phase 3 output to < 500 words** before passing to Phases 4-7
15. **Print `[Phase N/8]` progress headers** before each phase — never skip progress indicators

## DO NOT

1. Generate structured data without first reading the actual page content via Read or agent-explore
2. Recommend techniques a competitor uses if those techniques violate Google guidelines
3. Launch all agents simultaneously — Phase 1 (intake) MUST complete before Phase 2-3
4. Pass raw agent output to downstream phases — always compress to < 500 words first
5. Run the full 8-phase pipeline for Quick mode requests — generate the single artifact and stop
6. Invent Core Web Vitals metrics without analyzing the codebase — report "Not measurable from code analysis" if you cannot determine actual values
7. Give traffic or ranking predictions — these are speculations, not actionable data
8. Skip Phase 0 classification — even for seemingly obvious requests, confirm the mode
9. Generate code that doesn't match the detected framework — plain `<script>` tags in a Next.js App Router project is wrong
10. Silently skip a phase that failed — note the failure in the Phase 8 report

## Error Handling

| Scenario | Fallback |
|----------|----------|
| **agent-websearch fails** | Skip competitive analysis. Base Phase 7 content strategy on codebase analysis only. Note "Competitive data unavailable" in report. |
| **agent-explore fails** | Fall back to direct Glob/Grep scanning for Phase 2 checklist items. Reduce audit depth. |
| **agent-docs fails** | Use Phase 3a web research results for framework SEO library guidance. Note reduced doc coverage. |
| **No framework detected** (pure static HTML) | Treat as plain HTML. Skip framework-specific optimizations. Focus on raw HTML/meta tag/schema generation. |
| **No package.json / manifest found** | Treat as static site or non-web project. Ask user to confirm this is a web project before proceeding. |
| **Context7 MCP unavailable** | agent-docs falls back to web research for library documentation. Note the fallback. |
| **Exa MCP unavailable** | agent-websearch falls back to native WebSearch/WebFetch automatically. |
| **Target URL/domain not provided** | Ask user before proceeding with Phase 3 competitive intel. Phase 1-2 can run on local codebase without a target URL. |
| **Project is API-only (no HTML rendering)** | Stop after Phase 1. Inform user that SEO optimization requires web-facing pages. |

## References

- [Technical SEO](references/technical-seo.md) — robots.txt, sitemaps, canonicals, crawl budget, IndexNow, rendering
- [Structured Data](references/structured-data.md) — all JSON-LD templates, Schema.org types, rich results
- [AI Search Optimization](references/ai-search-optimization.md) — GEO/AEO, llms.txt, E-E-A-T, AI citation
- [Core Web Vitals](references/core-web-vitals.md) — LCP, CLS, INP optimization, performance patterns
- [Content Strategy](references/content-strategy.md) — topic clusters, featured snippets, content architecture
- [Competitive Analysis](references/competitive-analysis.md) — keyword gap, backlink gap, SERP features
- [Meta Tags Templates](references/meta-tags-templates.md) — ready-to-use meta tag templates for every page type

## Done When

- [ ] Phase 0 (Classify) completed — mode determined and routed
- [ ] All applicable phases for the mode have been executed
- [ ] Generated JSON-LD is syntactically valid (parsed inline)
- [ ] Framework-appropriate code generated (not raw `<script>` tags in App Router)
- [ ] Phase 8 report produced with priority-ranked action items
- [ ] No fake schema data — all structured data reflects actual content

## Constraints (Three-Tier)

### ALWAYS
- Run Phase 0 (Classify) — route before loading references or spawning agents
- Complete Phase 1 before Phase 2 — project profile is required context
- Spawn Phase 3a + 3b in a SINGLE message for parallel execution
- Compress Phase 3 output to < 500 words before passing downstream
- Validate all JSON-LD by parsing each block inline

### ASK FIRST
- Target URL/domain when not provided (required for competitive intel)
- Mode selection when request is ambiguous

### NEVER
- Generate fake or misleading schema data
- Recommend black-hat techniques (link schemes, cloaking, hidden text, PBNs)
- Add scripts/resources that degrade Core Web Vitals without justification
- Give traffic or ranking predictions — these are speculations
- Generate code that doesn't match the detected framework
