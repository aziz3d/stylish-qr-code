---
title: AI QR Code Generator
emoji: 🌍
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
tags:
- mcp-server-track
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


## Demo
[Watch the demo video](showcase.mp4)

## Optional analytics

The Space includes an optional analytics consent toggle for both UI and MCP usage.

- Generated images are not used for analytics.
- Full prompts, QR payload text, and settings are only stored when analytics opt-in is enabled.
- Minimal operational events can still be logged for reliability and product metrics.

## URL normalization

When `Input Type` is set to `URL`, the app normalizes links on CPU before any GPU work starts.

- Common tracking params such as `utm_*`, `fbclid`, and `gclid` are removed automatically.
- Scheme and host casing are normalized to reduce unnecessary QR payload length.
- Plain text payloads are left untouched.

### Required Space secrets

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `POSTHOG_API_KEY` (or reuse `POSTHOG_SECRET_KEY`)
- `POSTHOG_HOST` (optional, defaults to `https://us.i.posthog.com`)
- `ANALYTICS_ENABLED` (optional, defaults to `true`)
- `ANALYTICS_DEFAULT_OPT_IN` (optional, defaults to `false`)

### Supabase schema

Apply `analytics_supabase_schema.sql` to your Supabase project before enabling writes from the Space.

The analytics schema is intended for backend-only writes using `SUPABASE_SERVICE_ROLE_KEY`.
Do not grant `anon` or `authenticated` direct access to the analytics tables or derived views.
