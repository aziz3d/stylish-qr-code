create extension if not exists pgcrypto;

create table if not exists public.analytics_generation_events (
  id uuid primary key default gen_random_uuid(),
  generation_id text not null,
  timestamp timestamptz not null default now(),
  product text not null,
  source text not null check (source in ('ui', 'mcp')),
  pipeline text not null check (pipeline in ('standard', 'artistic')),
  tool_name text not null,
  analytics_opt_in boolean not null default false,
  status text not null check (status in ('success', 'error')),
  error_bucket text not null,
  error_message_excerpt text,
  error_message_hash text,
  anonymous_id text not null,
  prompt_full text,
  qr_payload_full text,
  settings_full jsonb,
  created_at timestamptz not null default now()
);

create index if not exists analytics_generation_events_generation_id_idx
  on public.analytics_generation_events (generation_id);

create index if not exists analytics_generation_events_source_pipeline_idx
  on public.analytics_generation_events (source, pipeline, timestamp desc);

alter table public.analytics_generation_events enable row level security;

revoke all on table public.analytics_generation_events from anon, authenticated;

create table if not exists public.analytics_download_events (
  id uuid primary key default gen_random_uuid(),
  generation_id text,
  timestamp timestamptz not null default now(),
  product text not null,
  source text not null check (source in ('ui', 'mcp')),
  pipeline text not null,
  tool_name text not null,
  analytics_opt_in boolean not null default false,
  format text not null check (format in ('png', 'svg')),
  anonymous_id text not null,
  qr_payload_full text,
  seed bigint,
  created_at timestamptz not null default now()
);

create index if not exists analytics_download_events_generation_id_idx
  on public.analytics_download_events (generation_id);

create index if not exists analytics_download_events_source_pipeline_idx
  on public.analytics_download_events (source, pipeline, timestamp desc);

alter table public.analytics_download_events enable row level security;

revoke all on table public.analytics_download_events from anon, authenticated;

create or replace view public.analytics_generation_outcomes as
select
  g.generation_id,
  g.timestamp as generation_timestamp,
  g.source,
  g.pipeline,
  g.analytics_opt_in,
  g.status,
  g.error_bucket,
  exists (
    select 1
    from public.analytics_download_events d
    where d.generation_id = g.generation_id
  ) as has_download
from public.analytics_generation_events g;
