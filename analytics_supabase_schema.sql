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

create table if not exists public.analytics_validation_events (
  id uuid primary key default gen_random_uuid(),
  generation_id text not null,
  timestamp timestamptz not null default now(),
  product text not null,
  source text not null check (source in ('ui', 'mcp')),
  pipeline text not null check (pipeline in ('standard', 'artistic')),
  tool_name text not null,
  analytics_opt_in boolean not null default false,
  error_bucket text not null,
  error_message_excerpt text,
  error_message_hash text,
  anonymous_id text not null,
  prompt_full text,
  qr_payload_full text,
  settings_full jsonb,
  created_at timestamptz not null default now()
);

create index if not exists analytics_validation_events_source_pipeline_idx
  on public.analytics_validation_events (source, pipeline, timestamp desc);

alter table public.analytics_validation_events enable row level security;

revoke all on table public.analytics_validation_events from anon, authenticated;

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

create or replace view public.analytics_download_events_inferred as
select
  d.id,
  d.generation_id,
  d.timestamp,
  d.product,
  d.source,
  d.pipeline,
  d.tool_name,
  d.analytics_opt_in,
  d.format,
  d.anonymous_id,
  d.qr_payload_full,
  d.seed,
  d.created_at,
  case
    when d.tool_name like '%_standard' then 'standard'
    when d.tool_name like '%_artistic' then 'artistic'
    when d.source = 'mcp' and d.tool_name like '%_1' then 'standard'
    when d.source = 'mcp' and d.tool_name not like '%_1' then 'artistic'
    else d.pipeline
  end as pipeline_inferred
from public.analytics_download_events d;

create or replace view public.analytics_generation_signals as
with ordered_generations as (
  select
    g.id,
    g.generation_id,
    g.timestamp,
    g.product,
    g.source,
    g.pipeline,
    g.tool_name,
    g.analytics_opt_in,
    g.status,
    g.error_bucket,
    g.anonymous_id,
    g.prompt_full,
    g.qr_payload_full,
    g.settings_full,
    g.created_at,
    lead(g.timestamp) over (
      partition by g.source, g.anonymous_id, g.pipeline
      order by g.timestamp
    ) as next_generation_timestamp
  from public.analytics_generation_events g
), generation_with_downloads as (
  select
    g.generation_id,
    g.timestamp,
    g.product,
    g.source,
    g.pipeline,
    g.analytics_opt_in,
    g.status,
    g.error_bucket,
    g.anonymous_id,
    g.prompt_full,
    g.qr_payload_full,
    g.settings_full,
    g.next_generation_timestamp,
    exists (
      select 1
      from public.analytics_download_events d
      where d.source = g.source
        and d.anonymous_id = g.anonymous_id
        and d.timestamp >= g.timestamp
        and d.timestamp <= g.timestamp + interval '10 minutes'
    ) as has_download_within_10m
  from ordered_generations g
)
select
  generation_id,
  timestamp,
  product,
  source,
  pipeline,
  analytics_opt_in,
  status,
  error_bucket,
  anonymous_id,
  prompt_full,
  qr_payload_full,
  settings_full,
  has_download_within_10m,
  next_generation_timestamp,
  case
    when error_bucket = 'infra_limited' then 'infra_limited'
    when status = 'success' and has_download_within_10m then 'happy'
    when status = 'success'
      and next_generation_timestamp is not null
      and next_generation_timestamp <= timestamp + interval '10 minutes'
      and not has_download_within_10m then 'unhappy'
    when status = 'error' then 'error'
    else 'neutral'
  end as outcome_signal
from generation_with_downloads;

alter view public.analytics_generation_outcomes set (security_invoker = true);
alter view public.analytics_download_events_inferred set (security_invoker = true);
alter view public.analytics_generation_signals set (security_invoker = true);

revoke all on table public.analytics_generation_outcomes from anon, authenticated;
revoke all on table public.analytics_download_events_inferred from anon, authenticated;
revoke all on table public.analytics_generation_signals from anon, authenticated;
