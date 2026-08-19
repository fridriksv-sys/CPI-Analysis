-- OPTION 1 for grocery price history (preferred): run this in the home_app
-- Supabase project (vamspsjnwpfpocdiquzw). Creates an append-only daily price
-- snapshot of the existing grocery_catalog mirror + a pg_cron job.
--
-- NOT applied automatically - the CPI model must not silently modify another
-- app's database. Apply via the Supabase SQL editor or ask Claude to apply it.
--
-- Rollback:
--   drop table kronan_price_history;
--   drop function snapshot_kronan_prices();
--   select cron.unschedule('kronan-price-snapshot');

create table if not exists kronan_price_history (
  snapshot_date date not null default current_date,
  sku text not null,
  price integer,
  discounted_price integer,
  on_sale boolean not null default false,
  unit text,
  category_path text,
  primary key (snapshot_date, sku)
);

alter table kronan_price_history enable row level security;
-- service-role/cron writes only; no public policies.

create or replace function snapshot_kronan_prices() returns integer
language sql security definer set search_path = public as $$
  with ins as (
    insert into kronan_price_history
      (snapshot_date, sku, price, discounted_price, on_sale, unit, category_path)
    select current_date, sku, price, discounted_price, on_sale, unit, category_path
    from grocery_catalog
    where price is not null
      and listing_synced_at > now() - interval '14 days'
    on conflict (snapshot_date, sku) do nothing
    returning 1
  )
  select count(*)::integer from ins;
$$;

select cron.schedule(
  'kronan-price-snapshot',
  '30 8 * * *',  -- daily 08:30 UTC, after the morning catalog sync
  $$select snapshot_kronan_prices()$$
);

-- Seed with today's snapshot
select snapshot_kronan_prices();
