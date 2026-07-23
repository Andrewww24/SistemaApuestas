# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Single-script Python project (`mlb-ingest.py`) that ingests daily MLB data from the free, public MLB Stats API (`https://statsapi.mlb.com/api/v1`, no API key required) into a Supabase/PostgreSQL database via SQLAlchemy Core (no ORM models — raw `Table` definitions).

## Setup & running

```
pip install requests sqlalchemy psycopg2-binary python-dotenv
```

Requires a `.env` file in the repo root with:
```
DATABASE_URL=postgresql+psycopg2://postgres:PASSWORD@db.xxxx.supabase.co:5432/postgres
SEASON=2026
```
(Get the connection string from Supabase → Project Settings → Database → Connection string → URI, swapping the `postgresql://` prefix for `postgresql+psycopg2://`.)

Run:
```
python mlb-ingest.py                    # ingest today's data
python mlb-ingest.py --date 2026-07-22  # ingest a specific date
python mlb-ingest.py --teams            # refresh only the teams catalog
```

There is no build step, linter, or test suite configured in this repo.

## Architecture

Everything lives in `mlb-ingest.py`, structured top to bottom as:

1. **Table definitions** (SQLAlchemy Core `Table` objects, not ORM classes) for `teams`, `players`, `games`, `pitcher_game_stats`, `batter_game_stats`, `team_game_stats`. These column sets must stay in sync with the schema already created in Supabase — this script only inserts/updates the columns it defines, it doesn't manage migrations.
2. **Helpers**:
   - `get()` — thin wrapper over `requests.get` with error raising.
   - `upsert()` — generic Postgres `INSERT ... ON CONFLICT DO UPDATE` builder used by every ingest function; takes the target table and a list of conflict columns.
   - `ip_to_decimal()` — converts MLB's "innings pitched" notation (e.g. `6.1` = 6⅓) into true decimal (`6.33`), important for downstream K/9-style calculations.
   - `num()` — tolerant numeric getter for stats dicts that may omit keys.
3. **Ingest functions**, each opening/using a single connection passed in from the caller:
   - `ingest_teams` — refreshes the team catalog for the season.
   - `ingest_player` — upserts a single player's bio/handedness details (called lazily from `ingest_boxscore`, only for pitchers that appear in a boxscore).
   - `ingest_schedule` — pulls the day's games and returns their `gamePk` ids.
   - `ingest_boxscore` — pulls per-game team and player stats (pitching + batting) for one game.
4. **`run_daily(game_date, teams_only)`** — orchestrates the full flow inside one transaction (`engine.begin()`), so a failure rolls back everything for that run. Games without a boxscore yet (not started) are caught and skipped individually rather than aborting the whole run.

Key behavior to preserve when modifying:
- All writes use upsert semantics keyed by natural IDs (`team_id`, `player_id`, `game_id` (+`player_id`/`team_id` for stat tables)) — re-running ingestion for the same date is idempotent.
- `ingest_boxscore` only inserts a batter row if `atBats` is present and non-negative, and only inserts a pitcher row if `inningsPitched` is present — these guard against partial/in-progress boxscores.
- The script and its comments/docstrings are in Spanish; keep new code comments consistent with that if extending the file.
