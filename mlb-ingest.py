"""
mlb_ingest.py
=============
Ingesta diaria de datos de la MLB Stats API (oficial, gratuita, sin API key)
hacia una base de datos Supabase / PostgreSQL usando SQLAlchemy.

Llena las tablas: teams, players, games, pitcher_game_stats,
batter_game_stats, team_game_stats.

Uso:
    python mlb_ingest.py                # ingesta de HOY
    python mlb_ingest.py --date 2026-07-22
    python mlb_ingest.py --teams        # refresca solo el catálogo de equipos

Requisitos:
    pip install requests sqlalchemy psycopg2-binary python-dotenv

Configuración: crea un archivo .env junto a este script con:
    DATABASE_URL=postgresql+psycopg2://postgres:TU_PASSWORD@db.xxxx.supabase.co:5432/postgres
    SEASON=2026
(La cadena exacta la encontrás en Supabase → Project Settings → Database → Connection string → URI,
 cambiando el prefijo "postgresql://" por "postgresql+psycopg2://".)
"""

from __future__ import annotations

import argparse
import os
from datetime import date

import requests
from dotenv import load_dotenv
from sqlalchemy import (
    Boolean, Column, Date, Integer, MetaData, Numeric, String, Table,
    create_engine,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------
load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
SEASON = os.getenv("SEASON", str(date.today().year))
API = "https://statsapi.mlb.com/api/v1"

engine = create_engine(DATABASE_URL, future=True)
meta = MetaData()

# --------------------------------------------------------------------------
# Definición de tablas (solo las columnas que insertamos; deben coincidir
# con el esquema que ya creaste en Supabase)
# --------------------------------------------------------------------------
teams = Table(
    "teams", meta,
    Column("team_id", Integer, primary_key=True),
    Column("name", String), Column("abbreviation", String),
    Column("league", String), Column("division", String),
)

players = Table(
    "players", meta,
    Column("player_id", Integer, primary_key=True),
    Column("full_name", String), Column("team_id", Integer),
    Column("primary_position", String),
    Column("bat_side", String), Column("throw_side", String),
)

games = Table(
    "games", meta,
    Column("game_id", Integer, primary_key=True),
    Column("game_date", Date),
    Column("home_team_id", Integer), Column("away_team_id", Integer),
    Column("home_pitcher_id", Integer), Column("away_pitcher_id", Integer),
    Column("status", String),
    Column("home_score", Integer), Column("away_score", Integer),
)

pitcher_game_stats = Table(
    "pitcher_game_stats", meta,
    Column("game_id", Integer), Column("player_id", Integer),
    Column("team_id", Integer), Column("is_home", Boolean),
    Column("innings_pitched", Numeric), Column("strikeouts", Integer),
    Column("hits_allowed", Integer), Column("walks", Integer),
    Column("earned_runs", Integer), Column("home_runs_allowed", Integer),
    Column("pitch_count", Integer), Column("batters_faced", Integer),
)

batter_game_stats = Table(
    "batter_game_stats", meta,
    Column("game_id", Integer), Column("player_id", Integer),
    Column("team_id", Integer), Column("is_home", Boolean),
    Column("at_bats", Integer), Column("hits", Integer),
    Column("doubles", Integer), Column("triples", Integer),
    Column("home_runs", Integer), Column("runs", Integer),
    Column("rbi", Integer), Column("walks", Integer),
    Column("strikeouts", Integer),
)

team_game_stats = Table(
    "team_game_stats", meta,
    Column("game_id", Integer), Column("team_id", Integer),
    Column("is_home", Boolean), Column("runs", Integer),
    Column("hits", Integer), Column("errors", Integer),
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def get(url: str, **params) -> dict:
    """GET con manejo básico de errores."""
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def upsert(conn, table: Table, rows: list[dict], conflict_cols: list[str]):
    """Inserta filas; si ya existen (según conflict_cols) las actualiza."""
    if not rows:
        return
    stmt = pg_insert(table).values(rows)
    update_cols = {
        c.name: stmt.excluded[c.name]
        for c in table.columns
        if c.name not in conflict_cols
    }
    stmt = stmt.on_conflict_do_update(
        index_elements=conflict_cols, set_=update_cols
    )
    conn.execute(stmt)


def ip_to_decimal(ip) -> float | None:
    """Convierte 'inningsPitched' de la MLB (6.1 = 6 y 1/3) a decimal real 6.33.
    Así el cálculo de K/9 en la vista pitcher_last5 queda correcto."""
    if ip is None:
        return None
    whole, _, frac = str(ip).partition(".")
    thirds = {"": 0, "0": 0, "1": 1, "2": 2}.get(frac, 0)
    return round(int(whole) + thirds / 3, 2)


def num(stats: dict, key: str, default=0):
    """Lee un valor numérico de un dict de stats, tolerando ausencias."""
    val = stats.get(key, default)
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# --------------------------------------------------------------------------
# Ingesta: catálogo
# --------------------------------------------------------------------------
def ingest_teams(conn):
    data = get(f"{API}/teams", sportId=1, season=SEASON)
    rows = [{
        "team_id": t["id"],
        "name": t["name"],
        "abbreviation": t.get("abbreviation"),
        "league": t.get("league", {}).get("name"),
        "division": t.get("division", {}).get("name"),
    } for t in data["teams"]]
    upsert(conn, teams, rows, ["team_id"])
    print(f"  teams: {len(rows)} equipos")


def ingest_player(conn, player_id: int, team_id: int | None):
    """Trae detalles de un jugador (mano de bateo/lanzamiento, posición)."""
    data = get(f"{API}/people/{player_id}")
    p = data["people"][0]
    row = {
        "player_id": p["id"],
        "full_name": p["fullName"],
        "team_id": team_id,
        "primary_position": p.get("primaryPosition", {}).get("abbreviation"),
        "bat_side": p.get("batSide", {}).get("code"),
        "throw_side": p.get("pitchHand", {}).get("code"),
    }
    upsert(conn, players, [row], ["player_id"])
    

def ensure_players(conn, ids, team_id=None):
    """Registra en lote los jugadores que falten. Evita violar la FK de games."""
    ids = {i for i in ids if i}
    if not ids:
        return
    rows = []
    id_list = list(ids)
    # La API acepta lotes; troceamos por seguridad
    for i in range(0, len(id_list), 40):
        chunk = id_list[i:i + 40]
        data = get(f"{API}/people", personIds=",".join(map(str, chunk)))
        for p in data.get("people", []):
            rows.append({
                "player_id": p["id"],
                "full_name": p["fullName"],
                "team_id": p.get("currentTeam", {}).get("id") or team_id,
                "primary_position": p.get("primaryPosition", {}).get("abbreviation"),
                "bat_side": p.get("batSide", {}).get("code"),
                "throw_side": p.get("pitchHand", {}).get("code"),
            })
    upsert(conn, players, rows, ["player_id"])


# --------------------------------------------------------------------------
# Ingesta: juegos y estadísticas
# --------------------------------------------------------------------------
def ingest_schedule(conn, game_date: str) -> list[int]:
    """Registra los juegos del día y devuelve sus gamePk."""
    data = get(f"{API}/schedule", sportId=1, date=game_date,
               hydrate="probablePitcher")
    game_ids = []
    for d in data.get("dates", []):
        rows = []
        for g in d["games"]:
            home, away = g["teams"]["home"], g["teams"]["away"]
            rows.append({
                "game_id": g["gamePk"],
                "game_date": game_date,
                "home_team_id": home["team"]["id"],
                "away_team_id": away["team"]["id"],
                "home_pitcher_id": home.get("probablePitcher", {}).get("id"),
                "away_pitcher_id": away.get("probablePitcher", {}).get("id"),
                "status": g["status"]["detailedState"],
                "home_score": home.get("score"),
                "away_score": away.get("score"),
            })
            game_ids.append(g["gamePk"])

        # Los abridores deben existir en players antes de insertar los juegos
        pitcher_ids = {r["home_pitcher_id"] for r in rows} | {r["away_pitcher_id"] for r in rows}
        ensure_players(conn, pitcher_ids)

        upsert(conn, games, rows, ["game_id"])
    print(f"  games: {len(game_ids)} partidos el {game_date}")
    return game_ids


def ingest_boxscore(conn, game_id: int):
    """Extrae stats por jugador y por equipo de un juego terminado o en curso."""
    data = get(f"{API}/game/{game_id}/boxscore")
    p_rows, b_rows, t_rows = [], [], []

    for side in ("home", "away"):
        side_data = data["teams"][side]
        team_id = side_data["team"]["id"]
        is_home = side == "home"

        # Stats de equipo
        ts = side_data.get("teamStats", {})
        t_rows.append({
            "game_id": game_id, "team_id": team_id, "is_home": is_home,
            "runs": num(ts.get("batting", {}), "runs"),
            "hits": num(ts.get("batting", {}), "hits"),
            "errors": num(ts.get("fielding", {}), "errors"),
        })

        # Stats por jugador
        for pdata in side_data["players"].values():
            pid = pdata["person"]["id"]
            stats = pdata.get("stats", {})

            pit = stats.get("pitching", {})
            if pit and pit.get("inningsPitched") is not None:
                # Asegura que el pitcher exista en la tabla players
                
                p_rows.append({
                    "game_id": game_id, "player_id": pid, "team_id": team_id,
                    "is_home": is_home,
                    "innings_pitched": ip_to_decimal(pit.get("inningsPitched")),
                    "strikeouts": num(pit, "strikeOuts"),
                    "hits_allowed": num(pit, "hits"),
                    "walks": num(pit, "baseOnBalls"),
                    "earned_runs": num(pit, "earnedRuns"),
                    "home_runs_allowed": num(pit, "homeRuns"),
                    "pitch_count": num(pit, "numberOfPitches"),
                    "batters_faced": num(pit, "battersFaced"),
                })

            bat = stats.get("batting", {})
            if bat and bat.get("atBats") is not None and num(bat, "atBats") >= 0:
                b_rows.append({
                    "game_id": game_id, "player_id": pid, "team_id": team_id,
                    "is_home": is_home,
                    "at_bats": num(bat, "atBats"), "hits": num(bat, "hits"),
                    "doubles": num(bat, "doubles"), "triples": num(bat, "triples"),
                    "home_runs": num(bat, "homeRuns"), "runs": num(bat, "runs"),
                    "rbi": num(bat, "rbi"), "walks": num(bat, "baseOnBalls"),
                    "strikeouts": num(bat, "strikeOuts"),
                })
    ensure_players(conn, {r["player_id"] for r in p_rows} | {r["player_id"] for r in b_rows})
    upsert(conn, pitcher_game_stats, p_rows, ["game_id", "player_id"])
    upsert(conn, batter_game_stats, b_rows, ["game_id", "player_id"])
    upsert(conn, team_game_stats, t_rows, ["game_id", "team_id"])


# --------------------------------------------------------------------------
# Rutina principal
# --------------------------------------------------------------------------
def run_daily(game_date: str, teams_only: bool = False):
    with engine.begin() as conn:  # una transacción; hace rollback si algo falla
        print(f"Ingesta {game_date} (temporada {SEASON})")
        ingest_teams(conn)
        if teams_only:
            return
        game_ids = ingest_schedule(conn, game_date)
        for gid in game_ids:
            try:
                ingest_boxscore(conn, gid)
            except requests.HTTPError as e:
                # Un juego que aún no arranca puede no tener boxscore; se ignora.
                print(f"  (sin boxscore aún para {gid}: {e})")
        print("Listo.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingesta MLB → Supabase")
    ap.add_argument("--date", default=date.today().isoformat(),
                    help="Fecha YYYY-MM-DD (por defecto hoy)")
    ap.add_argument("--teams", action="store_true",
                    help="Refresca solo el catálogo de equipos")
    args = ap.parse_args()
    run_daily(args.date, teams_only=args.teams)