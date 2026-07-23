"""
mlb_odds.py
===========
Carga cuotas a mano en la tabla odds, sin escribir SQL.

Las casas no publican historial de cuotas, así que este registro solo se puede
construir hacia adelante: lo que no anotes antes del primer lanzamiento se
pierde para siempre. Ese archivo es lo que después permite saber si el modelo
encuentra valor real o solo repite lo que el mercado ya sabe.

Uso:
    python mlb_odds.py                  # partidos de hoy
    python mlb_odds.py --date 2026-07-24
    python mlb_odds.py --list           # solo muestra los mercados del día

Requisitos: los mismos del modelo.
"""

from __future__ import annotations

import argparse
import os
from datetime import date

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

engine = create_engine(os.environ["DATABASE_URL"], future=True)

# Deben coincidir con las que evalúa mlb_model.py, o el cruce no encuentra nada.
K_LINES = [4, 5, 6, 7, 8]
TEAM_RUN_LINES = [2.5, 3.5, 4.5]


def load_slate(conn, game_date: str) -> pd.DataFrame:
    """Partidos del día con sus abridores probables."""
    return pd.read_sql(text("""
        select g.game_id,
               ht.abbreviation as local, at.abbreviation as visita,
               g.home_pitcher_id, g.away_pitcher_id,
               hp.full_name as lanzador_local,
               ap.full_name as lanzador_visita
        from games g
        join teams ht on ht.team_id = g.home_team_id
        join teams at on at.team_id = g.away_team_id
        left join players hp on hp.player_id = g.home_pitcher_id
        left join players ap on ap.player_id = g.away_pitcher_id
        where g.game_date = :d
        order by g.game_id
    """), conn, params={"d": game_date})


def existing_odds(conn, game_date: str) -> pd.DataFrame:
    return pd.read_sql(text("""
        select o.game_id, o.player_id, o.market_type, o.line, o.side,
               o.decimal_odds
        from odds o
        join games g using (game_id)
        where g.game_date = :d
    """), conn, params={"d": game_date})


def show_slate(slate: pd.DataFrame, odds: pd.DataFrame):
    print(f"\n{len(slate)} partidos\n")
    for i, r in slate.iterrows():
        n = len(odds[odds["game_id"] == r["game_id"]]) if not odds.empty else 0
        marca = f"  [{n} cuotas]" if n else ""
        print(f"  [{i}] {r['visita']} @ {r['local']}{marca}")
        print(f"       {r['lanzador_visita'] or '?'} vs "
              f"{r['lanzador_local'] or '?'}")


def ask(prompt: str, options: list[str] | None = None) -> str | None:
    """Pide un dato; ENTER vacío cancela y vuelve al menú."""
    while True:
        val = input(prompt).strip()
        if not val:
            return None
        if options is None or val in options:
            return val
        print(f"    Opciones válidas: {', '.join(options)}")


def insert_odds(conn, row: dict):
    """Guarda una cuota; si ya existe ese mercado, la reemplaza."""
    conn.execute(text("""
        delete from odds
        where game_id = :game_id
          and market_type = :market_type
          and line = :line
          and side = :side
          and player_id is not distinct from :player_id
    """), row)
    conn.execute(text("""
        insert into odds (game_id, player_id, market_type, line, side,
                          decimal_odds, bookmaker)
        values (:game_id, :player_id, :market_type, :line, :side,
                :decimal_odds, :bookmaker)
    """), row)


def capture_strikeouts(conn, game):
    """Prop de ponches de uno de los dos abridores."""
    print(f"\n  1) {game['lanzador_visita'] or '?'} ({game['visita']})")
    print(f"  2) {game['lanzador_local'] or '?'} ({game['local']})")
    quien = ask("  Lanzador [1/2]: ", ["1", "2"])
    if quien is None:
        return
    pid = game["away_pitcher_id"] if quien == "1" else game["home_pitcher_id"]
    if pd.isna(pid):
        print("    Ese abridor no está cargado todavía.")
        return

    linea = ask(f"  Línea {K_LINES}: ", [str(x) for x in K_LINES])
    if linea is None:
        return
    cuota = ask("  Cuota decimal (ej. 1.52): ")
    if cuota is None:
        return

    insert_odds(conn, {
        "game_id": int(game["game_id"]), "player_id": int(pid),
        "market_type": "pitcher_strikeouts", "line": float(linea),
        "side": "over", "decimal_odds": float(cuota),
        "bookmaker": "DoradoBet",
    })
    print(f"    Guardado: {linea}+ K a {cuota}")


def capture_team_total(conn, game):
    """Total de carreras de uno de los dos equipos."""
    print(f"\n  1) {game['visita']}   2) {game['local']}")
    quien = ask("  Equipo [1/2]: ", ["1", "2"])
    if quien is None:
        return

    linea = ask(f"  Línea {TEAM_RUN_LINES}: ",
                [str(x) for x in TEAM_RUN_LINES])
    if linea is None:
        return
    cuota = ask("  Cuota decimal: ")
    if cuota is None:
        return

    # El modelo identifica el equipo por el game_id + lado, no por team_id,
    # así que basta con guardar el mercado; el cruce lo resuelve el modelo.
    insert_odds(conn, {
        "game_id": int(game["game_id"]), "player_id": None,
        "market_type": "team_total", "line": float(linea),
        "side": "over", "decimal_odds": float(cuota),
        "bookmaker": "DoradoBet",
    })
    print(f"    Guardado: más de {linea} carreras a {cuota}")


def run(game_date: str, solo_listar: bool):
    with engine.begin() as conn:
        slate = load_slate(conn, game_date)
        if slate.empty:
            print(f"No hay partidos cargados para {game_date}. "
                  f"Corré primero la ingesta.")
            return

        odds = existing_odds(conn, game_date)
        print(f"\n=== Cuotas para {game_date} ===")
        show_slate(slate, odds)

        if solo_listar:
            return

        print("\nENTER vacío en cualquier momento cancela y vuelve acá.")
        while True:
            elegido = ask("\nNúmero de partido (o ENTER para salir): ")
            if elegido is None:
                break
            try:
                game = slate.iloc[int(elegido)]
            except (ValueError, IndexError):
                print("  Número inválido.")
                continue

            print(f"\n  {game['visita']} @ {game['local']}")
            print("  1) Ponches del abridor")
            print("  2) Total de carreras de un equipo")
            tipo = ask("  Mercado [1/2]: ", ["1", "2"])
            if tipo == "1":
                capture_strikeouts(conn, game)
            elif tipo == "2":
                capture_team_total(conn, game)

        total = len(existing_odds(conn, game_date))
        print(f"\n{total} cuotas registradas para {game_date}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Registro de cuotas")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--list", action="store_true",
                    help="Solo muestra los partidos y cuántas cuotas tienen")
    args = ap.parse_args()
    run(args.date, args.list)

# --------------------------------------------------------------------------
# Nota sobre sesgo de selección
# --------------------------------------------------------------------------
# Registrá los mercados que veas, no los que te gusten. Si solo anotás las
# cuotas de picks que ya te convencieron, la muestra queda sesgada y el
# backtesting va a mostrar un rendimiento mejor que el real. Un registro
# aburrido y completo vale más que uno curado.
