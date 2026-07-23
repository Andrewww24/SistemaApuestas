"""
mlb_parlay.py
=============
Constructor de múltiples (Fase 3).

Toma las predicciones del día que ya tienen cuota cargada y busca la
combinación de N patas cuya cuota combinada cae en el rango pedido,
maximizando la probabilidad conjunta.

Regla importante: no combina dos patas del mismo partido. Los eventos de un
mismo juego están correlacionados (si el abridor se derrumba, caen a la vez el
prop de ponches y el total de carreras), y multiplicar sus probabilidades como
si fueran independientes sobreestima la probabilidad real de la múltiple.

Uso:
    python mlb_parlay.py                          # hoy, 3 patas, cuota 2.0-3.0
    python mlb_parlay.py --date 2026-07-24
    python mlb_parlay.py --legs 2 --min-odds 1.8 --max-odds 2.5
    python mlb_parlay.py --top 5                  # muestra las 5 mejores

Requiere que existan cuotas cargadas (mlb_odds.py) y predicciones guardadas
(mlb_model.py sin --dry-run) para esa fecha.
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from itertools import combinations

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

engine = create_engine(os.environ["DATABASE_URL"], future=True)


def load_candidates(conn, game_date: str, min_prob: float) -> pd.DataFrame:
    """Predicciones del día que tienen cuota y superan la probabilidad mínima."""
    return pd.read_sql(text("""
        select pr.game_id, pr.player_id, pr.team_id, pr.market_type, pr.line,
               pr.side, pr.model_probability, pr.decimal_odds, pr.edge,
               ht.abbreviation as local, at.abbreviation as visita,
               tt.abbreviation as equipo,
               pl.full_name as jugador
        from predictions pr
        join games g using (game_id)
        join teams ht on ht.team_id = g.home_team_id
        join teams at on at.team_id = g.away_team_id
        left join teams tt on tt.team_id = pr.team_id
        left join players pl on pl.player_id = pr.player_id
        where g.game_date = :d
          and pr.decimal_odds is not null
          and pr.model_probability >= :p
        order by pr.model_probability desc
    """), conn, params={"d": game_date, "p": min_prob})


def describe(row) -> str:
    if row["market_type"] == "pitcher_strikeouts":
        quien = row["jugador"] or f"jugador {row['player_id']}"
        return f"{quien} {int(row['line'])}+ K"
    equipo = row.get("equipo") or "?"
    rival = row["local"] if equipo == row["visita"] else row["visita"]
    return (f"{equipo} más de {row['line']} carreras "
            f"(vs {rival})")


def best_parlays(cands: pd.DataFrame, legs: int,
                 min_odds: float, max_odds: float, top: int) -> list[dict]:
    """Evalúa todas las combinaciones posibles y devuelve las mejores.

    Con ~40 candidatos y 3 patas son ~10.000 combinaciones: instantáneo.
    """
    resultados = []
    idx = list(cands.index)

    for combo in combinations(idx, legs):
        filas = [cands.loc[i] for i in combo]

        # Sin dos patas del mismo partido: estarían correlacionadas.
        juegos = {f["game_id"] for f in filas}
        if len(juegos) < legs:
            continue

        cuota = 1.0
        prob = 1.0
        for f in filas:
            cuota *= float(f["decimal_odds"])
            prob *= float(f["model_probability"])

        if not (min_odds <= cuota <= max_odds):
            continue

        resultados.append({
            "patas": filas,
            "cuota": cuota,
            "probabilidad": prob,
            # Valor esperado por cada 1 apostado. >1 significa que, SI el
            # modelo tiene razón, la apuesta es rentable a largo plazo.
            "ev": prob * cuota,
        })

    resultados.sort(key=lambda r: r["probabilidad"], reverse=True)
    return resultados[:top]


def mostrar(resultados: list[dict], game_date: str):
    if not resultados:
        print("\nNinguna combinación cae en el rango de cuota pedido.")
        print("Probá ampliar el rango, bajar --min-prob o cargar más cuotas.")
        return

    print(f"\nMejores múltiples para {game_date}\n")
    for n, r in enumerate(resultados, 1):
        print(f"  #{n}  cuota {r['cuota']:.2f}  ·  "
              f"probabilidad {r['probabilidad']*100:.1f}%  ·  "
              f"EV {r['ev']:.2f}")
        for f in r["patas"]:
            print(f"       {describe(f):<45} {float(f['decimal_odds']):.2f}  "
                  f"({float(f['model_probability'])*100:.0f}%)")
        print()

    mejor = resultados[0]
    print(f"La mejor combinación acierta, según el modelo, "
          f"{mejor['probabilidad']*100:.0f} de cada 100 veces.")
    if mejor["ev"] < 1.0:
        print(f"EV de {mejor['ev']:.2f}: aun si el modelo acierta, esta "
              f"combinación pierde dinero a largo plazo.")
    else:
        print(f"EV de {mejor['ev']:.2f}: rentable a largo plazo SI el modelo "
              f"está bien calibrado (verificalo con el backtesting).")


def run(game_date, legs, min_odds, max_odds, min_prob, top):
    with engine.begin() as conn:
        cands = load_candidates(conn, game_date, min_prob)
        if cands.empty:
            print(f"No hay predicciones con cuota para {game_date}.")
            print("Corré: mlb-ingest.py → mlb_odds.py → mlb_model.py")
            return

        print(f"{len(cands)} patas candidatas "
              f"(probabilidad >= {min_prob*100:.0f}%)")
        resultados = best_parlays(cands, legs, min_odds, max_odds, top)
        mostrar(resultados, game_date)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Constructor de múltiples MLB")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--legs", type=int, default=3, help="Número de patas")
    ap.add_argument("--min-odds", type=float, default=2.0)
    ap.add_argument("--max-odds", type=float, default=3.0)
    ap.add_argument("--min-prob", type=float, default=0.60,
                    help="Probabilidad mínima por pata (0.60 = 60%%)")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    run(args.date, args.legs, args.min_odds, args.max_odds,
        args.min_prob, args.top)

# --------------------------------------------------------------------------
# Sobre el rango de cuota
# --------------------------------------------------------------------------
# Una cuota combinada de 2.5 con 3 patas implica patas de ~72% cada una, y una
# probabilidad conjunta cercana al 37%: la múltiple falla más veces de las que
# acierta. Eso no es un defecto de la selección, es lo que ese pago significa.
# Si preferís acertar seguido, bajá --max-odds a 1.6-1.8; si preferís pagos
# altos, aceptá que la mayoría se caen. No existe la combinación de ambos.
#
# El campo EV es la única medida que dice si conviene: probabilidad x cuota.
# Por debajo de 1.00 la apuesta pierde dinero a largo plazo aunque el modelo
# acierte, porque el pago no compensa las veces que falla.
