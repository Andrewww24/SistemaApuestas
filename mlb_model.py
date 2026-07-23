"""
mlb_model.py
============
Motor de probabilidad (Fase 2).

Lee de Supabase las vistas pitcher_last5, pitcher_season y team_batting_last10,
proyecta valores esperados para los partidos del día y convierte esas
proyecciones en probabilidades usando la distribución de Poisson. Si hay cuotas
cargadas en la tabla odds, calcula el edge y ordena los picks.

Escribe en la tabla predictions.

Uso:
    python mlb_model.py                      # partidos de hoy
    python mlb_model.py --date 2026-07-23
    python mlb_model.py --min-edge 0.05      # solo picks con 5%+ de ventaja
    python mlb_model.py --dry-run            # calcula y muestra, no guarda

Requisitos: los mismos del script de ingesta, más pandas (no necesita scipy).
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import date

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
MODEL_VERSION = "v1.1-poisson"
CONFIDENCE_CAP = 0.93   # ningún evento de un solo juego es más seguro que esto

# Líneas típicas que evaluamos para cada mercado.
K_LINES = [4, 5, 6, 7, 8]          # "4+", "5+", ... ponches del abridor
TEAM_RUN_LINES = [2.5, 3.5, 4.5]   # "más de X" carreras de un equipo

engine = create_engine(DATABASE_URL, future=True)


# --------------------------------------------------------------------------
# Poisson (implementado a mano para no depender de scipy)
# --------------------------------------------------------------------------
def poisson_pmf(k: int, lam: float) -> float:
    """Probabilidad de exactamente k eventos si el promedio esperado es lam."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def poisson_at_least(k: int, lam: float) -> float:
    """P(X >= k). Es lo que necesitamos para un mercado '4+' o 'más de 3.5'."""
    if k <= 0:
        return 1.0
    below = sum(poisson_pmf(i, lam) for i in range(k))
    return max(0.0, min(1.0, 1.0 - below))


# --------------------------------------------------------------------------
# Carga de datos
# --------------------------------------------------------------------------
def load_context(conn, game_date: str):
    """Trae los partidos del día más las vistas de forma reciente."""
    games = pd.read_sql(text("""
        select g.game_id, g.game_date,
               g.home_team_id, g.away_team_id,
               g.home_pitcher_id, g.away_pitcher_id,
               ht.abbreviation as home_abbr, at.abbreviation as away_abbr
        from games g
        join teams ht on ht.team_id = g.home_team_id
        join teams at on at.team_id = g.away_team_id
        where g.game_date = :d
        order by g.game_id
    """), conn, params={"d": game_date})

    pitchers = pd.read_sql(text("""
        select l.*, p.full_name,
               s.k9 as k9_season, s.avg_ip as avg_ip_season
        from pitcher_last5 l
        join players p using (player_id)
        left join pitcher_season s using (player_id)
    """), conn).set_index("player_id")

    teams = pd.read_sql(
        text("select * from team_batting_last10"), conn
    ).set_index("team_id")

    odds = pd.read_sql(text("""
        select o.* from odds o
        join games g using (game_id)
        where g.game_date = :d
    """), conn, params={"d": game_date})

    return games, pitchers, teams, odds


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def project_strikeouts(pitcher, opp_team, league_k_rate: float) -> float | None:
    """Ponches esperados = ritmo del lanzador x entradas esperadas x ajuste rival.

    El ritmo mezcla forma reciente y nivel de temporada (shrinkage), para no
    sobrerreaccionar a una racha de 5 juegos.
    """
    if pitcher is None or opp_team is None:
        return None

    ip = float(pitcher["avg_ip"] or 0)
    k9_recent = float(pitcher["k9"] or 0)

    season_val = pitcher.get("k9_season")
    k9_season = float(season_val) if pd.notna(season_val) else k9_recent

    ip_recent = float(pitcher["total_ip"] or 0)
    w = min(ip_recent / 30.0, 0.60)          # tope: 60% de peso a lo reciente
    k9 = w * k9_recent + (1 - w) * k9_season

    if k9 <= 0 or ip <= 0:
        return None

    opp_k_rate = float(opp_team["k_rate"] or 0)
    adjustment = opp_k_rate / league_k_rate if league_k_rate > 0 else 1.0
    # Acotamos: ninguna ofensiva es 50% peor o mejor de forma sostenida.
    adjustment = max(0.80, min(1.20, adjustment))

    return (k9 / 9) * ip * adjustment


def project_team_runs(team, opp_pitcher, league_era: float) -> float | None:
    """Carreras esperadas de un equipo, ajustadas por el abridor rival.

    Un equipo que promedia 5 carreras debería proyectarse más bajo contra un as
    y más alto contra un abridor castigado. El factor compara la efectividad del
    rival con la media de la liga.
    """
    if team is None:
        return None
    runs = float(team["avg_runs"] or 0)
    if runs <= 0:
        return None

    if opp_pitcher is not None and league_era > 0:
        era_val = opp_pitcher.get("era")
        if pd.notna(era_val) and float(era_val) > 0:
            factor = float(era_val) / league_era
            # Acotado: ni el mejor as reduce a la mitad, ni el peor duplica.
            runs *= max(0.70, min(1.30, factor))

    return runs


# --------------------------------------------------------------------------
# Generación de picks
# --------------------------------------------------------------------------
def build_predictions(games, pitchers, teams, odds) -> pd.DataFrame:
    league_k_rate = float(teams["k_rate"].mean()) if len(teams) else 0.0
    # Mediana y no promedio: un relevista con 3 carreras en 1/3 de entrada
    # tiene ERA de 81 y arrastraría el promedio de la liga hacia arriba.
    league_era = float(pitchers["era"].median()) if "era" in pitchers else 0.0
    rows = []

    def look(df, key):
        return df.loc[key] if key in df.index else None

    for _, g in games.iterrows():
        matchups = [
            (g["home_pitcher_id"], g["away_team_id"]),
            (g["away_pitcher_id"], g["home_team_id"]),
        ]

        # --- Props de ponches del abridor ---
        for pitcher_id, opp_id in matchups:
            pitcher = look(pitchers, pitcher_id)
            opp = look(teams, opp_id)
            lam = project_strikeouts(pitcher, opp, league_k_rate)
            if lam is None:
                continue
            for line in K_LINES:
                prob = min(poisson_at_least(line, lam), CONFIDENCE_CAP)
                if prob < 0.50:    # solo picks que el modelo cree probables
                    continue
                rows.append({
                    "game_id": g["game_id"],
                    "player_id": int(pitcher_id),
                    "label": f"{pitcher['full_name']} {line}+ K",
                    "market_type": "pitcher_strikeouts",
                    "line": line,
                    "side": "over",
                    "expected": round(lam, 2),
                    "model_probability": round(prob, 4),
                })

        # --- Totales de carreras por equipo ---
        for team_id, opp_id, opp_pitcher_id in (
            (g["home_team_id"], g["away_team_id"], g["away_pitcher_id"]),
            (g["away_team_id"], g["home_team_id"], g["home_pitcher_id"]),
        ):
            team = look(teams, team_id)
            opp_pitcher = look(pitchers, opp_pitcher_id)
            lam = project_team_runs(team, opp_pitcher, league_era)
            if lam is None:
                continue
            abbr = g["home_abbr"] if team_id == g["home_team_id"] else g["away_abbr"]
            for line in TEAM_RUN_LINES:
                # "más de 2.5" se cumple con 3 o más carreras
                prob = min(poisson_at_least(math.ceil(line), lam), CONFIDENCE_CAP)
                if prob < 0.50:
                    continue
                rows.append({
                    "game_id": g["game_id"],
                    "player_id": None,
                    "label": f"{abbr} más de {line} carreras",
                    "market_type": "team_total",
                    "line": line,
                    "side": "over",
                    "expected": round(lam, 2),
                    "model_probability": round(prob, 4),
                })

    preds = pd.DataFrame(rows)
    if preds.empty:
        return preds

    # --- Cruce con cuotas para calcular edge ---
    preds["decimal_odds"] = None
    preds["implied_probability"] = None
    preds["edge"] = None

    if not odds.empty:
        key = ["game_id", "market_type", "line", "side"]
        o = odds.copy()
        o["player_id"] = o["player_id"].astype("Int64")
        preds["player_id"] = preds["player_id"].astype("Int64")
        merged = preds.merge(
            o[key + ["player_id", "decimal_odds"]],
            on=key + ["player_id"], how="left", suffixes=("", "_o"),
        )
        merged["decimal_odds"] = merged["decimal_odds_o"].fillna(merged["decimal_odds"])
        merged = merged.drop(columns=["decimal_odds_o"])
        merged["implied_probability"] = (1 / merged["decimal_odds"]).round(4)
        merged["edge"] = (
            merged["model_probability"] - merged["implied_probability"]
        ).round(4)
        preds = merged

    return preds


# --------------------------------------------------------------------------
# Persistencia
# --------------------------------------------------------------------------
def save_predictions(conn, preds: pd.DataFrame, game_date: str):
    """Reemplaza las predicciones del día (para poder recalcular sin duplicar)."""
    conn.execute(text("""
        delete from predictions
        where game_id in (select game_id from games where game_date = :d)
          and model_version = :v
    """), {"d": game_date, "v": MODEL_VERSION})

    cols = ["game_id", "player_id", "market_type", "line", "side",
            "model_probability", "implied_probability", "edge", "decimal_odds"]
    out = preds[cols].copy()
    out["model_version"] = MODEL_VERSION
    out.to_sql("predictions", conn, if_exists="append", index=False)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(game_date: str, min_edge: float | None, dry_run: bool):
    with engine.begin() as conn:
        games, pitchers, teams, odds = load_context(conn, game_date)
        if games.empty:
            print(f"No hay partidos cargados para {game_date}. "
                  f"Corré primero la ingesta.")
            return

        preds = build_predictions(games, pitchers, teams, odds)
        if preds.empty:
            print("El modelo no produjo picks (¿faltan datos de forma reciente?).")
            return

        shown = preds
        if min_edge is not None:
            shown = shown[shown["edge"].notna() & (shown["edge"] >= min_edge)]
            shown = shown.sort_values("edge", ascending=False)
        else:
            shown = shown.sort_values("model_probability", ascending=False)

        print(f"\n{len(preds)} picks calculados para {game_date} "
              f"({len(games)} partidos)\n")
        if shown.empty:
            print("Ningún pick supera el edge mínimo pedido.")
        else:
            print(shown[["label", "expected", "model_probability",
                         "decimal_odds", "edge"]].head(25).to_string(index=False))

        if dry_run:
            print("\n(dry-run: no se guardó nada)")
        else:
            save_predictions(conn, preds, game_date)
            print(f"\nGuardados en predictions ({MODEL_VERSION}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Motor de probabilidad MLB")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--min-edge", type=float, default=None,
                    help="Filtra picks con al menos este edge (ej. 0.05)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args.date, args.min_edge, args.dry_run)

# --------------------------------------------------------------------------
# Limitaciones conocidas de v1.1 (documentadas a propósito)
# --------------------------------------------------------------------------
# 1. El ajuste por abridor rival usa ERA de 5 juegos, métrica ruidosa. FIP o
#    xERA serían mejores estimadores del nivel real de un lanzador.
# 2. No modela el bullpen. Un equipo con relevo pésimo debería permitir más
#    carreras tardías de lo que sugiere solo su abridor.
# 3. Poisson asume independencia entre eventos; en béisbol hay correlación
#    (un rally genera más turnos al bate). Tiende a subestimar la cola alta.
# 4. No distingue casa/visita ni zurdo/derecho. Ambos son splits reales y
#    medibles con los datos que ya guardás (is_home, bat_side, throw_side).
# 5. La proyección de entradas usa el promedio reciente del lanzador, sin
#    considerar conteo de lanzamientos ni si el equipo lo cuida.
# 6. CONFIDENCE_CAP es una estimación, no un valor calibrado. Ajustalo cuando
#    tengas resultados reales en la tabla results (Fase 4).