"""
mlb_backtest.py
===============
Backtesting de calibración (Fase 4).

Corre el modelo sobre un rango de fechas pasadas y compara lo que predijo
contra lo que efectivamente pasó. No necesita cuotas: responde la pregunta
anterior a la del valor, que es si el modelo predice bien.

Un modelo bien calibrado cumple: de los picks a los que asignó 80%, aciertan
cerca del 80%. Si asigna 80% y aciertan 60%, es un modelo optimista y todo lo
que construyas encima va a perder dinero, tenga o no buenas cuotas.

Uso:
    python mlb_backtest.py --from 2026-05-01 --to 2026-07-20
    python mlb_backtest.py --from 2026-06-01 --to 2026-07-20 --market pitcher_strikeouts

Depende de mlb_model.py (lo importa para reutilizar el modelo exacto).
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import text

from mlb_model import build_predictions, engine, load_context, MODEL_VERSION


# --------------------------------------------------------------------------
# Resultados reales
# --------------------------------------------------------------------------
def load_actuals(conn, desde: str, hasta: str):
    """Ponches reales por lanzador y carreras reales por equipo."""
    ks = pd.read_sql(text("""
        select pgs.game_id, pgs.player_id, pgs.strikeouts
        from pitcher_game_stats pgs
        join games g using (game_id)
        where g.game_date between :a and :b
    """), conn, params={"a": desde, "b": hasta})

    runs = pd.read_sql(text("""
        select tgs.game_id, tgs.team_id, tgs.runs
        from team_game_stats tgs
        join games g using (game_id)
        where g.game_date between :a and :b
    """), conn, params={"a": desde, "b": hasta})

    return (
        ks.set_index(["game_id", "player_id"])["strikeouts"].to_dict(),
        runs.set_index(["game_id", "team_id"])["runs"].to_dict(),
    )


def settle(row, ks: dict, runs: dict):
    """Devuelve (valor_real, acertó) o (None, None) si no hay dato."""
    gid = int(row["game_id"])

    if row["market_type"] == "pitcher_strikeouts":
        pid = row["player_id"]
        if pd.isna(pid):
            return None, None
        real = ks.get((gid, int(pid)))
    else:
        tid = row.get("team_id")
        if pd.isna(tid):
            return None, None
        real = runs.get((gid, int(tid)))

    if real is None:
        # El abridor probable cambió, el juego se suspendió, etc.
        return None, None

    # El "over" se cumple con line o más (línea entera: 5+ K = 5 o más;
    # línea .5: más de 2.5 = 3 o más). El "under" es lo complementario.
    umbral = int(row["line"]) if float(row["line"]).is_integer() \
        else int(float(row["line"])) + 1
    acerto = real >= umbral if row["side"] == "over" else real < umbral
    return real, acerto


# --------------------------------------------------------------------------
# Corrida
# --------------------------------------------------------------------------
def run_range(desde: str, hasta: str, market: str | None):
    d0 = date.fromisoformat(desde)
    d1 = date.fromisoformat(hasta)
    todas = []

    with engine.begin() as conn:
        ks, runs = load_actuals(conn, desde, hasta)

        d = d0
        while d <= d1:
            iso = d.isoformat()
            games, pitchers, teams, odds = load_context(conn, iso)
            if not games.empty:
                preds = build_predictions(games, pitchers, teams, odds)
                if not preds.empty:
                    preds["game_date"] = iso
                    todas.append(preds)
            d += timedelta(days=1)

    if not todas:
        print("No se generaron predicciones en ese rango.")
        return

    df = pd.concat(todas, ignore_index=True)
    if market:
        df = df[df["market_type"] == market]

    resultados = df.apply(lambda r: settle(r, ks, runs), axis=1, result_type="expand")
    df["real"] = resultados[0]
    df["acerto"] = resultados[1]

    sin_dato = df["acerto"].isna().sum()
    df = df[df["acerto"].notna()].copy()
    df["acerto"] = df["acerto"].astype(bool)

    if df.empty:
        print("Ninguna predicción pudo verificarse contra resultados reales.")
        return

    reportar(df, sin_dato)


# --------------------------------------------------------------------------
# Reporte
# --------------------------------------------------------------------------
def reportar(df: pd.DataFrame, sin_dato: int):
    print(f"\n{'='*62}")
    print(f"BACKTESTING · {MODEL_VERSION}")
    print(f"{'='*62}")
    print(f"Predicciones verificadas: {len(df):,}"
          f"   (sin resultado: {sin_dato:,})")
    print(f"Aciertos: {df['acerto'].mean()*100:.1f}%"
          f"   ·   Probabilidad media predicha: "
          f"{df['model_probability'].mean()*100:.1f}%")

    # Brier score: error cuadrático medio de la probabilidad.
    # 0 = perfecto. 0.25 = equivale a decir siempre 50%.
    brier = ((df["model_probability"] - df["acerto"].astype(int)) ** 2).mean()
    print(f"Brier score: {brier:.4f}   (más bajo es mejor; 0.25 = sin valor)")

    print(f"\n{'-'*62}")
    print("CALIBRACIÓN POR TRAMO")
    print(f"{'-'*62}")
    print(f"{'tramo':<14}{'n':>7}{'predicho':>11}{'real':>9}{'sesgo':>10}")

    bins = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]
    df["tramo"] = pd.cut(df["model_probability"], bins=bins, right=False)

    for tramo, grupo in df.groupby("tramo", observed=True):
        if len(grupo) == 0:
            continue
        pred = grupo["model_probability"].mean()
        real = grupo["acerto"].mean()
        sesgo = pred - real
        marca = "  <-- optimista" if sesgo > 0.05 else (
                "  <-- pesimista" if sesgo < -0.05 else "")
        etiqueta = f"{tramo.left*100:.0f}-{min(tramo.right,1.0)*100:.0f}%"
        print(f"{etiqueta:<14}{len(grupo):>7}{pred*100:>10.1f}%"
              f"{real*100:>8.1f}%{sesgo*100:>+9.1f}%{marca}")

    print(f"\n{'-'*62}")
    print("POR MERCADO")
    print(f"{'-'*62}")
    for mercado, grupo in df.groupby("market_type"):
        pred = grupo["model_probability"].mean()
        real = grupo["acerto"].mean()
        print(f"{mercado:<24}{len(grupo):>7}{pred*100:>10.1f}%"
              f"{real*100:>8.1f}%{(pred-real)*100:>+9.1f}%")

    print(f"\n{'-'*62}")
    print("POR LADO")
    print(f"{'-'*62}")
    for lado, grupo in df.groupby("side"):
        pred = grupo["model_probability"].mean()
        real = grupo["acerto"].mean()
        print(f"{lado:<24}{len(grupo):>7}{pred*100:>10.1f}%"
              f"{real*100:>8.1f}%{(pred-real)*100:>+9.1f}%")

    print(f"\n{'-'*62}")
    print("CÓMO LEERLO")
    print(f"{'-'*62}")
    print("La columna 'sesgo' es lo que importa: predicho menos real.")
    print("Positivo = el modelo promete más de lo que entrega.")
    print("Un sesgo de +10% en el tramo 80-85% significa que esos picks")
    print("aciertan 10 puntos menos de lo anunciado; una múltiple armada")
    print("con ellos falla mucho más seguido de lo que el sistema cree.")
    print("\nCon menos de ~100 casos por tramo, el ruido domina: no saques")
    print("conclusiones de un tramo con 12 observaciones.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Backtesting de calibración")
    ap.add_argument("--from", dest="desde", required=True)
    ap.add_argument("--to", dest="hasta", required=True)
    ap.add_argument("--market", default=None,
                    choices=["pitcher_strikeouts", "team_total"])
    args = ap.parse_args()
    run_range(args.desde, args.hasta, args.market)
