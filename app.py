"""
app.py
======
Interfaz web del sistema (Streamlit).

Reemplaza los cuatro comandos de consola por una app con formularios. Reutiliza
el mismo motor: importa las funciones de mlb_model.py y mlb_parlay.py, así que
no hay lógica duplicada — si cambiás el modelo, la app cambia con él.

Uso:
    pip install streamlit
    
    python -m streamlit run app.py.

Se abre en el navegador (http://localhost:8501).
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import text

from mlb_model import (
    build_predictions, engine, load_context, save_predictions, MODEL_VERSION,
)
from mlb_parlay import best_parlays, describe, load_candidates

# Nombre del script de ingesta. Tiene guion, así que no se puede importar:
# se ejecuta como proceso aparte.
INGEST = "mlb-ingest.py"

K_LINES = [4, 5, 6, 7, 8]
TEAM_RUN_LINES = [2.5, 3.5, 4.5]

# Rangos típicos de mercado. Fuera de esto casi siempre es un error de carga.
RANGOS = {
    ("team_total", 2.5): (1.15, 1.45),
    ("team_total", 3.5): (1.45, 2.10),
    ("team_total", 4.5): (2.00, 3.20),
    ("pitcher_strikeouts", 4.0): (1.10, 1.45),
    ("pitcher_strikeouts", 5.0): (1.25, 1.75),
    ("pitcher_strikeouts", 6.0): (1.55, 2.40),
    ("pitcher_strikeouts", 7.0): (2.10, 3.60),
    ("pitcher_strikeouts", 8.0): (3.00, 6.00),
}

st.set_page_config(page_title="Sistema MLB", page_icon="⚾", layout="wide")


# --------------------------------------------------------------------------
# Consultas
# --------------------------------------------------------------------------
def slate(d: str) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql(text("""
            select g.game_id, g.status,
                   ht.abbreviation as local, at.abbreviation as visita,
                   g.home_team_id, g.away_team_id,
                   g.home_pitcher_id, g.away_pitcher_id,
                   hp.full_name as lanzador_local,
                   ap.full_name as lanzador_visita,
                   g.home_score, g.away_score
            from games g
            join teams ht on ht.team_id = g.home_team_id
            join teams at on at.team_id = g.away_team_id
            left join players hp on hp.player_id = g.home_pitcher_id
            left join players ap on ap.player_id = g.away_pitcher_id
            where g.game_date = :d
            order by g.game_id
        """), conn, params={"d": d})


def odds_del_dia(d: str) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql(text("""
            select o.id, o.market_type, o.line, o.decimal_odds,
                   coalesce(pl.full_name, t.abbreviation) as sujeto,
                   at.abbreviation || ' @ ' || ht.abbreviation as partido
            from odds o
            join games g using (game_id)
            join teams ht on ht.team_id = g.home_team_id
            join teams at on at.team_id = g.away_team_id
            left join players pl on pl.player_id = o.player_id
            left join teams t on t.team_id = o.team_id
            where g.game_date = :d
            order by o.captured_at desc
        """), conn, params={"d": d})


def guardar_odds(row: dict):
    with engine.begin() as conn:
        conn.execute(text("""
            delete from odds
            where game_id = :game_id and market_type = :market_type
              and line = :line and side = :side
              and player_id is not distinct from :player_id
              and team_id is not distinct from :team_id
        """), row)
        conn.execute(text("""
            insert into odds (game_id, player_id, team_id, market_type, line,
                              side, decimal_odds, bookmaker)
            values (:game_id, :player_id, :team_id, :market_type, :line,
                    :side, :decimal_odds, :bookmaker)
        """), row)


def borrar_odd(odd_id: int):
    with engine.begin() as conn:
        conn.execute(text("delete from odds where id = :i"), {"i": odd_id})


def correr_ingesta(d: str) -> tuple[bool, str]:
    if not Path(INGEST).exists():
        return False, f"No encuentro {INGEST} en esta carpeta."
    r = subprocess.run(
        [sys.executable, INGEST, "--date", d],
        capture_output=True, text=True,
    )
    return r.returncode == 0, (r.stdout or "") + (r.stderr or "")


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
st.sidebar.title("⚾ Sistema MLB")
fecha = st.sidebar.date_input("Fecha", value=date.today())
fecha_iso = fecha.isoformat()
st.sidebar.caption(f"Modelo: {MODEL_VERSION}")

juegos = slate(fecha_iso)
cuotas = odds_del_dia(fecha_iso)

st.sidebar.metric("Partidos", len(juegos))
st.sidebar.metric("Cuotas cargadas", len(cuotas))

if len(juegos) and not len(cuotas):
    st.sidebar.info("Sin cuotas no hay picks: el modelo puede calcular "
                    "probabilidades, pero no sabe si conviene apostarlas.")

tab_hoy, tab_cuotas, tab_picks, tab_multi = st.tabs(
    ["Partidos", "Cuotas", "Picks", "Múltiples"]
)


# --------------------------------------------------------------------------
# Pestaña: partidos
# --------------------------------------------------------------------------
with tab_hoy:
    st.subheader(f"Partidos del {fecha_iso}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Actualizar este día", width="stretch"):
            with st.spinner("Consultando la MLB Stats API…"):
                ok, salida = correr_ingesta(fecha_iso)
            st.code(salida.strip() or "(sin salida)")
            if ok:
                st.rerun()
            else:
                st.error("La ingesta falló.")
    with c2:
        ayer = (fecha - timedelta(days=1)).isoformat()
        if st.button(f"Cerrar resultados de {ayer}", width="stretch",
                     help="La forma reciente se calcula con los juegos "
                          "anteriores: si quedaron a medias, el modelo "
                          "trabaja con datos incompletos."):
            with st.spinner("Actualizando…"):
                ok, salida = correr_ingesta(ayer)
            st.code(salida.strip() or "(sin salida)")

    if juegos.empty:
        st.warning("No hay partidos cargados para esta fecha.")
    else:
        vista = juegos.assign(
            partido=juegos["visita"] + " @ " + juegos["local"],
            marcador=juegos.apply(
                lambda r: "—" if pd.isna(r["away_score"])
                else f"{int(r['away_score'])}-{int(r['home_score'])}", axis=1),
            abridores=juegos["lanzador_visita"].fillna("?") + " vs "
                      + juegos["lanzador_local"].fillna("?"),
        )
        st.dataframe(
            vista[["partido", "abridores", "status", "marcador"]],
            width="stretch", hide_index=True,
        )


# --------------------------------------------------------------------------
# Pestaña: cuotas
# --------------------------------------------------------------------------
with tab_cuotas:
    st.subheader("Registrar cuotas de DoradoBet")

    if juegos.empty:
        st.warning("Cargá primero los partidos del día.")
    else:
        st.caption("Anotá los mercados que veas, no solo los que te gusten: "
                   "si filtrás por gusto, la muestra queda sesgada y el "
                   "análisis posterior te va a dar la razón por construcción.")

        etiquetas = [f"{r['visita']} @ {r['local']}"
                     for _, r in juegos.iterrows()]
        elegido = st.selectbox("Partido", range(len(etiquetas)),
                               format_func=lambda i: etiquetas[i])
        g = juegos.iloc[elegido]

        mercado = st.radio("Mercado",
                           ["Ponches del abridor", "Carreras de un equipo"],
                           horizontal=True)

        with st.form("form_cuota", clear_on_submit=True):
            if mercado == "Ponches del abridor":
                opciones = {
                    f"{g['lanzador_visita'] or '?'} ({g['visita']})":
                        g["away_pitcher_id"],
                    f"{g['lanzador_local'] or '?'} ({g['local']})":
                        g["home_pitcher_id"],
                }
                quien = st.selectbox("Lanzador", list(opciones))
                linea = st.selectbox("Línea (ponches o más)", K_LINES)
                market_type, player_id, team_id = (
                    "pitcher_strikeouts", opciones[quien], None
                )
            else:
                opciones = {g["visita"]: g["away_team_id"],
                            g["local"]: g["home_team_id"]}
                quien = st.selectbox("Equipo", list(opciones))
                linea = st.selectbox("Línea (más de)", TEAM_RUN_LINES)
                market_type, player_id, team_id = (
                    "team_total", None, opciones[quien]
                )

            cuota = st.number_input("Cuota decimal", min_value=1.01,
                                    max_value=20.0, value=1.30, step=0.01)
            enviar = st.form_submit_button("Guardar")

        if enviar:
            if player_id is not None and pd.isna(player_id):
                st.error("Ese abridor no está cargado todavía.")
            else:
                rango = RANGOS.get((market_type, float(linea)))
                fuera = rango and not (rango[0] <= cuota <= rango[1])
                guardar_odds({
                    "game_id": int(g["game_id"]),
                    "player_id": None if player_id is None else int(player_id),
                    "team_id": None if team_id is None else int(team_id),
                    "market_type": market_type, "line": float(linea),
                    "side": "over", "decimal_odds": float(cuota),
                    "bookmaker": "DoradoBet",
                })
                if fuera:
                    st.warning(
                        f"Guardado, pero {cuota} es raro para la línea "
                        f"{linea} (lo normal es {rango[0]}–{rango[1]}). "
                        f"Revisá que la línea sea la correcta antes de "
                        f"apostar: un error acá inventa ventajas que no existen."
                    )
                else:
                    st.success("Guardado.")
                st.rerun()

    st.divider()
    st.markdown("**Cargadas hoy**")
    if cuotas.empty:
        st.caption("Ninguna todavía.")
    else:
        for _, o in cuotas.iterrows():
            c1, c2 = st.columns([6, 1])
            tipo = "K" if o["market_type"] == "pitcher_strikeouts" else "carreras"
            c1.write(f"`{o['partido']}` · **{o['sujeto']}** "
                     f"{o['line']}+ {tipo} — {o['decimal_odds']}")
            if c2.button("Borrar", key=f"del{o['id']}"):
                borrar_odd(int(o["id"]))
                st.rerun()


# --------------------------------------------------------------------------
# Pestaña: picks
# --------------------------------------------------------------------------
with tab_picks:
    st.subheader("Probabilidades del modelo")

    if st.button("Calcular y guardar", type="primary"):
        with st.spinner("Corriendo el modelo…"):
            with engine.begin() as conn:
                gm, pit, tm, od = load_context(conn, fecha_iso)
                if gm.empty:
                    st.warning("No hay partidos para esta fecha.")
                else:
                    preds = build_predictions(gm, pit, tm, od)
                    if preds.empty:
                        st.warning("El modelo no produjo picks.")
                    else:
                        save_predictions(conn, preds, fecha_iso)
                        st.session_state["preds"] = preds
                        st.success(f"{len(preds)} picks calculados.")

    preds = st.session_state.get("preds")
    if preds is not None and not preds.empty:
        solo_con_cuota = st.checkbox("Solo los que tienen cuota cargada",
                                     value=True)
        v = preds.copy()
        if solo_con_cuota:
            v = v[v["decimal_odds"].notna()]
        v = v.sort_values("model_probability", ascending=False)

        st.dataframe(
            v[["label", "expected", "model_probability",
               "decimal_odds", "implied_probability", "edge"]]
            .rename(columns={
                "label": "pick", "expected": "proyección",
                "model_probability": "prob. modelo",
                "decimal_odds": "cuota",
                "implied_probability": "prob. casa", "edge": "ventaja",
            }),
            width="stretch", hide_index=True,
        )
        st.caption("La ventaja es prob. modelo menos prob. casa. Positiva "
                   "significa que el modelo cree que la casa paga de más. "
                   "Desconfiá de ventajas grandes: casi siempre son un error "
                   "de carga antes que un error de la casa.")


# --------------------------------------------------------------------------
# Pestaña: múltiples
# --------------------------------------------------------------------------
with tab_multi:
    st.subheader("Constructor de múltiples")

    c1, c2, c3, c4 = st.columns(4)
    patas = c1.number_input("Patas", 2, 5, 3)
    cuota_min = c2.number_input("Cuota mínima", 1.0, 20.0, 2.0, 0.1)
    cuota_max = c3.number_input("Cuota máxima", 1.0, 50.0, 3.0, 0.1)
    prob_min = c4.slider("Prob. mínima por pata", 0.50, 0.90, 0.60, 0.05)

    if st.button("Buscar combinaciones", type="primary"):
        with engine.begin() as conn:
            cands = load_candidates(conn, fecha_iso, prob_min)

        if cands.empty:
            st.warning("No hay predicciones con cuota para esta fecha. "
                       "Cargá cuotas y corré el modelo primero.")
        else:
            st.caption(f"{len(cands)} patas candidatas")
            res = best_parlays(cands, int(patas), cuota_min, cuota_max, 5)

            if not res:
                st.warning("Ninguna combinación cae en ese rango de cuota.")
            else:
                for n, r in enumerate(res, 1):
                    with st.container(border=True):
                        a, b, c = st.columns(3)
                        a.metric("Cuota", f"{r['cuota']:.2f}")
                        b.metric("Probabilidad", f"{r['probabilidad']*100:.1f}%")
                        c.metric("Valor esperado", f"{r['ev']:.2f}")
                        for f in r["patas"]:
                            st.write(
                                f"• {describe(f)} — **{float(f['decimal_odds']):.2f}** "
                                f"({float(f['model_probability'])*100:.0f}%)"
                            )
                        if r["ev"] < 1.0:
                            st.caption(
                                "Valor esperado bajo 1.00: aunque el modelo "
                                "acierte, el pago no compensa las veces que "
                                "falla."
                            )

    st.divider()
    st.caption(
        "Una cuota combinada de 2.5 con 3 patas implica patas de ~72% cada "
        "una y una probabilidad conjunta cercana al 37%: la múltiple falla "
        "más veces de las que acierta. No es un defecto de la selección, es "
        "lo que ese pago significa."
    )
