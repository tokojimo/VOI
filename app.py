"""
Usage:
1) pip install -r requirements.txt
2) streamlit run app.py
"""

from __future__ import annotations

import io
import json
from datetime import datetime, time, timedelta
from typing import Iterable, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PARIS_TZ = ZoneInfo("Europe/Paris")
MAX_WINDOW_HOURS_DEFAULT = 12


# -----------------------------
# Parsing utilities
# -----------------------------

def parse_timestamp(row: dict) -> Optional[pd.Timestamp]:
    """Parse a timestamp from multiple possible fields.

    Supported fields: t_iso (ISO 8601), t_ms (epoch ms), t as .NET /Date(ms)/.
    Returned timestamp is timezone-aware in Europe/Paris.
    """

    raw_iso = row.get("t_iso")
    raw_ms = row.get("t_ms")
    raw_misc = row.get("t")

    ts: Optional[pd.Timestamp] = None

    try:
        if pd.notna(raw_iso):
            ts = pd.to_datetime(raw_iso)
        elif pd.notna(raw_ms):
            ts = pd.to_datetime(int(raw_ms), unit="ms", utc=True)
        elif isinstance(raw_misc, str) and raw_misc.startswith("/Date("):
            ms_value = int(raw_misc.strip("/Date()"))
            ts = pd.to_datetime(ms_value, unit="ms", utc=True)
    except Exception:
        return None

    if ts is None:
        return None

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    return ts.tz_convert(PARIS_TZ)


# -----------------------------
# Data loading and enrichment
# -----------------------------

def read_ndjson_files(files: Iterable[io.BytesIO]) -> pd.DataFrame:
    frames = []
    for file in files:
        try:
            frames.append(pd.read_json(file, lines=True))
        except ValueError:
            file.seek(0)
            lines = [json.loads(l) for l in file.read().decode("utf-8").splitlines() if l.strip()]
            frames.append(pd.json_normalize(lines))
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    return data


@st.cache_data(show_spinner=False)
def load_events(files: Iterable[io.BytesIO]) -> pd.DataFrame:
    raw = read_ndjson_files(files)
    if raw.empty:
        return raw

    raw["ts"] = raw.apply(parse_timestamp, axis=1)
    events = raw.loc[raw["type"].isin(["IN", "OUT"])].copy()
    events = events.dropna(subset=["ts", "v"])
    events = events.sort_values("ts")
    events = events.drop_duplicates(subset=["v", "ts", "type", "station_id"])
    return events


@st.cache_data(show_spinner=False)
def load_stations(file: io.BytesIO) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    try:
        stations = pd.read_csv(file)
    except Exception:
        file.seek(0)
        stations = pd.read_csv(file, delimiter=";")
    expected_cols = {"station_id", "name", "lat", "lon"}
    missing = expected_cols - set(stations.columns)
    if missing:
        st.warning(f"Colonnes manquantes dans le CSV stations: {', '.join(sorted(missing))}")
    stations = stations.rename(columns={"station_id": "station_id", "name": "station_name"})
    return stations


def attach_station_metadata(events: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    if stations.empty:
        events["station_name"] = np.nan
        events["lat"] = np.nan
        events["lon"] = np.nan
        return events
    return events.merge(
        stations[["station_id", "station_name", "lat", "lon"]],
        on="station_id",
        how="left",
    )


def enrich_events(events: pd.DataFrame, stations: pd.DataFrame, window_hours: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    window_td = pd.Timedelta(hours=window_hours)

    events = attach_station_metadata(events, stations)

    ins = events.loc[events["type"] == "IN"].copy()
    outs = events.loc[events["type"] == "OUT"].copy()

    ins_sorted = ins.sort_values(["v", "ts"]).copy()
    outs_sorted = outs.sort_values(["v", "ts"]).copy()

    outs_for_merge = outs_sorted[
        ["v", "ts", "station_id", "station", "station_name", "lat", "lon"]
    ].rename(
        columns={
            "station_id": "origin_station_id",
            "station": "origin_station_code",
            "station_name": "origin_station_name",
            "lat": "origin_lat",
            "lon": "origin_lon",
        }
    )
    outs_for_merge["origin_ts"] = outs_for_merge["ts"]

    ins_enriched = pd.merge_asof(
        ins_sorted,
        outs_for_merge,
        on="ts",
        by="v",
        direction="backward",
        tolerance=window_td,
    )
    ins_enriched["minutes_since_prev_out"] = (
        ins_enriched["ts"] - ins_enriched["origin_ts"]
    ).dt.total_seconds() / 60

    ins_enriched["origin_station_label"] = ins_enriched[
        ["origin_station_name", "origin_station_code"]
    ].apply(lambda x: next((val for val in x if pd.notna(val) and val != ""), None), axis=1)

    ins_enriched["arrival_station_label"] = ins_enriched[
        ["station_name", "station"]
    ].apply(lambda x: next((val for val in x if pd.notna(val) and val != ""), None), axis=1)

    ins_enriched["origin_known"] = ins_enriched["origin_station_id"].notna()

    ins_enriched.loc[ins_enriched["origin_ts"].isna(), "minutes_since_prev_out"] = np.nan

    # Destination for OUT
    ins_for_merge = ins_sorted[
        ["v", "ts", "station_id", "station", "station_name", "lat", "lon"]
    ].rename(
        columns={
            "station_id": "dest_station_id",
            "station": "dest_station_code",
            "station_name": "dest_station_name",
            "lat": "dest_lat",
            "lon": "dest_lon",
        }
    )
    ins_for_merge["ts_dest"] = ins_for_merge["ts"]

    outs_enriched = pd.merge_asof(
        outs_sorted,
        ins_for_merge,
        on="ts",
        by="v",
        direction="forward",
        tolerance=window_td,
    )
    outs_enriched["minutes_to_next_in"] = (
        outs_enriched["ts_dest"] - outs_enriched["ts"]
    ).dt.total_seconds() / 60

    outs_enriched["destination_known"] = outs_enriched["dest_station_id"].notna()

    outs_enriched["departure_station_label"] = outs_enriched[
        ["station_name", "station"]
    ].apply(lambda x: x[0] or x[1], axis=1)
    outs_enriched["destination_station_label"] = outs_enriched[
        ["dest_station_name", "dest_station_code"]
    ].apply(lambda x: x[0] or x[1], axis=1)

    return ins_enriched, outs_enriched


@st.cache_data(show_spinner=False)
def build_all_tables(
    files: Iterable[io.BytesIO],
    stations_file: Optional[io.BytesIO],
    window_hours: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = load_events(files)
    stations = load_stations(stations_file) if stations_file else pd.DataFrame()
    ins_enriched, outs_enriched = enrich_events(events, stations, window_hours)
    return events, stations, ins_enriched, outs_enriched


# -----------------------------
# Filtering helpers
# -----------------------------

def apply_filters(
    df: pd.DataFrame,
    date_range: Tuple[datetime, datetime],
    time_range: Tuple[time, time],
    types: Iterable[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    start_date, end_date = date_range
    start_dt = datetime.combine(start_date, time_range[0]).replace(tzinfo=PARIS_TZ)
    end_dt = datetime.combine(end_date, time_range[1]).replace(tzinfo=PARIS_TZ)
    mask = (df["ts"] >= start_dt) & (df["ts"] <= end_dt)
    if types:
        mask &= df["type"].isin(types)
    return df.loc[mask]


# -----------------------------
# Visualization helpers
# -----------------------------

def build_kpis(events: pd.DataFrame, ins: pd.DataFrame, outs: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Arrivées (IN)", len(ins))
    with col2:
        st.metric("Départs (OUT)", len(outs))
    with col3:
        st.metric("Véhicules uniques", events["v"].nunique())
    with col4:
        top_arrival = ins["arrival_station_label"].value_counts().head(1)
        top_depart = outs["departure_station_label"].value_counts().head(1)
        arrival_label = top_arrival.index[0] if not top_arrival.empty else "-"
        depart_label = top_depart.index[0] if not top_depart.empty else "-"
        st.metric("Top arrivée / départ", f"{arrival_label} / {depart_label}")


def arrivals_tab(ins: pd.DataFrame, show_unknown: bool):
    st.subheader("Arrivées")
    if not show_unknown:
        ins = ins.loc[ins["origin_station_id"].notna()]
    st.write(f"Total arrivées: {len(ins)}")
    if ins.empty:
        st.info("Aucune arrivée pour les filtres sélectionnés.")
        return
    st.dataframe(
        ins[
            [
                "ts",
                "v",
                "arrival_station_label",
                "origin_station_label",
                "minutes_since_prev_out",
            ]
        ].rename(
            columns={
                "ts": "Horodatage",
                "v": "Véhicule",
                "arrival_station_label": "Station d'arrivée",
                "origin_station_label": "Station d'origine",
                "minutes_since_prev_out": "Minutes depuis dernier OUT",
            }
        ),
        use_container_width=True,
    )
    csv = ins.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger CSV des arrivées", csv, "arrivees.csv", "text/csv")


def departures_tab(outs: pd.DataFrame, show_unknown: bool):
    st.subheader("Départs")
    if not show_unknown:
        outs = outs.loc[outs["dest_station_id"].notna()]
    st.write(f"Total départs: {len(outs)}")
    if outs.empty:
        st.info("Aucun départ pour les filtres sélectionnés.")
        return
    st.dataframe(
        outs[
            [
                "ts",
                "v",
                "departure_station_label",
                "destination_station_label",
                "minutes_to_next_in",
            ]
        ].rename(
            columns={
                "ts": "Horodatage",
                "v": "Véhicule",
                "departure_station_label": "Station de départ",
                "destination_station_label": "Station de destination",
                "minutes_to_next_in": "Minutes jusqu'au prochain IN",
            }
        ),
        use_container_width=True,
    )
    csv = outs.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger CSV des départs", csv, "depart.csv", "text/csv")


def trips_tab(ins: pd.DataFrame):
    st.subheader("Trajets reconstitués")
    if ins.empty:
        st.info("Aucun trajet disponible.")
        return
    trips = (
        ins.groupby(["origin_station_label", "arrival_station_label"], dropna=False)
        .agg(
            trajets=("v", "count"),
            duree_moyenne_min=("minutes_since_prev_out", "mean"),
            duree_mediane_min=("minutes_since_prev_out", "median"),
        )
        .reset_index()
        .sort_values("trajets", ascending=False)
    )

    st.dataframe(trips, use_container_width=True)

    csv = trips.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger CSV des trajets", csv, "trajets.csv", "text/csv")

    trips_known = trips.dropna(subset=["origin_station_label", "arrival_station_label"])
    if trips_known.empty:
        st.info("Trajets insuffisants pour les visualisations.")
        return

    sankey_data = trips_known.head(50)  # limit to keep readable
    labels = pd.unique(
        pd.concat([
            sankey_data["origin_station_label"],
            sankey_data["arrival_station_label"],
        ])
    ).tolist()
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    sankey_fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, pad=15, thickness=10),
                link=dict(
                    source=[label_to_idx[o] for o in sankey_data["origin_station_label"]],
                    target=[label_to_idx[d] for d in sankey_data["arrival_station_label"]],
                    value=sankey_data["trajets"],
                ),
            )
        ]
    )
    sankey_fig.update_layout(title="Flux origine → destination (top 50)")
    st.plotly_chart(sankey_fig, use_container_width=True)


def timeline_tab(events: pd.DataFrame):
    st.subheader("Chronologie")
    if events.empty:
        st.info("Pas de données pour cette période.")
        return
    events = events.copy()
    events.set_index("ts", inplace=True)
    per_min = events.groupby("type").resample("1min").size().unstack(0).fillna(0)
    per_min_fig = px.line(per_min, labels={"value": "Compte", "ts": "Temps"})
    per_min_fig.update_layout(title="Événements par minute")
    st.plotly_chart(per_min_fig, use_container_width=True)

    events["day"] = events.index.date
    events["hour"] = events.index.hour
    heatmap = (
        events.groupby(["day", "hour", "type"]).size().reset_index(name="count")
    )
    if heatmap.empty:
        st.info("Pas assez de données pour la heatmap.")
        return
    fig = px.density_heatmap(
        heatmap,
        x="hour",
        y="day",
        z="count",
        facet_col="type",
        color_continuous_scale="Viridis",
        labels={"hour": "Heure", "day": "Jour"},
    )
    fig.update_layout(title="Heatmap heure/jour")
    st.plotly_chart(fig, use_container_width=True)


def map_tab(ins: pd.DataFrame, outs: pd.DataFrame, stations: pd.DataFrame):
    st.subheader("Carte")
    if stations.empty:
        st.info("Charger un CSV de stations pour la cartographie.")
        return

    ins_counts = ins.groupby("station_id").size().rename("in_count")
    outs_counts = outs.groupby("station_id").size().rename("out_count")
    station_counts = stations.set_index("station_id").join([ins_counts, outs_counts]).fillna(0)

    st.map(
        station_counts,
        latitude="lat",
        longitude="lon",
        size=10,
    )

    # Arcs for flows
    arcs_data = (
        ins.dropna(subset=["origin_lat", "origin_lon", "lat", "lon"])
        .groupby(
            [
                "origin_station_label",
                "arrival_station_label",
                "origin_lat",
                "origin_lon",
                "lat",
                "lon",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )

    if arcs_data.empty:
        st.info("Aucun flux cartographiable.")
        return

    arcs_layer = {
        "type": "ArcLayer",
        "data": arcs_data,
        "get_source_position": "[origin_lon, origin_lat]",
        "get_target_position": "[lon, lat]",
        "get_source_color": [255, 165, 0],
        "get_target_color": [0, 128, 255],
        "get_width": "count",
        "width_scale": 10,
        "pickable": True,
    }

    st.pydeck_chart(
        {
            "initialViewState": {
                "latitude": station_counts["lat"].mean(),
                "longitude": station_counts["lon"].mean(),
                "zoom": 11,
                "pitch": 30,
            },
            "layers": [arcs_layer],
            "mapStyle": "mapbox://styles/mapbox/light-v9",
        }
    )


# -----------------------------
# Streamlit interface
# -----------------------------

def main():
    st.set_page_config(page_title="VOI Grenoble - IN/OUT", layout="wide")
    st.title("Analyse des événements IN/OUT des véhicules Voi - Grenoble")
    st.caption(
        "Les trajets sont reconstruits à partir des événements et peuvent différer des trajets officiels."
    )

    with st.sidebar:
        st.header("Fichiers")
        ndjson_files = st.file_uploader(
            "NDJSON des événements", type=["ndjson", "json"], accept_multiple_files=True
        )
        stations_file = st.file_uploader("CSV des stations", type=["csv"])

        st.header("Filtres temporels")
        today = datetime.now(PARIS_TZ).date()
        date_range = st.date_input(
            "Période",
            value=(today - timedelta(days=7), today),
        )
        if not isinstance(date_range, tuple) or len(date_range) != 2:
            st.error("Veuillez sélectionner une période de début et fin.")
            return
        time_start = st.time_input("Heure début", value=time(0, 0))
        time_end = st.time_input("Heure fin", value=time(23, 59))

        st.header("Filtres stations")
        arrival_filter = st.text_input("Filtrer stations d'arrivée (codes séparés par virgule)")
        departure_filter = st.text_input("Filtrer stations de départ (codes séparés par virgule)")

        st.header("Options")
        event_types = st.multiselect(
            "Types d'événements", options=["IN", "OUT"], default=["IN", "OUT"]
        )
        window_hours = st.number_input(
            "Fenêtre max de couplage (heures)",
            min_value=1,
            max_value=48,
            value=MAX_WINDOW_HOURS_DEFAULT,
        )
        show_unknown = st.checkbox("Afficher les origines/destinations inconnues", value=True)

    if not ndjson_files:
        st.info("Chargez au moins un fichier NDJSON pour commencer.")
        return

    events, stations, ins_enriched, outs_enriched = build_all_tables(
        ndjson_files, stations_file, window_hours
    )

    if events.empty:
        st.error("Aucun événement IN/OUT valide trouvé.")
        return

    filtered_events = apply_filters(
        events,
        date_range=(date_range[0], date_range[1]),
        time_range=(time_start, time_end),
        types=event_types,
    )

    filtered_ins = apply_filters(
        ins_enriched,
        date_range=(date_range[0], date_range[1]),
        time_range=(time_start, time_end),
        types=["IN"],
    )

    filtered_outs = apply_filters(
        outs_enriched,
        date_range=(date_range[0], date_range[1]),
        time_range=(time_start, time_end),
        types=["OUT"],
    )

    if arrival_filter:
        allowed_arrivals = [x.strip() for x in arrival_filter.split(",") if x.strip()]
        filtered_ins = filtered_ins[filtered_ins["station"].isin(allowed_arrivals)]
    if departure_filter:
        allowed_departures = [x.strip() for x in departure_filter.split(",") if x.strip()]
        filtered_outs = filtered_outs[filtered_outs["station"].isin(allowed_departures)]

    build_kpis(filtered_events, filtered_ins, filtered_outs)

    tabs = st.tabs(["Arrivées", "Départs", "Trajets", "Chronologie", "Carte"])

    with tabs[0]:
        arrivals_tab(filtered_ins, show_unknown)
    with tabs[1]:
        departures_tab(filtered_outs, show_unknown)
    with tabs[2]:
        trips_tab(filtered_ins)
    with tabs[3]:
        timeline_tab(filtered_events)
    with tabs[4]:
        map_tab(filtered_ins, filtered_outs, stations)


if __name__ == "__main__":
    main()
