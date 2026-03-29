from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from backend_client import BackendRequestError, PoseGuardBackendClient


def _fmt_ts_ms(ts_ms: Optional[int]) -> str:
    if not ts_ms:
        return "—"
    return datetime.fromtimestamp(ts_ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_prob(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3f}"


def _status_chip(label: str, ok: bool) -> str:
    bg = "#DDEEE7" if ok else "#F4DFD5"
    fg = "#1D4338" if ok else "#7A5138"
    return (
        f"<span style='display:inline-block;padding:0.28rem 0.58rem;border-radius:999px;"
        f"background:{bg};color:{fg};font-size:0.8rem;font-weight:600;margin-right:0.35rem;'>{label}</span>"
    )


def render_backend_connection_controls() -> Dict[str, Any]:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Live Backend")
    base_url = st.sidebar.text_input(
        "FastAPI base URL",
        value=st.session_state.get("pg_backend_base_url", "http://127.0.0.1:8000"),
        help="The Streamlit product UI reads status and confirmed events from this FastAPI backend.",
    ).strip()
    st.session_state["pg_backend_base_url"] = base_url

    event_limit = st.sidebar.slider(
        "Confirmed event limit",
        min_value=10,
        max_value=100,
        value=int(st.session_state.get("pg_backend_event_limit", 25)),
        step=5,
    )
    st.session_state["pg_backend_event_limit"] = event_limit

    refresh_clicked = st.sidebar.button("Refresh backend", use_container_width=True)
    show_raw = st.sidebar.toggle(
        "Show raw backend JSON",
        value=bool(st.session_state.get("pg_backend_show_raw", False)),
        help="Useful during debugging or reviewer walkthroughs.",
    )
    st.session_state["pg_backend_show_raw"] = show_raw

    return {
        "base_url": base_url,
        "event_limit": event_limit,
        "refresh_clicked": refresh_clicked,
        "show_raw": show_raw,
    }


def _render_status_header(status: Dict[str, Any]) -> None:
    latest_inference = status.get("latest_inference") or {}
    event_manager = status.get("event_manager") or {}
    camera = status.get("camera") or {}

    ready = bool(status.get("ready"))
    inference_ready = bool(status.get("inference_ready"))
    event_ready = bool(status.get("event_ready"))
    camera_connected = bool(camera.get("connected"))

    chips = (
        _status_chip("Backend Ready", ready)
        + _status_chip("Inference Ready", inference_ready)
        + _status_chip("Event Manager Ready", event_ready)
        + _status_chip("Camera Connected", camera_connected)
    )
    st.markdown(chips, unsafe_allow_html=True)

    prob = float(latest_inference.get("fall_probability", 0.0) or 0.0)
    state = str(event_manager.get("state") or "NORMAL")
    prediction = latest_inference.get("predicted_label_name", "—")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current state", state)
    col2.metric("Fall probability", _fmt_prob(prob))
    col3.metric("Prediction", str(prediction))
    col4.metric("Camera", str(camera.get("camera_id") or "—"))
    st.progress(min(max(prob, 0.0), 1.0), text=f"Fall probability {prob:.3f}")


def _render_operational_snapshot(status: Dict[str, Any]) -> None:
    latest_inference = status.get("latest_inference") or {}
    latest_event = status.get("latest_event") or {}
    camera = status.get("camera") or {}
    event_manager = status.get("event_manager") or {}

    left, right = st.columns([1.1, 1.0], gap="large")
    with left:
        st.markdown("#### System and camera")
        st.write(f"**Video source:** `{camera.get('video_path', '—')}`")
        st.write(f"**Target FPS:** {camera.get('target_fps', '—')}")
        st.write(f"**Frames emitted:** {camera.get('frames_emitted', '—')}")
        st.write(f"**Frames skipped:** {camera.get('frames_skipped', '—')}")
        st.write(f"**Last error:** {status.get('last_error') or 'none'}")

    with right:
        st.markdown("#### Inference and event state")
        st.write(f"**Window index:** {latest_inference.get('window_index', '—')}")
        st.write(f"**Threshold:** {latest_inference.get('threshold', '—')}")
        st.write(f"**Current event:** {latest_event.get('event_id', '—')}")
        st.write(f"**Cooldown active:** {bool(event_manager.get('in_cooldown'))}")
        st.write(f"**Last confirmed event:** {event_manager.get('last_confirmed_event') or '—'}")


def _render_events(client: PoseGuardBackendClient, events: List[Dict[str, Any]]) -> None:
    st.markdown("#### Confirmed events")
    if not events:
        st.info("No confirmed fall events have been persisted yet.")
        return

    header = st.columns([2.5, 1.2, 1.2, 1.4, 1.4, 1.0, 1.0])
    header[0].markdown("**Event ID**")
    header[1].markdown("**State**")
    header[2].markdown("**ACK**")
    header[3].markdown("**Peak Prob.**")
    header[4].markdown("**Confirmed At**")
    header[5].markdown("**Action**")
    header[6].markdown("**Dismiss**")

    for event in events:
        event_id = str(event.get("event_id", "—"))
        ack_status = str(event.get("ack_status", "PENDING"))
        cols = st.columns([2.5, 1.2, 1.2, 1.4, 1.4, 1.0, 1.0])
        cols[0].write(event_id)
        cols[1].write(str(event.get("state", "—")))
        cols[2].write(ack_status)
        cols[3].write(_fmt_prob(event.get("peak_probability")))
        cols[4].write(_fmt_ts_ms(event.get("confirmed_at_ts_ms")))

        ack_disabled = ack_status.upper() == "ACKNOWLEDGED"
        if cols[5].button("ACK", key=f"ack_{event_id}", disabled=ack_disabled, use_container_width=True):
            try:
                client.ack_event(event_id)
                st.success(f"Acknowledged {event_id}")
                st.rerun()
            except BackendRequestError as exc:
                st.error(str(exc))

        if cols[6].button("Dismiss", key=f"dismiss_{event_id}", use_container_width=True):
            try:
                client.dismiss_event(event_id)
                st.success(f"Dismissed {event_id}")
                st.rerun()
            except BackendRequestError as exc:
                st.error(str(exc))


def _render_events_table(events: List[Dict[str, Any]]) -> None:
    if not events:
        return
    rows = []
    for item in events:
        rows.append(
            {
                "event_id": item.get("event_id"),
                "state": item.get("state"),
                "ack_status": item.get("ack_status"),
                "peak_probability": round(float(item.get("peak_probability", 0.0)), 4),
                "confirmed_at": _fmt_ts_ms(item.get("confirmed_at_ts_ms")),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_live_backend_panel(base_url: str, event_limit: int = 25, show_raw: bool = False) -> None:
    st.markdown("### Connected live backend")
    st.caption("This product-facing Streamlit view reads system status and confirmed events from the FastAPI backend.")

    client = PoseGuardBackendClient(base_url)
    try:
        status = client.get_status()
        events_payload = client.get_events(limit=event_limit)
    except BackendRequestError as exc:
        st.error(str(exc))
        st.info("Start the FastAPI backend first, then verify the base URL matches the running server.")
        return

    events = list(events_payload.get("items") or [])

    _render_status_header(status)
    _render_operational_snapshot(status)
    st.markdown("---")
    _render_events(client, events)
    st.markdown("#### Event table")
    _render_events_table(events)

    if show_raw:
        with st.expander("Raw /api/v1/status JSON", expanded=False):
            st.json(status)
        with st.expander("Raw /api/v1/events JSON", expanded=False):
            st.json(events_payload)
