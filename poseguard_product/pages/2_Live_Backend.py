from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from live_backend_panel import render_backend_connection_controls, render_live_backend_panel

st.set_page_config(page_title="PoseGuard Live Backend", layout="wide")

st.title("PoseGuard Live Backend")
st.caption(
    "Use this page for the caregiver / product view. It reads live system status and confirmed alerts from the FastAPI backend."
)

controls = render_backend_connection_controls()
render_live_backend_panel(
    base_url=controls["base_url"],
    event_limit=controls["event_limit"],
    show_raw=controls["show_raw"],
)
