"""Simple charts from ICDAS counts (0–4 only)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .theme import ICDAS_COLORS


def icdas_charts(distribution: dict) -> None:
    grades = [str(i) for i in range(5)]
    values = [int(distribution.get(g, 0) or 0) for g in grades]
    df = pd.DataFrame({"ICDAS": grades, "Count": values})
    c1, c2 = st.columns(2)
    with c1:
        st.bar_chart(df.set_index("ICDAS"))
    with c2:
        nonempty = df[df["Count"] > 0]
        if nonempty.empty:
            st.info("No classified teeth yet.")
        else:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(3.2, 3.2))
                ax.pie(
                    nonempty["Count"],
                    labels=["ICDAS " + x for x in nonempty["ICDAS"]],
                    colors=[ICDAS_COLORS[int(x)] for x in nonempty["ICDAS"]],
                    autopct="%1.0f%%",
                )
                ax.set_title("ICDAS share")
                st.pyplot(fig, clear_figure=True)
            except Exception:
                st.dataframe(nonempty, hide_index=True, use_container_width=True)
    st.caption("Colors: 0 green · 1 lime · 2 yellow · 3 orange · 4 red. ICDAS 5–6 are not used.")
    chips = " ".join(
        f'<span style="background:{ICDAS_COLORS[i]};color:#fff;padding:3px 8px;'
        f'border-radius:8px;margin-right:6px;">ICDAS {i}: {values[i]}</span>'
        for i in range(5)
    )
    st.markdown(chips, unsafe_allow_html=True)
