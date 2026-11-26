"""Small, restrained grade visualizations shared by patient and admin views."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .theme import ICDAS_COLORS


def icdas_charts(distribution: dict) -> None:
    grades = [str(i) for i in range(5)]
    values = [int(distribution.get(g, 0) or 0) for g in grades]
    total = sum(values)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.2, 2.8), layout="constrained")
        fig.patch.set_facecolor("#fffefa")
        ax.set_facecolor("#fffefa")
        if total:
            bars = ax.barh(
                [f"Grade {g}" for g in grades],
                values,
                color=[ICDAS_COLORS[i] for i in range(5)],
                height=.55,
                edgecolor="none",
            )
            ax.invert_yaxis()
            ax.set_xlim(0, max(values) * 1.18)
            ax.grid(axis="x", color="#e9ebe5", linewidth=.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            ax.tick_params(axis="y", length=0, labelcolor="#405266", labelsize=9, pad=9)
            ax.tick_params(axis="x", length=0, labelcolor="#7a8793", labelsize=8)
            ax.set_xticks([x for x in ax.get_xticks() if x >= 0])
            ax.set_title("ICDAS grade distribution", loc="left", color="#203448", fontsize=11, fontweight="semibold", pad=12)
            for bar, value in zip(bars, values):
                if value:
                    ax.text(value + max(values) * .025, bar.get_y() + bar.get_height() / 2, f"{value:,}",
                            va="center", ha="left", color="#536477", fontsize=9)
        else:
            ax.axis("off")
            ax.text(.02, .7, "ICDAS grade distribution", color="#203448", fontsize=11, fontweight="semibold", transform=ax.transAxes)
            ax.text(.02, .35, "No classified crops yet", color="#778596", fontsize=9, transform=ax.transAxes)
        st.pyplot(fig, clear_figure=True, use_container_width=True)
        plt.close(fig)
    except Exception:
        df = pd.DataFrame({"ICDAS": grades, "Count": values})
        st.bar_chart(df.set_index("ICDAS"), color="#397e7b")

    st.caption("Grade counts are taken from the saved classifier results and labels. ICDAS 5–6 are out of scope.")
