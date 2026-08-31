"""Downloadable HTML report (print to PDF from the browser)."""

from __future__ import annotations


def html_bytes(html: str) -> bytes:
    return (html or "").encode("utf-8")
