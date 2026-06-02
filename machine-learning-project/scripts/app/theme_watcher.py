"""OS·Streamlit 테마 변경 시 앱 자동 rerun (matplotlib 등 서버 렌더 요소 동기화)."""

from __future__ import annotations

import re

import streamlit as st
import streamlit.components.v1 as components

_SESSION_KEY = "_kbo_theme_signature"

_THEME_WATCHER_HTML = """
<script>
(function () {
  const win = window.parent;
  const doc = win.document;
  const root = doc.documentElement;
  const Streamlit = win.Streamlit || window.Streamlit;
  if (!Streamlit) return;

  function themeSignature() {
    const main =
      doc.querySelector('[data-testid="stAppViewContainer"]') ||
      doc.querySelector('[data-testid="stApp"]') ||
      doc.body;
    const mainCs = win.getComputedStyle(main);
    const sidebar = doc.querySelector('[data-testid="stSidebar"]');
    const panelBg = sidebar
      ? win.getComputedStyle(sidebar).backgroundColor
      : mainCs.backgroundColor;
    const scheme = win.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
    const dataTheme = root.getAttribute("data-theme") || "";
    return [panelBg, mainCs.color, scheme, dataTheme].join("|");
  }

  let last = null;
  let debounceTimer = null;

  function publish() {
    const sig = themeSignature();
    if (sig === last) return;
    last = sig;
    Streamlit.setComponentValue(sig);
  }

  function publishDebounced() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(publish, 40);
  }

  publish();

  win
    .matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", publish);

  const observer = new MutationObserver(publishDebounced);
  observer.observe(root, {
    attributes: true,
    attributeFilter: ["class", "style", "data-theme", "color-scheme"],
  });

  const app = doc.querySelector('[data-testid="stApp"]');
  if (app) {
    observer.observe(app, {
      attributes: true,
      attributeFilter: ["class", "style"],
      subtree: true,
    });
  }

  setInterval(publishDebounced, 800);
})();
</script>
"""


def mount_theme_watcher() -> None:
    sig = components.html(_THEME_WATCHER_HTML, height=0, width=0)
    if isinstance(sig, str) and sig.strip():
        st.session_state[_SESSION_KEY] = sig.strip()


def theme_signature_from_session() -> str | None:
    raw = st.session_state.get(_SESSION_KEY)
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


def is_light_from_watcher() -> bool | None:
    sig = theme_signature_from_session()
    if not sig:
        return None
    parts = sig.split("|")
    if not parts:
        return None
    lum = css_color_luminance(parts[0])
    if lum is not None:
        return lum > 0.5
    if len(parts) > 2 and parts[2] in ("light", "dark"):
        return parts[2] == "light"
    if len(parts) > 3 and parts[3] in ("light", "dark"):
        return parts[3] == "light"
    return None


def css_color_luminance(css_color: str) -> float | None:
    m = re.match(
        r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)",
        css_color.strip(),
    )
    if not m:
        return None
    r, g, b = (float(m.group(i)) for i in range(1, 4))
    if max(r, g, b) > 1.0:
        r, g, b = r / 255.0, g / 255.0, b / 255.0
    return 0.299 * r + 0.587 * g + 0.114 * b
