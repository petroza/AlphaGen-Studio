"""Pravý panel – nastavení ve třech režimech: Automaticky / Jednoduše / Expert.

* Automaticky: vybereš jen rychlost (Turbo / Rychlý / Normální / Vysoká kvalita) a výstup,
  vše ostatní aplikace zvolí podle videa a grafické karty (ověřené limity). Před spuštěním
  ukáže, co použije, a odhad času z předchozích výpočtů na tomto PC.
* Jednoduše: rychlost + pár srozumitelných úprav hrany.
* Expert: všechny parametry – s lidskými názvy a krátkým vysvětlením pod každou volbou.
"""

from __future__ import annotations

import random
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..export import FORMAT_LABELS, ExportSettings
from ..gpu import GpuInfo, MemoryPlan, recommend
from ..postprocess import MatteSettings
from ..segmentation import HARD_MAX_FRAMES, RECOMMENDED_MAX_FRAMES, compute_geometry, plan_segments
from ..video_io import VideoInfo
from .widgets import LabeledSlider, Section, dim_label, panel_title

MODES = [
    ("auto", "Automatic", "Pick only the speed and output – everything else is set from the clip and your GPU."),
    ("simple", "Simple", "Speed plus a few basic edge adjustments."),
    ("expert", "Expert", "All compute, matte and export parameters."),
]
RESOLUTIONS = [("Full (up to 1920×1088)", 0), ("1280 px", 1280), ("1024 px", 1024),
               ("960 px", 960), ("768 px", 768), ("640 px", 640), ("512 px", 512)]
# Rychlostní profily: (klíč, název, max. delší strana masky (0 = plné), překryv úseků, proti blikání, popis)
PROFILES = [
    ("turbo", "Turbo", 768, 0, 0.3, "Matte at 768 px, refined against the source. Fastest – for previews and rough keys."),
    ("fast", "Fast ★", 1280, 8, 0.3, "Recommended. Matte up to 1280 px, smooth transitions – 2× faster than Normal, "
     "the shape matches it by 93–98 %."),
    ("normal", "Normal", 0, 8, 0.3, "Full resolution (up to Full HD). Finest hair and motion blur – "
     "2× slower than Fast, best for the final key."),
    ("high", "High quality", 0, 16, 0.5, "Full resolution, longer segment blending and stronger anti-flicker."),
]
PROFILE_KEYS = [p[0] for p in PROFILES]
DEFAULT_PROFILE = "fast"  # grafik po prvním spuštění nic nenastavuje
DEFAULTS_VERSION = 3  # zvýšit při změně výchozích hodnot – starší uložený stav se jednou přepne na nové
MAIN_FORMATS = ("prores4444", "matte_png", "matte_h264", "preview")  # ostatní v roletce „Další formáty“


class Row(QWidget):
    """Popisek + ovládací prvek + jednořádkové vysvětlení pod ním."""

    def __init__(self, label: str, widget: QWidget, hint: str = "") -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 3, 0, 3)
        lay.setSpacing(2)
        top = QHBoxLayout()
        self.label = QLabel(label + ("  ⓘ" if hint else ""))
        self.label.setMinimumWidth(120)
        top.addWidget(self.label)
        top.addWidget(widget, 1)
        lay.addLayout(top)
        if hint:  # vysvětlivka jen po najetí myší
            self.label.setToolTip(hint)
            widget.setToolTip(hint)

    def show_hint(self, on: bool) -> None:  # kompatibilita – vysvětlivky jsou v bublinách
        pass


class SettingsPanel(QWidget):
    matteChanged = Signal(object)  # noqa: N815
    planChanged = Signal()  # noqa: N815

    def __init__(self, app_settings, gpus: list[GpuInfo], ram_total: float) -> None:
        super().__init__()
        self.setObjectName("Panel")
        self.app_settings = app_settings
        self.gpus = gpus
        self.ram_total = ram_total
        self.info: VideoInfo | None = None
        self.frame_range = (0, 0)
        self.mode = "auto"
        self.profile = DEFAULT_PROFILE
        self.estimator = None  # callable(geo, segs, vae) -> sekundy | None (nastaví hlavní okno)
        self._recommended: MemoryPlan | None = None
        self._loading = False
        self.rows: list[Row] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 6, 14, 12)
        lay.addWidget(panel_title("Settings"))

        # --- Přepínač režimu --------------------------------------------
        self.mode_buttons = self._button_bar(lay, [(k, n, t) for k, n, t in MODES], self.set_mode)

        # --- Rychlost (profily) -------------------------------------------
        self.g_profile = QGroupBox("Speed")
        pv = QVBoxLayout(self.g_profile)
        self.profile_buttons = self._button_bar(pv, [(k, n, d) for k, n, _, _, _, d in PROFILES], self.set_profile)
        self.profile_desc = dim_label("")
        pv.addWidget(self.profile_desc)
        lay.addWidget(self.g_profile)

        # --- Automatika: shrnutí -------------------------------------------
        self.g_auto = QGroupBox("Automatic settings")
        av = QVBoxLayout(self.g_auto)
        self.auto_label = QLabel("Load a clip – Automatic picks the settings from it and your GPU.")
        self.auto_label.setWordWrap(True)
        av.addWidget(self.auto_label)
        lay.addWidget(self.g_auto)

        # --- Výpočet (model a GPU) ---------------------------------------
        self.g_model = Section("Compute & memory")
        v = self.g_model.body
        self.gpu = QComboBox()
        for gi in gpus:
            self.gpu.addItem(gi.label(), gi.index)
        if not gpus:
            self.gpu.addItem("No NVIDIA GPU found", -1)
        self.gpu.currentIndexChanged.connect(self._gpu_changed)
        self.row_gpu = self._row(v, "GPU", self.gpu, "GPU used to compute the matte.")
        self.quant = self._combo([("Automatic", "auto"), ("None (most VRAM)", "none"),
                                  ("FP8 – half the memory", "fp8-cast"),
                                  ("FP8 fast (H100 and newer only)", "fp8-scaled-mm")])
        self.row_quant = self._row(v, "Quantization", self.quant,
                                   "Shrinks the model weights (22B parameters) to fit in VRAM. Virtually the same quality.")
        self.offload = self._combo([("Automatic", "auto"), ("Fully in VRAM (fastest)", "none"),
                                    ("In RAM, streamed to GPU", "cpu"), ("On disk (slowest)", "disk")])
        self.row_offload = self._row(v, "Model offload", self.offload,
                                     "If the model doesn't fit in VRAM, its layers are streamed in from RAM or disk.")
        self.vae = self._combo([("Automatic", "auto"), ("Accurate, per docs (more VRAM)", "diffvae"),
                                ("Lightweight", "conv")])
        self.row_vae = self._row(v, "Decoder (VAE)", self.vae,
                                 "Turns the model output into pixels. Lightweight allows Full HD even on a 12 GB card.")
        srow = QWidget()
        sl = QHBoxLayout(srow)
        sl.setContentsMargins(0, 0, 0, 0)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setValue(1234)
        sl.addWidget(self.seed, 1)
        rb = QPushButton("🎲")
        rb.setFixedWidth(32)
        rb.setToolTip("Random seed")
        rb.clicked.connect(lambda: self.seed.setValue(random.randint(0, 2**31 - 1)))
        sl.addWidget(rb)
        self.row_seed = self._row(v, "Seed", srow,
                                  "Same number = same result. Different number = another take (if the matte fails).")
        self.gpu_note = QLabel("ⓘ Recommendation for this GPU")
        self.gpu_note.setObjectName("Dim")
        v.addWidget(self.gpu_note)
        lay.addWidget(self.g_model)

        # --- Rozlišení a úseky (Expert) ------------------------------------
        self.g_res = Section("Resolution & segments")
        v = self.g_res.body
        self.resolution = self._combo(RESOLUTIONS)
        self.resolution.currentIndexChanged.connect(self._manual_changed)
        self.row_res = self._row(v, "Compute resolution", self.resolution,
                                 "Resolution the model computes the matte at. Output always matches the clip; lower = faster.")
        self.seg_len = QSpinBox()
        self.seg_len.setRange(9, HARD_MAX_FRAMES)
        self.seg_len.setSingleStep(8)
        self.seg_len.setValue(49)
        self.seg_len.valueChanged.connect(self._seg_changed)
        self.row_seg = self._row(v, "Frames per segment", self.seg_len,
                                 "Longer segment = fewer transitions but more VRAM. The model handles at most 145 frames (8n+1).")
        self.overlap = QSpinBox()
        self.overlap.setRange(0, 64)
        self.overlap.setValue(8)
        self.overlap.valueChanged.connect(self._manual_changed)
        self.row_overlap = self._row(v, "Segment overlap", self.overlap,
                                     "Frames blended where two segments meet, so the matte doesn't jump.")
        lay.addWidget(self.g_res)
        self.plan_label = dim_label("")
        lay.addWidget(self.plan_label)

        # --- Úprava masky --------------------------------------------------
        self.g_matte = Section("Matte refinement", expanded=True)
        self.g_matte.header.setToolTip("Updates live in the viewer without recomputing (then just use “Export only”).")
        v = self.g_matte.body
        self.g_matte_adv = Section("Advanced refinement", sub=True)
        av = self.g_matte_adv.body
        self.edge_refine = QCheckBox("Refine edges from image")
        self.edge_refine.setToolTip("Snaps the matte edge to edges in the source (guided filter). "
                                    "Useful when the matte was computed downscaled.")
        av.addWidget(self.edge_refine)
        self.edge_radius = LabeledSlider("  Refine radius", 1, 16, 4, 1, 0, " px")
        self.edge_strength = LabeledSlider("  Refine strength", 0, 1, 0.6, 0.05, 2)
        self.black = LabeledSlider("Clean background", 0, 0.5, 0.0, 0.005, 3,
                                   tooltip="Makes faint grey residue in the background fully transparent (black point).")
        self.white = LabeledSlider("Solidify subject", 0.5, 1.0, 1.0, 0.005, 3,
                                   tooltip="Makes nearly opaque areas of the subject fully solid (white point).")
        self.gamma = LabeledSlider("Semi-transparency density", 0.2, 3.0, 1.0, 0.05, 2,
                                   tooltip="Higher = denser smoke, hair and veils; lower = more transparent.")
        self.erode = LabeledSlider("Shrink / grow edge", -20, 20, 0, 0.5, 1, " px",
                                   "Negative shrinks the edge inward (removes fringe), positive grows it.")
        self.feather = LabeledSlider("Soften edge", 0, 20, 0, 0.1, 1, " px", "Blurs the edge transition.")
        self.temporal = LabeledSlider("Edge anti-flicker", 0, 1, 0.3, 0.05, 2,
                                      tooltip="Smooths the edge over time (3-frame median). Solid areas are untouched.")
        self.despeckle = LabeledSlider("Despeckle up to", 0, 5000, 0, 10, 0, " px²",
                                       "Removes small specks in the background and holes in the subject.")
        for w in (self.erode, self.feather, self.temporal):
            v.addWidget(w)
        self.invert = QCheckBox("Invert matte (keep background instead of subject)")
        v.addWidget(self.invert)
        for w in (self.edge_radius, self.edge_strength, self.black, self.white, self.gamma, self.despeckle):
            av.addWidget(w)
        v.addWidget(self.g_matte_adv)
        lay.addWidget(self.g_matte)

        self.g_color = Section("Edge color fringe")
        self.g_color.header.setToolTip("Alpha outputs only (ProRes 4444, PNG, EXR).")
        v = self.g_color.body
        self.despill = self._combo([("Keep colors", "off"), ("Remove original background color", "edge"),
                                    ("Remove green spill", "green"), ("Remove blue spill", "blue")])
        self.row_despill = self._row(v, "Despill", self.despill,
                                     "Affects only semi-transparent edge pixels. Green/blue only for green/blue screen shots.")
        self.despill_strength = LabeledSlider("Strength", 0, 1, 1.0, 0.05, 2)
        v.addWidget(self.despill_strength)
        lay.addWidget(self.g_color)

        # --- Výstup ----------------------------------------------------------
        self.g_export = QGroupBox("Output")
        v = QVBoxLayout(self.g_export)
        self.fmt_checks: dict[str, QCheckBox] = {}
        defaults = set(ExportSettings().formats)
        self.g_export_more = Section("More formats & options", sub=True)
        mv = self.g_export_more.body
        for key, label in FORMAT_LABELS.items():
            cb = QCheckBox(label)
            cb.setChecked(key in defaults)
            self.fmt_checks[key] = cb
            (v if key in MAIN_FORMATS else mv).addWidget(cb)
        self.alpha_mode = self._combo([("Straight – for Premiere/AE (recommended)", "straight"),
                                       ("Premultiplied", "premultiplied")])
        self.row_alpha = self._row(mv, "Alpha type", self.alpha_mode,
                                   "Applies to PNG and ProRes. EXR is always premultiplied (OpenEXR standard).")
        self.audio = QCheckBox("Include audio (ProRes, preview)")
        self.audio.setChecked(True)
        v.addWidget(self.audio)
        v.addWidget(self.g_export_more)
        self.exr_lin = QCheckBox("Linear EXR (for Nuke)")
        self.exr_lin.setToolTip("Off = EXR holds the clip's exact pixel values.")
        mv.addWidget(self.exr_lin)
        orow = QWidget()
        ol = QHBoxLayout(orow)
        ol.setContentsMargins(0, 0, 0, 0)
        self.out_dir = QLineEdit(app_settings.output_dir)
        ol.addWidget(self.out_dir, 1)
        ob = QPushButton("…")
        ob.setFixedWidth(30)
        ob.clicked.connect(self._pick_out)
        ol.addWidget(ob)
        self._row(v, "Folder", orow)
        self.base_name = QLineEdit()
        self.base_name.setPlaceholderText("auto (from input name)")
        self.row_name = self._row(mv, "Name", self.base_name)
        lay.addWidget(self.g_export)
        lay.addStretch(1)

        for cb in self.findChildren(QComboBox):  # dlouhé položky nesmí roztáhnout panel
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(12)
        for w in (self.edge_radius, self.edge_strength, self.black, self.white, self.gamma, self.erode,
                  self.feather, self.temporal, self.despeckle, self.despill_strength):
            w.valueChanged.connect(self._matte_changed)
        for w in (self.edge_refine, self.invert):
            w.toggled.connect(self._matte_changed)
        self.despill.currentIndexChanged.connect(self._matte_changed)
        self.vae.currentIndexChanged.connect(self._plan_changed)
        self._gpu_changed()
        self.set_profile(DEFAULT_PROFILE)
        self.set_mode("auto")

    # ------------------------------------------------------------------
    @staticmethod
    def _combo(items: list[tuple[str, object]]) -> QComboBox:
        cb = QComboBox()
        for label, val in items:
            cb.addItem(label, val)
        return cb

    @staticmethod
    def _button_bar(layout, items, callback) -> dict[str, QPushButton]:
        row = QHBoxLayout()
        row.setSpacing(2)
        buttons = {}
        for key, label, tip in items:
            b = QPushButton(label)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: callback(k))
            buttons[key] = b
            row.addWidget(b)
        layout.addLayout(row)
        return buttons

    def _row(self, layout: QVBoxLayout, label: str, widget: QWidget, hint: str = "") -> Row:
        r = Row(label, widget, hint)
        layout.addWidget(r)
        self.rows.append(r)
        return r

    def _pick_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    def current_gpu(self) -> GpuInfo | None:
        idx = self.gpu.currentData()
        return next((x for x in self.gpus if x.index == idx), None)

    def max_segment(self) -> int:
        """Nejdelší úsek, který se na zvolené kartě vejde (ověřené limity, viz gpu.recommend)."""
        plan = self._recommended
        return plan.recommended_segment if plan and plan.feasible and plan.recommended_segment else 121

    # ------------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        self.mode = mode
        for k, b in self.mode_buttons.items():
            b.setChecked(k == mode)
        auto, simple, expert = mode == "auto", mode == "simple", mode == "expert"
        self.g_auto.setVisible(auto)
        self.g_model.setVisible(expert or len(self.gpus) > 1)
        for r in (self.row_quant, self.row_offload, self.row_vae, self.row_seed):
            r.setVisible(expert)
        self.gpu_note.setVisible(expert)
        self.g_res.setVisible(expert)
        self.plan_label.setVisible(not auto)
        self.g_matte.setVisible(not auto)
        self.g_matte_adv.setVisible(expert)
        self.g_color.setVisible(not auto)
        self.despill_strength.setVisible(expert)

        for r in self.rows:
            r.show_hint(expert or simple)
        if auto:
            self.set_profile(self.profile)  # automatika = profil + hodnoty podle videa a karty
        self._plan_changed()

    def set_profile(self, key: str) -> None:
        """Rychlostní profil – nastaví rozlišení, úseky a výchozí úpravy podle videa a karty."""
        if key not in PROFILE_KEYS:
            key = DEFAULT_PROFILE
        self.profile = key
        for k, b in self.profile_buttons.items():
            b.setChecked(k == key)
        _, name, cap, ov, tf, desc = PROFILES[PROFILE_KEYS.index(key)]
        self.profile_desc.setText(desc)
        plan = self._recommended
        if plan and plan.recommended_long_side and plan.recommended_long_side < 1920:
            cap = min(cap or plan.recommended_long_side, plan.recommended_long_side)
        self._loading = True
        try:
            if self.mode != "expert":
                for cb in (self.quant, self.offload, self.vae):
                    cb.setCurrentIndex(0)  # Automaticky
            res_idx = 0 if not cap else next((i for i, (_, v) in enumerate(RESOLUTIONS) if v and v <= cap), 0)
            self.resolution.setCurrentIndex(res_idx)
            self.seg_len.setValue(min(121, self.max_segment()))
            self.overlap.setValue(ov)
            downscaled = False
            if self.info:
                downscaled = compute_geometry(self.info.width, self.info.height,
                                              self.resolution.currentData() or None).downscaled
            if self.mode == "auto":
                # Výchozí maska = výstup modelu + potlačení blikání okrajů; zpřesnění hrany jen když
                # se maska počítala zmenšená (Turbo, 4K) – jinak by zbytečně měnilo výsledek modelu.
                self._set_matte_widgets(MatteSettings(temporal=tf, edge_refine=downscaled, edge_radius=3,
                                                      edge_strength=0.5))
            else:
                self.temporal.setValue(tf)
                self.edge_refine.setChecked(downscaled)
        finally:
            self._loading = False
        self._plan_changed()
        self._matte_changed()

    def _set_matte_widgets(self, m: MatteSettings) -> None:
        self.black.setValue(m.black_point)
        self.white.setValue(m.white_point)
        self.gamma.setValue(m.gamma)
        self.erode.setValue(m.erode_dilate)
        self.feather.setValue(m.feather)
        self.edge_refine.setChecked(m.edge_refine)
        self.edge_radius.setValue(m.edge_radius)
        self.edge_strength.setValue(m.edge_strength)
        self.temporal.setValue(m.temporal)
        self.despeckle.setValue(m.despeckle)
        self.invert.setChecked(m.invert)
        self.despill.setCurrentIndex(max(0, self.despill.findData(m.despill)))
        self.despill_strength.setValue(m.despill_strength)

    # ------------------------------------------------------------------
    def _gpu_changed(self) -> None:
        plan = recommend(self.current_gpu(), self.ram_total)
        self._recommended = plan
        txt = (f"For this GPU: segments up to {plan.recommended_segment} frames, "
               f"{'lightweight' if plan.vae == 'conv' else 'accurate'} decoder.")
        if plan.notes:
            txt += "\n" + "\n".join("• " + n for n in plan.notes)
        self.gpu_note.setToolTip(txt)
        if hasattr(self, "profile_buttons"):
            self.set_profile(self.profile)

    def apply_recommendation(self) -> None:
        self.set_mode("auto")

    def _manual_changed(self) -> None:
        if not self._loading:
            for b in self.profile_buttons.values():  # ruční změna = vlastní nastavení
                b.setChecked(False)
            self.profile_desc.setText("Custom settings (Expert).")
        self._plan_changed()

    def _seg_changed(self, v: int) -> None:
        fixed = (v - 1) // 8 * 8 + 1  # vynutit 8n+1
        if fixed != v:
            self.seg_len.blockSignals(True)
            self.seg_len.setValue(fixed)
            self.seg_len.blockSignals(False)
        self._manual_changed()

    def set_source(self, info: VideoInfo) -> None:
        self.info = info
        self.set_profile(self.profile) if self.mode != "expert" else self._plan_changed()

    def set_range(self, a: int, b: int) -> None:
        self.frame_range = (a, b)
        self._plan_changed()

    def _plan_changed(self) -> None:
        if self._loading:
            return
        if not self.info:
            self.plan_label.setText("")
            self.planChanged.emit()
            return
        geo = compute_geometry(self.info.width, self.info.height, self.resolution.currentData() or None)
        a, b = self.frame_range
        total = max(1, b - a + 1)
        segs = plan_segments(total, self.seg_len.value(), self.overlap.value())
        est = self._estimate(geo, segs)
        txt = (f"Matte computed at {geo.work_w}×{geo.work_h}"
               + (f" (downscaled from {geo.src_w}×{geo.src_h})" if geo.downscaled else "")
               + f", {len(segs)} segment(s) of ≤{self.seg_len.value()} frames.")
        if self.seg_len.value() > self.max_segment():
            txt += f"\n⚠ This GPU fits at most ~{self.max_segment()} frames per segment – risk of running out of VRAM."
        if self.seg_len.value() > RECOMMENDED_MAX_FRAMES:
            txt += f"\n⚠ Above {RECOMMENDED_MAX_FRAMES} frames the model is less reliable."
        if est:
            txt += f"\nEstimated compute time: {est}"
        self.plan_label.setText(txt)
        self._update_auto_text(geo, segs, est)
        self.planChanged.emit()

    def resolved_vae(self) -> str:
        v = self.vae.currentData()
        if v != "auto":
            return v
        plan = self._recommended
        return plan.vae if plan else "diffvae"

    def _estimate(self, geo, segs) -> str:
        if not self.estimator:
            return ""
        sec = self.estimator(geo, segs, self.resolved_vae())
        if sec is None:
            return "measured on first run"
        m = sec / 60
        return f"~{m:.0f} min" if m >= 1.5 else "under 2 min"

    def _update_auto_text(self, geo, segs, est: str) -> None:
        if not self.info:
            return
        g = self.current_gpu()
        plan = self._recommended
        name = PROFILES[PROFILE_KEYS.index(self.profile)][1] if self.profile in PROFILE_KEYS else "Custom"
        lines = [
            f"<b>Speed:</b> {name}",
            f"<b>Matte:</b> {geo.work_w}×{geo.work_h}"
            + (f" (downscaled from {geo.src_w}×{geo.src_h}, then refined against the source)" if geo.downscaled
               else " – full clip resolution"),
            f"<b>Compute:</b> {len(segs)} segment(s) of ≤{self.seg_len.value()} frames"
            + (", transitions smoothly blended" if self.overlap.value() and len(segs) > 1 else ""),
            f"<b>GPU:</b> {g.name if g else 'not found'}"
            + (f" – {'model streamed from RAM' if plan.offload != 'none' else 'model fully in VRAM'}, "
               f"{'full bf16 precision' if not plan.quantization else 'FP8 precision'}, "
               f"{'lightweight' if self.resolved_vae() == 'conv' else 'accurate'} decoder" if plan else ""),
            "<b>Refinement:</b> edge anti-flicker only – otherwise the exact model output",
        ]
        from ..gpu import ram_info

        avail, total = ram_info()
        if plan and plan.offload != "none" and total:
            if avail < 32:
                lines.append(f"<span style='color:#f2c14e'><b>RAM:</b> only {avail:.0f} GB free – the model will "
                             "stream from disk (slower). Close heavy applications.</span>")
            else:
                lines.append(f"<b>RAM:</b> {avail:.0f} of {total:.0f} GB free – sufficient")
        if est:
            lines.append(f"<b>Estimated time:</b> {est}")
        lines.append("<span style='color:#a9b1bd'>Matte not right? Fine-tune it in Simple or Expert mode – "
                     "adjustments don't need a recompute.</span>")
        self.auto_label.setText("<br>".join(lines))

    def _matte_changed(self, *_) -> None:
        if not self._loading:
            self.matteChanged.emit(self.matte_settings())

    # ------------------------------------------------------------------
    def matte_settings(self) -> MatteSettings:
        return MatteSettings(
            black_point=self.black.value(), white_point=self.white.value(), gamma=self.gamma.value(),
            erode_dilate=self.erode.value(), feather=self.feather.value(),
            edge_refine=self.edge_refine.isChecked(), edge_radius=int(self.edge_radius.value()),
            edge_strength=self.edge_strength.value(), temporal=self.temporal.value(),
            despeckle=int(self.despeckle.value()), invert=self.invert.isChecked(),
            despill=self.despill.currentData(), despill_strength=self.despill_strength.value(),
        )

    def export_settings(self) -> ExportSettings:
        return ExportSettings(
            formats=[k for k, cb in self.fmt_checks.items() if cb.isChecked()],
            alpha_mode=self.alpha_mode.currentData(), include_audio=self.audio.isChecked(),
            exr_linearize=self.exr_lin.isChecked(),
        )

    def gpu_index(self) -> int:
        return max(0, int(self.gpu.currentData() or 0))

    def output_dir(self) -> str:
        return self.out_dir.text().strip() or self.app_settings.output_dir

    def state(self) -> dict:
        return {"defaults_version": DEFAULTS_VERSION, "mode": self.mode, "profile": self.profile, "resolution": self.resolution.currentIndex(),
                "seg": self.seg_len.value(), "overlap": self.overlap.value(), "seed": self.seed.value(),
                "quant": self.quant.currentIndex(), "offload": self.offload.currentIndex(),
                "vae": self.vae.currentIndex(), "matte": self.matte_settings().to_dict(),
                "export": self.export_settings().to_dict(), "out_dir": self.out_dir.text()}

    def restore(self, st: dict) -> None:
        if not st:
            return
        try:
            e = ExportSettings.from_dict(st.get("export", {}))
            fresh = st.get("defaults_version", st.get("formats_version", 1)) < DEFAULTS_VERSION
            if fresh:  # po aktualizaci výchozích hodnot: luma H.264 + náhled, Automatic + doporučená rychlost
                e.formats = ExportSettings().formats
            for k, cb in self.fmt_checks.items():
                cb.setChecked(k in e.formats)
            self.alpha_mode.setCurrentIndex(max(0, self.alpha_mode.findData(e.alpha_mode)))
            self.audio.setChecked(e.include_audio)
            self.exr_lin.setChecked(e.exr_linearize)
            if st.get("out_dir") and Path(st["out_dir"]).anchor:
                self.out_dir.setText(st["out_dir"])
            self.profile = DEFAULT_PROFILE if fresh else st.get("profile", DEFAULT_PROFILE)
            mode = "auto" if fresh else st.get("mode", "auto")
            if mode == "expert":  # ruční hodnoty obnovujeme jen v režimu Expert
                self.set_mode("expert")
                self._loading = True
                self.resolution.setCurrentIndex(st.get("resolution", 0))
                self.seg_len.setValue(st.get("seg", 49))
                self.overlap.setValue(st.get("overlap", 8))
                self.seed.setValue(st.get("seed", 1234))
                self.quant.setCurrentIndex(st.get("quant", 0))
                self.offload.setCurrentIndex(st.get("offload", 0))
                self.vae.setCurrentIndex(st.get("vae", 0))
                self._set_matte_widgets(MatteSettings.from_dict(st.get("matte", {})))
                self._loading = False
                self._plan_changed()
                self._matte_changed()
            else:
                self.set_mode(mode if mode in self.mode_buttons else "auto")
                if mode == "simple":
                    m = MatteSettings.from_dict(st.get("matte", {}))
                    self.erode.setValue(m.erode_dilate)
                    self.feather.setValue(m.feather)
                    self.temporal.setValue(m.temporal)
                    self.invert.setChecked(m.invert)
        except (TypeError, ValueError):
            self._loading = False
