"""Draw the R20 stream-SAGH architecture as editable SVG and vector PDF.

This standalone drawing follows StreamControlPPO at scientific commit
35975cb3d73c84ddbabf01f6749bbc2d60c826de and manuscript revision R20.
It is distinct from the legacy semantic_discrete_5 SA-GHMAPPO controller.
Requires reportlab; raster previews are rendered separately with Poppler.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


WIDTH, HEIGHT = 1800, 1560
INK = "#19334A"
MUTED = "#4D6273"
BLUE = "#205F92"
TEAL = "#18796F"
PURPLE = "#72578A"
LINE = "#B8C9D5"
WHITE = "#FFFFFF"


class Drawing:
    """Keep PDF and SVG geometry identical and check every text line."""

    def __init__(self, folder: Path, paper_width_mm: float = 180):
        self.folder = folder
        self.paper_width_mm = float(paper_width_mm)
        folder.mkdir(parents=True, exist_ok=True)
        fonts = Path("/System/Library/Fonts/Supplemental")
        self.fonts = {"regular": "Times-Roman", "bold": "Times-Bold", "italic": "Times-Italic"}
        for style, suffix in [("regular", ""), ("bold", " Bold"), ("italic", " Italic")]:
            file = fonts / f"Times New Roman{suffix}.ttf"
            if file.exists():
                name = f"SAGHTimes-{style}"
                pdfmetrics.registerFont(TTFont(name, str(file)))
                self.fonts[style] = name
        self.scale = (self.paper_width_mm / 25.4 * 72) / WIDTH
        self.pdf = canvas.Canvas(str(folder / "sagh_algorithm_architecture.pdf"),
                                 pagesize=(WIDTH * self.scale, HEIGHT * self.scale))
        self.pdf.setTitle("Multi-vehicle SAGH decision and learning architecture")
        self.pdf.setAuthor("PPO_MEC")
        self.pdf.scale(self.scale, self.scale)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.paper_width_mm:g}mm" '
                    f'height="{HEIGHT * self.paper_width_mm / WIDTH:g}mm" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
                    'aria-labelledby="title desc">',
                    '<title id="title">Multi-vehicle SAGH decision and learning architecture</title>',
                    '<desc id="desc">Causal observations, graph and candidate encoding, '
                    'two-pass shared-actor decisions, environment feedback, and centralized '
                    'team-return PPO training. Vehicles are agents; heads share parameters.</desc>']
        self.text_checks = []
        self.rect(0, 0, WIDTH, HEIGHT, WHITE, WHITE, radius=0)

    def rect(self, x, y, w, h, fill=WHITE, stroke=LINE, radius=8, dashed=False, lw=1.7):
        p = self.pdf
        p.setFillColor(HexColor(fill)); p.setStrokeColor(HexColor(stroke))
        p.setLineWidth(lw); p.setDash([8, 6] if dashed else [])
        p.roundRect(x, HEIGHT-y-h, w, h, radius, stroke=1, fill=1)
        dash = ' stroke-dasharray="8 6"' if dashed else ''
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
                        f'fill="{fill}" stroke="{stroke}" stroke-width="{lw}"{dash}/>')

    def line(self, points, color=INK, dashed=False, arrow=False, lw=2):
        p = self.pdf
        p.setStrokeColor(HexColor(color)); p.setFillColor(HexColor(color))
        p.setLineWidth(lw); p.setDash([8, 6] if dashed else [])
        path = p.beginPath(); path.moveTo(points[0][0], HEIGHT-points[0][1])
        for x, y in points[1:]: path.lineTo(x, HEIGHT-y)
        p.drawPath(path)
        pts = ' '.join(f'{x},{y}' for x, y in points)
        dash = ' stroke-dasharray="8 6"' if dashed else ''
        self.svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                        f'stroke-width="{lw}" stroke-linejoin="round"{dash}/>')
        if arrow:
            x, y = points[-1]; px, py = points[-2]
            a = math.atan2(y-py, x-px); length, half = 10, 4.5
            triangle = [(x, y), (x-length*math.cos(a)+half*math.sin(a), y-length*math.sin(a)-half*math.cos(a)),
                        (x-length*math.cos(a)-half*math.sin(a), y-length*math.sin(a)+half*math.cos(a))]
            self.polygon(triangle, color)

    def polygon(self, points, fill):
        p = self.pdf; p.setFillColor(HexColor(fill))
        path = p.beginPath(); path.moveTo(points[0][0], HEIGHT-points[0][1])
        for x, y in points[1:]: path.lineTo(x, HEIGHT-y)
        path.close(); p.drawPath(path, fill=1, stroke=0)
        self.svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="{fill}"/>')

    def circle(self, x, y, radius, fill=WHITE, stroke=BLUE):
        self.pdf.setFillColor(HexColor(fill)); self.pdf.setStrokeColor(HexColor(stroke))
        self.pdf.setLineWidth(1.7); self.pdf.setDash([])
        self.pdf.circle(x, HEIGHT-y, radius, fill=1, stroke=1)
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" '
                        f'stroke="{stroke}" stroke-width="1.7"/>')

    def text(self, x, y, text, size=26, style="regular", color=INK, anchor="start", max_width=None):
        font = self.fonts[style]; width = pdfmetrics.stringWidth(text, font, size)
        left = x if anchor == "start" else x-width/2 if anchor == "middle" else x-width
        if max_width is not None and width > max_width + 0.1:
            raise ValueError(f"Text exceeds its cell: {text!r}, {width:.1f} > {max_width}")
        if left < 0 or left+width > WIDTH or y-size < 0 or y > HEIGHT:
            raise ValueError(f"Text exceeds the canvas: {text!r}")
        self.text_checks.append({"text": text, "x": left, "baseline_y": y,
                                 "width": width, "font_size": size})
        p = self.pdf; p.setFillColor(HexColor(color)); p.setFont(font, size)
        p.drawString(left, HEIGHT-y, text)
        weight = ' font-weight="bold"' if style == "bold" else ''
        italic = ' font-style="italic"' if style == "italic" else ''
        self.svg.append(f'<text x="{x}" y="{y}" font-family="Times New Roman, Times, serif" '
                        f'font-size="{size}" text-anchor="{anchor}" fill="{color}"{weight}{italic}>'
                        f'{html.escape(text)}</text>')

    def lines(self, x, y, lines, size=26, leading=32, **kwargs):
        for i, line in enumerate(lines): self.text(x, y+i*leading, line, size=size, **kwargs)

    def panel(self, x, y, w, h, letter, title, color=BLUE, fill="#F5F9FC", dashed=False):
        self.rect(x, y, w, h, fill, LINE, dashed=dashed)
        self.rect(x+18, y+18, 35, 35, color, color, radius=3)
        self.text(x+35.5, y+44, letter, 26, "bold", WHITE, "middle")
        self.text(x+67, y+44, title, 30, "bold", color, max_width=w-86)

    def card(self, x, y, w, h, title, body, color=BLUE, fill=WHITE):
        self.rect(x, y, w, h, fill, LINE)
        self.line([(x+1,y+1),(x+w-1,y+1)], color=color, lw=3.2)
        self.text(x+15, y+36, title, 28, "bold", color, max_width=w-30)
        self.lines(x+15, y+74, body, max_width=w-30)

    def finish(self):
        self.pdf.showPage(); self.pdf.save()
        self.svg.append('</svg>')
        (self.folder / "sagh_algorithm_architecture.svg").write_text('\n'.join(self.svg))
        (self.folder / "layout_validation.json").write_text(json.dumps({
            "canvas_units": [WIDTH, HEIGHT], "paper_width_mm": self.paper_width_mm,
            "body_font_pt_at_180mm": 26*self.scale,
            "text_lines_checked": len(self.text_checks), "overflow_count": 0,
            "text": self.text_checks}, indent=2))


def draw(folder: Path):
    d = Drawing(folder)
    # Left: four online layers. Right: the separate learning branch.
    d.text(80, 42, "SAGH: MULTI-AGENT REINFORCEMENT LEARNING", 29, "bold", BLUE)
    d.text(1330, 42, "CENTRALIZED LEARNING", 29, "bold", PURPLE)
    d.panel(80, 65, 1200, 280, "a", "Causal observations and persistent resource state")
    d.card(103, 139, 365, 183, "Released DAG requests", [
        "Active frontier + FIFO queue", "Base / adapter dependencies", "Only released nodes are visible"])
    d.card(488, 139, 365, 183, "Vehicle and RSU state", [
        "Motion + reachable RSUs", "Resident objects and free bytes", "Shared link reservations"])
    d.card(873, 139, 384, 183, "Causal mobility context", [
        "Observed motion history", "GRU contact estimates", "Prediction confidence"])

    # Observation bus splits cleanly into the three representation branches.
    d.line([(680,345),(680,373)], BLUE, arrow=True)
    d.panel(80, 375, 1200, 294, "b", "Graph-aware and candidate-conditioned representation")
    d.card(103, 444, 365, 140, "Released-DAG encoding", [])
    positions = [(131,521),(171,499),(171,547),(213,521)]
    for a,b in [(0,1),(0,2),(1,3),(2,3)]:
        x,y=positions[a]; xx,yy=positions[b]
        d.line([(x+7,y),(xx-9,yy)],BLUE,arrow=True,lw=1.5)
    for i,(x,y) in enumerate(positions): d.circle(x,y,9,"#DCE9F3" if i!=3 else WHITE)
    d.lines(236,518,["Dependency graph", "Gated residual"],size=25,leading=32,max_width=216)
    d.card(488, 444, 365, 140, "Shared-resource encoding", [
        "Cache bytes + link calendar", "Own-vehicle / fleet context"])
    d.card(873, 444, 384, 140, "Candidate construction", [
        "Site, object and event features", "Contact context + valid masks"])
    for x in (285,670,1065): d.line([(x,584),(x,604)], BLUE, arrow=True)
    d.rect(103, 606, 1154, 43, "#E8F0F7", "#E8F0F7", radius=4)
    d.text(680, 636, "Shared policy features + candidate scores conditioned on the current decision prefix", 26,
           "bold", BLUE, "middle", max_width=1120)

    d.line([(680,669),(680,699)], BLUE, arrow=True)
    d.panel(80, 701, 1200, 514, "c", "SAGH agents and hierarchical conditional control", color=TEAL, fill="#F4FAF8")
    for x,label,obs in [(103,"Vehicle agent 1","Own observation o(1,t)"),
                         (513,"Vehicle agent 2","Own observation o(2,t)"),
                         (923,"Vehicle agent N","Own observation o(N,t)")]:
        d.rect(x,770,334,88,WHITE,"#9DC2B7",radius=6)
        d.text(x+167,803,label,28,"bold",TEAL,"middle",max_width=306)
        d.text(x+167,836,obs,26,color=INK,anchor="middle",max_width=306)
        d.line([(x+167,858),(x+167,880)],TEAL,arrow=True)
    d.text(475,821,"...",28,"bold",TEAL,"middle")
    d.text(885,821,"...",28,"bold",TEAL,"middle")
    d.rect(103,882,1154,46,"#E4F1EC","#C1D8CE",radius=4)
    d.text(680,914,"One shared actor πθ, reused across all active agents; vehicle order rotates each frame",27,
           "bold",TEAL,"middle",max_width=1120)
    xs = [103, 397, 691, 985]
    d.line([(680,928),(680,944),(239,944),(239,970)],TEAL,arrow=True)
    d.card(xs[0], 972, 272, 174, "1  Execution site", [
        "Fast service head", "Choose all agents' sites", "Reachable RSU / local"], TEAL)
    d.card(xs[1], 972, 272, 174, "2  Shared calendar", [
        "Book reactive transfers", "Protect current service", "Residual slots exposed"], TEAL)
    d.card(xs[2], 972, 272, 174, "3  Model admission", [
        "Slow cache head", "None / base / adapter", "Released-node targets"], TEAL)
    d.card(xs[3], 972, 272, 174, "4  Event + priority", [
        "Event handoff head", "Feasible preparation", "Contention priority"], TEAL)
    for x in (375,669,963): d.line([(x,1057),(x+22,1057)], TEAL, arrow=True)
    d.text(239, 1180, "PASS 1: all active agents", 25, "bold", TEAL, "middle", max_width=285)
    d.text(926, 1180, "PASS 2: agent-wise calendar updates", 25, "bold", TEAL, "middle", max_width=570)

    # The masks condition decisions; they are not an unconstrained post-hoc repair.
    d.line([(680,1215),(680,1252)], TEAL, arrow=True)
    d.panel(80, 1254, 1200, 222, "d", "Joint execution and state transition", color=TEAL, fill="#F4FAF8")
    d.rect(103, 1322, 1154, 47, "#E4F1EC", "#C1D8CE", radius=4)
    d.text(680, 1354, "Masks throughout: reachability | base closure | RSU capacity | protected objects | link spans", 26,
           "bold", TEAL, "middle", max_width=1120)
    d.text(680, 1405, "Joint frame action  { a(v,t) = (e, m, h, p) : v is active }", 29,
           "bold", INK, "middle", max_width=1130)
    d.text(680, 1451, "Advance DAG service, transfers and handoffs; return fleet reward, next state and lifecycle events", 26,
           color=MUTED, anchor="middle", max_width=1130)
    # Causal feedback loop sits outside all panels.
    d.line([(80,1385),(35,1385),(35,205),(80,205)], TEAL, arrow=True, lw=2.4)
    d.text(680, 1517, "Cache residency persists across requests within a session; current execution follows serial DAG order.",
           25, "italic", MUTED, "middle", max_width=1200)

    # Learning: on-policy records -> Monte Carlo team return -> value estimators
    # -> prefix baseline -> PPO. There is no direct surrogate-to-online-logit edge.
    d.panel(1330,65,420,1411,"e","Team-return PPO",PURPLE,"#FAF8FC",dashed=True)
    d.card(1352,139,376,154,"On-policy trajectories",[
        "States, masks, action prefixes", "Stored log probabilities", "Fleet rewards + lifecycle events"],PURPLE)
    d.card(1352,342,376,142,"Session return targets",[
        "Undiscounted team return", "Target scaling + fleet accounting"],PURPLE)
    d.line([(1540,293),(1540,342)], PURPLE, dashed=True, arrow=True)
    d.card(1352,533,376,141,"Central value functions",[
        "Own-vehicle + aggregate context", "Critic + prefix action values"],PURPLE)
    d.line([(1540,484),(1540,533)], PURPLE, dashed=True, arrow=True)
    d.card(1352,715,376,141,"Observational surrogate",[
        "Return and transfer-cost labels", "Auxiliary training estimator"],PURPLE)
    # Return labels also feed the surrogate; route in a narrow margin.
    d.line([(1728,413),(1739,413),(1739,782),(1728,782)], PURPLE, dashed=True, arrow=True)
    d.card(1352,907,376,142,"Conditional baselines",[
        "Policy-averaged action values", "Normalize head advantages"],PURPLE)
    d.line([(1460,674),(1341,674),(1341,975),(1352,975)], PURPLE, dashed=True, arrow=True)
    d.line([(1540,856),(1540,907)], PURPLE, dashed=True, arrow=True)
    d.card(1352,1103,376,160,"Clipped policy objective",[
        "E[min(r A, clip(r) A)]", "r = πθ / πold; clip: 1 ± ε", "Value loss + entropy term"],PURPLE)
    d.card(1352,1293,376,154,"Shared parameter updates",[
        "Shared actor + value regression", "Entropy + gradient clipping", "Update shared policy parameters"],PURPLE)
    d.line([(1540,1049),(1540,1103)], PURPLE, dashed=True, arrow=True)
    d.line([(1540,1263),(1540,1293)], PURPLE, dashed=True, arrow=True)
    # Environment-to-rollout and policy-update loop, each in its own gutter.
    d.line([(1280,1410),(1305,1410),(1305,214),(1352,214)], PURPLE, dashed=True, arrow=True)
    d.line([(1352,1370),(1317,1370),(1317,905),(1257,905)], PURPLE, dashed=True, arrow=True)
    d.text(1540,1517,"Surrogate use: training baseline only",25,"italic",PURPLE,"middle",max_width=420)
    # A restrained line-style legend avoids a large decorative header.
    d.line([(80,1547),(130,1547)], TEAL, arrow=True)
    d.text(143,1554,"Online control / state flow",23,color=MUTED)
    d.line([(460,1547),(510,1547)], PURPLE, dashed=True, arrow=True)
    d.text(523,1554,"Training / parameter flow",23,color=MUTED)
    d.text(1750,1554,"e: site   m: model admission   h: preparation   p: priority",23,
           color=MUTED,anchor="end")
    d.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path,
                        default=Path("artifacts/figures/sagh_architecture_20261027"))
    draw(parser.parse_args().output_dir)
