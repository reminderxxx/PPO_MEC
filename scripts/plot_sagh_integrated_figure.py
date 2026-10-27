"""Integrated paper figure: moderate detail, shared actor and compact learning loop.

Follows the R20 StreamControlPPO scientific contract at
35975cb3d73c84ddbabf01f6749bbc2d60c826de. Earlier figure candidates are preserved.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import plot_sagh_architecture as base

BLUE='#244A65'
TEAL='#23796F'
TRAIN='#5D6672'
EDGE='#AABAC3'
PALE='#EDF4F2'


def draw(folder: Path):
    base.WIDTH,base.HEIGHT=1800,930
    d=base.Drawing(folder,paper_width_mm=180)
    def t(x,y,s,size=30,color=BLUE,style='regular',anchor='middle',max_width=None):
        d.text(x,y,s,size,style,color,anchor,max_width)
    def arrow(points,color=BLUE,dashed=False):
        d.line(points,color=color,dashed=dashed,arrow=True,lw=2)
    def box(x,y,w,h,heading,color=BLUE,fill=base.WHITE,size=31):
        d.rect(x,y,w,h,fill,EDGE,radius=4,lw=1.5)
        t(x+w/2,y+40,heading,size,color,'bold',max_width=w-24)
    def mathtext(cx,y,markup,size=28,color=TEAL):
        pieces=[]
        for chunk in re.split(r'(_\{[^}]+\})',markup):
            if not chunk:continue
            sub=chunk.startswith('_{');s=chunk[2:-1] if sub else chunk
            fs=size*.72 if sub else size
            w=base.pdfmetrics.stringWidth(s,d.fonts['italic'],fs)
            pieces.append((s,fs,w,sub))
        x=cx-sum(p[2] for p in pieces)/2
        for s,fs,w,sub in pieces:
            t(x,y+size*.22 if sub else y,s,fs,color,'italic','start')
            x+=w

    for x,w,label in [(30,300,'Observations'),(360,300,'State encoding'),
                       (690,760,'Parameter-shared SAGH'),(1480,290,'Joint execution')]:
        t(x+w/2,43,label,34,style='bold',max_width=w)
        d.rect(x,65,w,500,base.WHITE,EDGE,radius=4,lw=1.5)

    # Observations: graph structure, shared resource state and causal motion.
    box(45,90,270,120,'Released DAG')
    nodes=[(78,169),(132,149),(132,190),(192,169),(269,169)]
    for a,b in [(0,1),(0,2),(1,3),(2,3),(3,4)]:
        x,y=nodes[a];xx,yy=nodes[b];arrow([(x+8,y),(xx-10,yy)])
    for k,(x,y) in enumerate(nodes):d.circle(x,y,9,'#D9E6EC' if k<3 else base.WHITE,BLUE)
    box(45,250,270,120,'RSU resources')
    for i,x in enumerate([70,150,230]):
        d.rect(x,313,45,38,'#F1F5F6',EDGE,radius=2)
        for j in range(2):d.rect(x+5+j*18,320,13,22,'#93BAB0' if i!=j else '#E1E8EB',EDGE,radius=0,lw=.7)
    box(45,410,270,120,'Motion history')
    pts=[(72,502),(117,483),(158,490),(207,472),(277,480)]
    d.line(pts,BLUE,lw=2)
    for x,y in pts:d.circle(x,y,3.5,BLUE,BLUE)

    # Encoders are schematic: network glyphs do not specify implementation widths.
    for y,heading,body in [(90,'DAG encoder','Pooling + residual'),
                            (250,'Resource encoder','Cache + link state'),
                            (410,'Contact features','GRU + confidence')]:
        box(375,y,270,120,heading)
        t(510,y+88,body,27,max_width=246)
        arrow([(315,y+60),(375,y+60)])
        d.line([(645,y+60),(670,y+60)],BLUE,lw=2)
    d.line([(670,150),(670,470)],BLUE,lw=2)
    arrow([(670,190),(715,190)])

    for x,label in [(735,'Agent 1'),(980,'Agent 2'),(1225,'Agent N')]:
        d.rect(x,94,170,45,'#F4F7F8',EDGE,radius=3)
        t(x+85,126,label,29,style='bold',max_width=150)
        arrow([(x+85,139),(x+85,165)],TEAL)
    d.rect(715,165,705,55,PALE,'#88B1A7',radius=4)
    # Small schematic neural glyph, kept subordinate to the decision structure.
    layers=[[(740,183),(740,203)],[(764,175),(764,193),(764,211)],[(788,183),(788,203)]]
    for left,right in zip(layers,layers[1:]):
        for x,y in left:
            for xx,yy in right:d.line([(x,y),(xx,yy)],'#A9C2BD',lw=.9)
    for layer in layers:
        for x,y in layer:d.circle(x,y,4.3,base.WHITE,TEAL)
    t(1072,203,'Shared actor',33,TEAL,'bold')
    mathtext(1220,203,'π_{θ}',33)
    arrow([(1075,220),(1075,231),(735,231),(735,270)],TEAL)
    t(1230,254,'1  All vehicle sites',28,TEAL,'bold',max_width=350)

    box(715,270,235,125,'Fast head',TEAL,size=31)
    t(832,346,'Site e',29,TEAL)
    mathtext(832,380,'π_{e}(e|z)',27)
    box(985,270,435,125,'Shared link calendar C',TEAL)
    t(1202,346,'Current dependencies first',29,TEAL,max_width=405)
    for k in range(9):d.rect(1012+k*42,365,36,16,'#91B7AC' if k in [0,1,3] else '#E7EEEB',EDGE,radius=0,lw=.7)
    arrow([(950,331),(985,331)],TEAL)
    arrow([(1202,395),(1202,405),(865,405),(865,445)],TEAL)
    t(1190,439,'2  Per-vehicle preparation',27,TEAL,'bold',max_width=400)
    box(715,445,300,100,'Slow cache head',TEAL)
    mathtext(865,522,'m ~ π_{m}(·|z,e,C)',28)
    box(1115,445,305,100,'Event head',TEAL)
    mathtext(1267,522,'(h,p) ~ π_{h}(·|z,e,m,C)',27)
    arrow([(1015,495),(1115,495)],TEAL)
    arrow([(1400,445),(1400,424),(1415,424),(1415,395)],TEAL)

    box(1495,110,260,160,'Joint action')
    t(1625,196,'a(1), …, a(N)',30,style='italic',max_width=235)
    t(1625,241,'(e, m, h, p)',30,style='italic',max_width=235)
    arrow([(1420,495),(1464,495),(1464,190),(1495,190)],TEAL)
    arrow([(1625,270),(1625,340)],TEAL)
    box(1495,340,260,190,'VEC transition')
    t(1625,430,'DAG service',29,max_width=235)
    t(1625,481,'Cache / handoff state',27,max_width=235)
    arrow([(1625,565),(1625,605),(180,605),(180,565)])
    mathtext(940,592,'Next state s(t+1), team reward r_{t}',28,BLUE)

    # Training is a secondary, compact loop, rather than a second full diagram.
    d.rect(30,650,1740,225,base.WHITE,EDGE,radius=4,dashed=True,lw=1.5)
    t(55,685,'Centralized training',31,TRAIN,'bold','start')
    specs=[(55,345,'Rollout + team targets','Monte Carlo, γ = 1'),
           (445,390,'Value + surrogate','Return / cost estimation'),
           (880,345,'Conditional baseline','Prefix-wise advantages'),
           (1270,475,'Clipped PPO','min(ρA, clip(ρ)A)')]
    for x,w,heading,body in specs:
        box(x,715,w,110,heading,TRAIN,size=30)
        t(x+w/2,795,body,29,TRAIN,max_width=w-24)
    for x,xx in [(400,445),(835,880),(1225,1270)]:arrow([(x,770),(xx,770)],TRAIN,True)
    arrow([(410,605),(410,700),(228,700),(228,715)],TRAIN,True)
    t(870,858,'Surrogate: training baseline only',27,TRAIN,'italic',max_width=700)
    arrow([(1745,770),(1790,770),(1790,2),(1420,2),(1420,165)],TRAIN,True)
    t(40,914,'Solid: online control / state flow',26,BLUE,anchor='start')
    t(670,914,'Dashed: training / parameter flow',26,TRAIN)
    t(1760,914,'e: site   m: model   h: handoff   p: priority',26,BLUE,anchor='end')
    d.finish()
    meta=json.loads((folder/'layout_validation.json').read_text())
    meta.update(paper_height_mm=93,typical_font_pt=30*d.scale,
                small_label_font_pt=27*d.scale)
    (folder/'layout_validation.json').write_text(json.dumps(meta,indent=2))
    (folder/'figure_caption.txt').write_text('Multi-agent SAGH architecture. Released DAGs, shared RSU resources and causal contact features are encoded for a parameter-shared actor applied to each active vehicle. The fast head selects all execution sites before the common reactive-transfer calendar is rebuilt. Slow model-admission and event/priority heads then condition on the remaining reservations and earlier choices, updating bookings before the next vehicle. The joint action advances DAG service, cache residency and handoffs. On-policy team targets, centralized value functions and an observational surrogate support conditional-baseline PPO training; the surrogate contributes no direct online logit bonus. All online choices use prefix-conditioned feasibility masks. Solid and dashed arrows denote online and training flows, respectively. Neural, DAG and reservation glyphs are schematic.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_dir',type=Path,default=Path('artifacts/figures/sagh_integrated_paper_20261027'))
    draw(p.parse_args().output_dir)
