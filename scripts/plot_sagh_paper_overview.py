"""Compact, vector-only paper overview of the frozen R20 multi-vehicle SAGH.

Uses the drawing primitives from plot_sagh_architecture, preserving the earlier
explanatory figure. Outputs a new standalone asset, not an edited manuscript.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import plot_sagh_architecture as base

BLUE, TEAL = '#244B68', '#237A72'
GRAY, EDGE = '#F5F7F9', '#A4B3BE'
TRAIN = '#5A6472'


def draw(folder: Path):
    base.WIDTH, base.HEIGHT = 1800, 1140
    d = base.Drawing(folder)

    def text(x,y,s,size=28,**kw):
        d.text(x,y,s,size=size,**kw)

    def mathtext(cx,y,markup,size=28,color=BLUE):
        pieces=[]
        for chunk in re.split(r'(_\{[^}]+\})',markup):
            if not chunk:continue
            sub=chunk.startswith('_{')
            value=chunk[2:-1] if sub else chunk
            fontsize=size*.72 if sub else size
            width=base.pdfmetrics.stringWidth(value,d.fonts['italic'],fontsize)
            pieces.append((value,fontsize,width,sub))
        x=cx-sum(p[2] for p in pieces)/2
        for value,fontsize,width,sub in pieces:
            text(x,y+size*.22 if sub else y,value,fontsize,style='italic',color=color)
            x+=width

    def box(x,y,w,h,title,color=BLUE):
        d.rect(x,y,w,h,base.WHITE,EDGE,radius=3,lw=1.5)
        text(x+15,y+36,title,29,style='bold',color=color,max_width=w-30)

    def panel(x,w,title):
        d.rect(x,40,w,610,base.WHITE,EDGE,radius=3,lw=1.6)
        text(x+15,80,title,29,style='bold',color=BLUE,max_width=w-30)
        d.line([(x+14,96),(x+w-14,96)],color=EDGE,lw=1.2)

    def net(cx,cy,color=TEAL):
        layers=[[(cx,cy-15),(cx,cy+15)],[(cx+33,cy-25),(cx+33,cy),(cx+33,cy+25)],[(cx+66,cy-15),(cx+66,cy+15)]]
        for left,right in zip(layers,layers[1:]):
            for x,y in left:
                for xx,yy in right:d.line([(x,y),(xx,yy)],color='#B7C8CB',lw=1)
        for layer in layers:
            for x,y in layer:d.circle(x,y,5.5,base.WHITE,color)

    def arrow(pts,color=BLUE,dash=False):
        d.line(pts,color=color,dashed=dash,arrow=True,lw=2)

    panel(20,300,'(a) Causal state')
    panel(340,340,'(b) Representation')
    panel(700,750,'(c) Shared SAGH policy')
    panel(1470,310,'(d) Environment')

    # The source contains only active/released graphs, shared resources and history.
    text(170,137,'Released DAGs',style='bold',anchor='middle')
    xy=[(72,207),(139,174),(139,240),(213,207),(267,207)]
    for a,b in [(0,1),(0,2),(1,3),(2,3),(3,4)]:
        x,y=xy[a];xx,yy=xy[b];arrow([(x+10,y),(xx-12,yy)])
    for i,(x,y) in enumerate(xy):d.circle(x,y,12,'#DAE8ED' if i<3 else base.WHITE,BLUE)
    text(170,282,'Frontier + released queue',26,anchor='middle',max_width=280)
    text(170,337,'RSU resources',style='bold',anchor='middle')
    for i,x in enumerate([51,139,227]):
        d.rect(x,357,63,68,GRAY,EDGE,radius=2)
        text(x+31,383,['R1','R2','Rj'][i],24,anchor='middle')
        for j in range(3):d.rect(x+7+j*16,395,12,18,'#9DC4BE' if (i+j)%3 else '#D9E0E5',EDGE,radius=0,lw=.7)
    text(170,461,'Cache bytes + reservations',25,anchor='middle',max_width=280)
    text(170,518,'Motion history',style='bold',anchor='middle')
    d.line([(54,579),(101,562),(149,567),(201,541),(269,547)],BLUE,lw=2)
    for x,y in [(54,579),(101,562),(149,567),(201,541),(269,547)]:d.circle(x,y,4,BLUE,BLUE)
    text(170,621,'Per-agent observation o(v,t)',25,anchor='middle',max_width=280)

    # Three compact encoder branches feed the same actor representation.
    box(359,120,302,132,'DAG encoder')
    net(386,202,BLUE)
    text(552,190,'Graph pooling',26,anchor='middle',max_width=170)
    text(552,225,'Gated residual',26,anchor='middle',max_width=170)
    box(359,280,302,119,'Resource / candidates')
    text(510,355,'Site, model, event features',25,anchor='middle',max_width=275)
    box(359,427,302,121,'GRU contact features')
    for x,lab in [(380,'t-2'),(473,'t-1'),(566,'t')]:
        d.rect(x,483,70,41,GRAY,EDGE,radius=2)
        text(x+35,513,lab,25,anchor='middle')
    arrow([(450,504),(473,504)])
    arrow([(543,504),(566,504)])
    d.rect(359,588,302,43,'#E6EEF2',EDGE,radius=2)
    text(510,618,'Latent z(v,t) + masks',27,style='bold',anchor='middle',max_width=280)
    for y in [206,357,530]:arrow([(320,y),(359,y)])
    # Branch wiring uses a gutter inside the encoder panel.
    for y in [252,399,548]:
        arrow([(510,y),(350,y),(350,566),(510,566),(510,588)])
    arrow([(661,609),(690,609),(690,239),(724,239)])

    # Vehicles are the agents. Repeated heads share parameters and a global calendar.
    for x,label in [(724,'Agent 1'),(960,'Agent 2'),(1196,'Agent N')]:
        d.rect(x,120,230,59,GRAY,EDGE,radius=3)
        text(x+115,158,label,29,style='bold',anchor='middle')
        arrow([(x+115,179),(x+115,212)],TEAL)
        d.rect(724,214,702,50,'#EAF3F0','#82AFA7',radius=3)
    text(1040,248,'Parameter-shared actor',30,style='bold',color=TEAL,anchor='middle')
    mathtext(1215,248,'π_{θ}',30,TEAL)
    arrow([(1075,264),(1075,278),(736,278),(736,320)],TEAL)
    text(850,309,'Pass 1: all vehicle sites',25,anchor='middle',color=TEAL)
    box(724,322,250,121,'Fast head πe',TEAL)
    net(744,400)
    text(889,402,'Site e',28,anchor='middle',color=TEAL,max_width=140)
    box(1001,322,425,121,'Shared link calendar C',TEAL)
    text(1213,393,'Reserve current service first',26,anchor='middle',max_width=395)
    for k in range(8):d.rect(1036+k*43,408,38,17,'#91B8B0' if k in [0,1,3] else '#E8EEEE',EDGE,radius=0,lw=.7)
    arrow([(974,382),(1001,382)],TEAL)
    arrow([(1213,443),(1213,450),(865,450),(865,492)],TEAL)
    box(724,494,300,121,'Slow head πm',TEAL)
    net(744,573)
    text(913,576,'Model m',27,anchor='middle',color=TEAL,max_width=170)
    box(1101,494,325,121,'Event head πh',TEAL)
    net(1121,573)
    text(1301,575,'Handoff h, p',26,anchor='middle',color=TEAL,max_width=215)
    arrow([(1024,552),(1101,552)],TEAL)
    text(1185,483,'Pass 2: per-agent controls',25,anchor='middle',color=TEAL,max_width=460)
    mathtext(1075,639,'π_{θ}(a|o,C) = π_{e}(e|o) π_{m}(m|o,e,C) π_{h}(h,p|o,e,m,C)',26)

    arrow([(1395,494),(1395,474),(1410,474),(1410,443)],TEAL)

    # Environment: joint action, constrained resources, persistent state transition.
    text(1625,146,'Joint action',29,style='bold',anchor='middle')
    d.rect(1490,162,270,152,GRAY,EDGE,radius=3)
    for j,label in enumerate(['a(1,t)','a(2,t)','...','a(N,t)']):
        text(1625,191+j*36,label,29,style='italic',anchor='middle')
    text(1625,335,'a(v,t) = (e,m,h,p)',28,anchor='middle',max_width=280)
    arrow([(1426,552),(1458,552),(1458,226),(1490,226)],TEAL)
    box(1490,374,270,154,'Feasible execution',TEAL)
    text(1625,448,'Cache + link constraints',25,anchor='middle',max_width=250)
    text(1625,485,'DAG service + handoff',25,anchor='middle',max_width=250)
    arrow([(1625,341),(1625,374)],TEAL)
    arrow([(1625,528),(1625,559)],TEAL)
    mathtext(1625,592,'s(t+1), r_{team}(t)',29)
    text(1625,632,'Persistent cache state',25,anchor='middle',max_width=280)

    # Online state-feedback path is separate from dashed training paths.
    arrow([(1625,650),(1625,713),(170,713),(170,650)],BLUE)
    text(1030,700,'Next-state feedback',27,anchor='middle',color=BLUE)

    # Compact training strip: PPO is on-policy; surrogate is baseline assistance.
    d.rect(20,766,1760,322,base.WHITE,EDGE,radius=3,dashed=True,lw=1.6)
    text(40,804,'(e) Centralized training with team returns',29,style='bold',color=TRAIN)
    box(45,838,260,110,'Rollout batch',TRAIN)
    mathtext(175,905,'o, a, log π_{old}',25,TRAIN)
    mathtext(175,935,'r_{team}, transfer cost',24,TRAIN)
    box(335,838,260,110,'Team return G(t)',TRAIN)
    text(465,914,'Monte Carlo, γ = 1',26,anchor='middle',max_width=235)
    box(625,838,300,110,'Central critic / Qφ',TRAIN)
    text(775,914,'Fleet + decision prefix',25,anchor='middle',max_width=275)
    box(970,838,340,110,'Conditional baseline',TRAIN)
    mathtext(1140,914,'b = E_{πold}[Q_{φ} + f_{ψ}]/2',25,TRAIN)
    box(1340,838,400,200,'Clipped PPO',TRAIN)
    text(1540,915,'min(ρ A, clip(ρ,1±ε) A)',28,style='italic',anchor='middle',max_width=370)
    text(1540,960,'Value loss + entropy',27,anchor='middle')
    text(1540,1005,'Shared actor update',28,style='bold',color=TEAL,anchor='middle')
    for left,right in [(305,335),(595,625),(925,970),(1310,1340)]:arrow([(left,888),(right,888)],TRAIN,True)
    box(625,982,300,80,'Surrogate fψ',TRAIN)
    text(775,1050,'Training baseline only',24,style='italic',anchor='middle',max_width=270)
    arrow([(465,948),(465,1020),(625,1020)],TRAIN,True)
    arrow([(925,1020),(947,1020),(947,930),(970,930)],TRAIN,True)
    arrow([(170,713),(170,838)],TRAIN,True)
    # The learned shared parameters return to the actor.
    arrow([(1740,1000),(1792,1000),(1792,17),(1075,17),(1075,40)],TRAIN,True)
    
    text(40,1126,'Solid: online decision / state flow',25,color=BLUE)
    text(535,1126,'Dashed: training / parameter flow',25,color=TRAIN)
    text(1760,1126,'Vehicles are agents; actor parameters are shared.',25,style='italic',anchor='end',color=TRAIN)
    d.finish()
    p=folder/'layout_validation.json'
    meta=json.loads(p.read_text());meta['paper_height_mm']=114
    meta['typical_label_font_pt_at_180mm']=28*d.scale
    meta['minimum_label_font_pt_at_180mm']=min(t['font_size'] for t in meta['text'] if t['text'])*d.scale
    p.write_text(json.dumps(meta,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir',type=Path,default=Path('artifacts/figures/sagh_paper_overview_20261027'))
    draw(parser.parse_args().output_dir)
