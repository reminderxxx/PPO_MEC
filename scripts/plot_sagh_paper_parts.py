"""Three single-column paper panels, preserving the frozen R20 SAGH contract."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import plot_sagh_architecture as base

BLUE='#244B68'
TEAL='#237A72'
GRAY='#F4F7F8'
EDGE='#A6B6BF'
TRAIN='#596472'


def make_part(folder,stem,height,title,render):
    base.WIDTH,base.HEIGHT=900,height
    d=base.Drawing(folder,paper_width_mm=85)
    def text(x,y,s,size=30,color=BLUE,style='regular',anchor='middle',max_width=None):
        d.text(x,y,s,size,style,color,anchor,max_width)
    def box(x,y,w,h,heading,body=(),color=BLUE):
        d.rect(x,y,w,h,base.WHITE,EDGE,radius=3,lw=1.6)
        text(x+w/2,y+43,heading,32,color,'bold',max_width=w-24)
        for i,line in enumerate(body):text(x+w/2,y+89+i*39,line,30,color,max_width=w-24)
    def arrow(points,color=BLUE,dashed=False):
        d.line(points,color=color,dashed=dashed,arrow=True,lw=2.1)
    text(450,46,title,34,style='bold',max_width=860)
    render(d,text,box,arrow)
    d.finish()
    # Keep standalone panels at the output root with meaningful file names.
    for ext in ['pdf','svg']:
        (folder/f'sagh_algorithm_architecture.{ext}').rename(folder/f'{stem}.{ext}')
    meta=json.loads((folder/'layout_validation.json').read_text())
    meta.update(part=stem,paper_height_mm=height*85/900,
                typical_font_pt=30*d.scale,heading_font_pt=32*d.scale)
    meta.pop('body_font_pt_at_180mm',None)
    (folder/f'{stem}_layout.json').write_text(json.dumps(meta,indent=2))
    (folder/'layout_validation.json').unlink()


def overview(d,t,b,a):
    for x,label in [(352,'v1'),(423,'v2'),(494,'vN')]:
        d.rect(x,84,65,43,GRAY,EDGE,radius=2)
        t(x+32,115,label,29)
    b(35,155,240,183,'Observations',['Released DAG','RSU resources','Motion history'])
    b(330,155,240,183,'Shared SAGH',['Vehicle agents','Graph encoding','Shared actor'],TEAL)
    b(625,155,240,183,'Joint action',['(e, m, h, p)','Execution','Preparation'])
    a([(275,246),(330,246)])
    a([(570,246),(625,246)],TEAL)
    b(230,438,440,132,'Vehicular edge environment',['DAG service + cache + handoff'])
    a([(745,338),(745,502),(670,502)],TEAL)
    a([(230,502),(17,502),(17,246),(35,246)])
    t(125,415,'Next state',29)
    t(450,622,'Agents share policy parameters and RSU resources.',29,style='italic',max_width=850)


def policy(d,t,b,a):
    d.rect(105,94,690,89,'#E9F3EF','#8DB3A9',radius=3)
    t(450,135,'Parameter-shared actor',34,TEAL,'bold')
    t(450,171,'Applied to each active vehicle',29,TEAL)
    t(450,232,'Pass 1: all vehicle sites',30,TEAL,max_width=850)
    b(45,254,330,119,'Fast service head',['Execution site e'],TEAL)
    b(475,254,380,150,'Shared link calendar',['Reserve current service'],TEAL)
    for k in range(7):d.rect(503+k*46,372,40,17,'#9CBDAD' if k<3 else '#E8EFEC',EDGE,radius=0,lw=.7)
    a([(450,183),(450,194),(210,194),(210,254)],TEAL)
    a([(375,312),(475,312)],TEAL)
    # Calendar conditions optional controls after all site choices are known.
    a([(665,404),(665,430),(210,430),(210,498)],TEAL)
    t(568,469,'Pass 2: per-vehicle controls',30,TEAL,max_width=480)
    b(45,500,330,129,'Slow cache head',['Model admission m'],TEAL)
    b(500,500,355,129,'Event head',['Handoff h, priority p'],TEAL)
    a([(375,565),(500,565)],TEAL)
    a([(807,500),(807,448),(833,448),(833,404)],TEAL)
    t(450,682,'Update bookings before the next vehicle.',29,style='italic',max_width=850)
    t(450,725,'All heads use prefix-conditioned feasibility masks.',29,style='italic',max_width=850)


def learning(d,t,b,a):
    b(35,110,330,119,'On-policy rollout',['States, actions, rewards'],TRAIN)
    b(535,110,330,119,'Team targets',['Monte Carlo, γ = 1'],TRAIN)
    a([(365,170),(535,170)],TRAIN,True)
    b(35,319,330,119,'Central critic / Q',['Fleet + action prefix'],TRAIN)
    b(535,319,330,119,'Return surrogate',['Return + transfer cost'],TRAIN)
    a([(700,229),(700,271),(200,271),(200,319)],TRAIN,True)
    a([(700,271),(700,319)],TRAIN,True)
    b(35,530,330,119,'Conditional baseline',['Normalized advantages'],TRAIN)
    b(535,530,330,119,'Clipped PPO',['Policy + value + entropy'],TRAIN)
    a([(200,438),(200,530)],TRAIN,True)
    a([(535,378),(450,378),(450,589),(365,589)],TRAIN,True)
    a([(365,630),(535,630)],TRAIN,True)
    # Outbound learned weights go back to the actor in panel (b).
    d.rect(285,707,330,56,'#E9F3EF','#8DB3A9',radius=3)
    t(450,746,'Shared actor → part (b)',30,TEAL,'bold',max_width=310)
    a([(700,649),(700,735),(615,735)],TRAIN,True)


def draw(folder):
    folder.mkdir(parents=True,exist_ok=True)
    make_part(folder,'a_control_overview',660,'(a) Multi-agent decision loop',overview)
    make_part(folder,'b_sagh_policy',760,'(b) Conditional SAGH policy',policy)
    make_part(folder,'c_centralized_training',800,'(c) Centralized policy learning',learning)
    captions={
      'a_control_overview':'Multi-agent decision loop. Each active vehicle contributes observations from its released DAG, motion and shared RSU resources. Graph/resource encoding and causal contact features feed a shared SAGH policy, which produces execution, model-admission and handoff/priority controls. The joint action advances the environment, whose next state closes the control loop.',
      'b_sagh_policy':'Conditional SAGH policy. The shared actor first selects execution sites for all active vehicles. Reactive service transfers then define a shared link calendar. Optional model admission and handoff/priority controls are selected vehicle by vehicle under updated reservations and prefix-conditioned feasibility masks. Vehicles are agents; policy heads are reused components.',
      'c_centralized_training':'Centralized policy learning. On-policy rollouts provide undiscounted session team returns and observational transfer-cost labels. Central value functions and an observational surrogate form stored-policy conditional baselines for normalized advantages and clipped PPO. Updated shared actor parameters are used in panel (b); the surrogate does not add an online action-logit bonus.'}
    (folder/'figure_captions.json').write_text(json.dumps(captions,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_dir',type=Path,default=Path('artifacts/figures/sagh_paper_parts_20261027'))
    draw(p.parse_args().output_dir)
