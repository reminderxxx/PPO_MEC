"""Static ReportLab scientific charts of existing Stage5B JSON; no model/env."""
from pathlib import Path
import argparse,hashlib,json,math
from reportlab.graphics.shapes import Drawing,String,Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor,Color
p=argparse.ArgumentParser();p.add_argument('--compact',type=Path,required=True);p.add_argument('--execution',type=Path,required=True);p.add_argument('--output_root',type=Path,required=True);a=p.parse_args();a.output_root.mkdir(exist_ok=False,parents=True)
c=json.loads(a.compact.read_text());e=json.loads(a.execution.read_text());DIM={'reference_arm','seed','reference_episode','case_id'};points=[]
colors=['#cbd5e1','#2563eb','#f59e0b','#172554'];blue=HexColor('#2563eb');amber=HexColor('#f59e0b')
def find(arr,scope,extra=None):
 hits=[r for r in arr if set(r)&DIM==set(scope) and all(r.get(k)==v for k,v in scope.items()) and all(r.get(k)==v for k,v in (extra or {}).items())]
 assert len(hits)==1,(scope,extra,len(hits));return hits[0]
def canvas(w,h,title,sub):
 d=Drawing(w,h);d.add(Rect(0,0,w,h,fillColor=HexColor('#ffffff'),strokeColor=None));label(d,28,h-32,title,18,True);label(d,28,h-53,sub,10);return d
def label(d,x,y,s,size=10,bold=False,color='#334155',anchor='start'):
 d.add(String(x,y,s,fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,fillColor=HexColor(color),textAnchor=anchor))
def chart(d,x,y,w,h,data,cats,ymax=None,step=None,stack=False,palette=None):
 bc=VerticalBarChart();bc.x=x;bc.y=y;bc.width=w;bc.height=h;bc.data=data;bc.categoryAxis.categoryNames=cats;bc.categoryAxis.labels.fontSize=10;bc.categoryAxis.labels.boxAnchor='n';bc.categoryAxis.labels.dy=-5;bc.categoryAxis.strokeColor=HexColor('#94a3b8');bc.valueAxis.strokeColor=HexColor('#94a3b8');bc.valueAxis.valueMin=0
 if ymax is not None:bc.valueAxis.valueMax=ymax
 if step is not None:bc.valueAxis.valueStep=step
 bc.valueAxis.labels.fontSize=9;bc.valueAxis.visibleGrid=1;bc.valueAxis.gridStrokeColor=HexColor('#e2e8f0');bc.valueAxis.gridStrokeWidth=.4
 bc.bars.strokeColor=None;bc.barSpacing=3;bc.groupSpacing=14
 if stack:bc.categoryAxis.style='stacked'
 for i in range(len(data)):bc.bars[i].fillColor=(palette or [blue,amber])[i]
 d.add(bc);return bc
def legend(d,x,y,items):
 for col,txt in items:d.add(Rect(x,y-2,11,9,fillColor=HexColor(col),strokeColor=None));label(d,x+16,y,txt,9);x+=len(txt)*5.5+34

def save(d,name,meta):
 renderSVG.drawToFile(d,str(a.output_root/(name+'.svg')));(a.output_root/(name+'.json')).write_text(json.dumps(meta,indent=2)+'\n')

# One pass per physical record; free counts are categorical, not effective rates.
d=canvas(960,690,'Action categories and conditional order classes','144 fixed reference identities; nine actual-step quantiles; correlated records and duplicated arm0 identities retained')
ops=[]
for head in range(3):
 r=find(c['head_opportunities'],{}, {'pass_name':'second_cache_event_pass','head':head});ops.append(r);points.append(dict(figure='action_opportunities',head=head,n=r['n'],free=r['free']))
values=[100*r['free']/r['n'] for r in ops];chart(d,80,412,780,170,[values],['Execution','Cache','Prepare / priority'],100,25,palette=[blue]);label(d,80,604,'A. Categorical multichoice (%)',12,True)
for i,v in enumerate(values):label(d,210+i*260,425+v*1.7,f'{v:.2f}%  ({ops[i]["free"]}/{ops[i]["n"]})',10,False,anchor='middle')
label(d,80,380,'Multichoice does not establish different physical effects; forced cache can still admit a required model.',10)
cats=[];dist=[[],[],[],[]];meta=[]
for seed in (7,17,27):
 for arm,short in [('sa_legacy_clip','L'),('sa_group_clip','G')]:
  r=find(c['priority_reference_prepare_classes'],{'seed':seed,'reference_arm':arm});cats.append(f'{short}{seed} (n={r["captured_states"]})');meta.append(r)
  for j in range(4):dist[j].append(100*r['reference_prepare_order_class_distribution'].get(str(j+1),0)/r['captured_states'])
  points.extend(dict(figure='priority_classes',seed=seed,arm=arm,k=j+1,count=r['reference_prepare_order_class_distribution'].get(str(j+1),0),n=r['captured_states']) for j in range(4))
chart(d,80,145,780,170,dist,cats,100,25,True,[HexColor(x) for x in colors]);label(d,80,341,'B. Four priority labels at fixed reference prepare: number of order classes (%)',12,True)
legend(d,100,104,[(colors[i],f'{i+1} order class'+('' if i==0 else 'es')) for i in range(4)])
label(d,80,76,'L = legacy; G = grouped. Complete reference co-controls and original own claims held fixed.',10)
label(d,80,57,'Order classes are a one-frame sufficient condition, unavailable as a ready-to-use online mask.',10)
label(d,80,38,'Other prepare targets remain UNVERIFIED; these counts do not estimate training gradient waste.',10)
save(d,'action_and_priority_classes',dict(opportunities=ops,priority_by_arm_seed=meta))

# Same input comparisons of checkpoint variants, not alternate trajectories.
d=canvas(1080,760,'Same-input execution rankings','All six checkpoint variants score the same first-pass inputs per seed; conditional teacher-forced comparisons only')
allmeta=[]
for col,seed in enumerate((7,17,27)):
 s=e['per_seed_same_input_execution'][str(seed)];free=s['counts']['free'];x=65+col*345
 for row,(field,title) in enumerate([('strict_prior_override','Strict public-prior override'),('argmax_differs_own0','Argmax differs from own0')]):
  data=[]
  for arm in ('sa_legacy_clip','sa_group_clip'):
   nums=[s['models'][f'{arm}:{ep}'][field] for ep in (0,32,64)];data.append([100*v/free for v in nums]);allmeta.append(dict(seed=seed,arm=arm,metric=field,n=free,counts=nums))
   points.extend(dict(figure='rankings',seed=seed,arm=arm,metric=field,episode=ep,count=v,n=free) for ep,v in zip((0,32,64),nums))
  y=425 if row==0 else 145
  chart(d,x,y,270,170,data,['0','32','64'],35,7);label(d,x,y+195,f'Seed {seed}: {title}',11,True);label(d,x,y-48,f'Free inputs: n={free}; percentage of these rows',9)
legend(d,400,82,[('#2563eb','Legacy checkpoint'),('#f59e0b','Grouped checkpoint')]);label(d,65,50,'0 / 32 / 64 are the three actual checkpoint variants. More or fewer ranking flips is not better performance.',10)
save(d,'same_input_execution_rankings',dict(groups=allmeta))

# Factual deviation uses only the exact reference model, no cross-policy labels.
d=canvas(1020,560,'Fixed64 factual prediction deviation','Free execution records only; exact reference model versus its greedy physical team MC /80')
meta=[];data=[[],[]];cats=[]
for seed in (7,17,27):
 for arm,short in [('sa_legacy_clip','L'),('sa_group_clip','G')]:
  r=find(c['factual_reference_calibration'],{'seed':seed,'reference_arm':arm,'reference_episode':64},{'head':0,'forced_free':'free'});meta.append(r);cats.append(f'{short}{seed} (n={r["target_team_mc_over_80"]["n"]})')
  for j,k in enumerate(('counterfactual_baseline','global_value')):
   v=r['signed_error'][k]['mse_about_zero'];data[j].append(v);points.append(dict(figure='calibration',seed=seed,arm=arm,metric=k,n=r['target_team_mc_over_80']['n'],mse=v))
chart(d,80,157,850,275,data,cats,4,.8);legend(d,310,112,[('#2563eb','Baseline squared deviation'),('#f59e0b','Global V squared deviation')]);label(d,80,83,'MSE has squared MC/80 units. Different reference trajectories give different state panels.',10);label(d,80,62,'This is not the original stochastic PPO loss, a causal Q error, or a convergence curve.',10);label(d,80,41,'Legacy seed17 loses reward from0 to64 while7/27 gain, despite its smaller factual deviations.',10)
save(d,'factual_reference_deviation',dict(groups=meta,units='squared physical MC/80'))
(a.output_root/'plot_points.json').write_text(json.dumps(points,indent=2)+'\n');print(json.dumps({'status':'pass','figures':3,'points':len(points),'output':str(a.output_root),'tool':'ReportLab charts; zero neural/env/optimizer calls'}))
