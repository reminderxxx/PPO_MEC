"""Offline visual replay of the measured corridor and eight active requests."""

from __future__ import annotations

import json
from typing import Any

from src.envs.core.vec_competition_contract import FrozenScenario


def render_replay_html(scenario: FrozenScenario, timeline: list[dict[str, Any]], ledger: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    events = [event for event in ledger if event["event"] in {"request_arrived", "handoff", "coverage_lost", "coverage_attached", "model_resident", "cache_evicted", "compute_started", "compute_completed", "workflow_delivered", "deadline_missed", "request_unfinished"}]
    data = {
        "scenario_id": scenario.scenario_id,
        "traces": {vehicle_id: [{"t": point.time_s, "x": point.x_m, "y": point.y_m, "road": point.road_key} for point in trace] for vehicle_id, trace in scenario.traces.items()},
        "rsus": [{"id": rsu.rsu_id, "x": rsu.x_m, "y": rsu.y_m, "r": rsu.radius_m} for rsu in scenario.rsus],
        "timeline": timeline,
        "events": events,
        "requests": {vehicle_id: {"arrival_s": request.arrival_s, "deadline_s": request.deadline_s} for vehicle_id, request in scenario.requests.items()},
        "results": summary["per_vehicle"],
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>4 RSU / 8 车辆工程回放</title>
<style>
body{margin:0;background:#07142a;color:#eef5ff;font:15px/1.45 system-ui,-apple-system,sans-serif}
main{max-width:1320px;margin:auto;padding:24px}h1{font-size:24px;margin:0 0 6px}p{color:#b7c6d8;margin:0 0 18px}
.grid{display:grid;grid-template-columns:minmax(440px,2fr) minmax(300px,1fr);gap:18px}.panel{background:#102443;border:1px solid #315071;border-radius:12px;padding:16px}
svg{display:block;width:100%;height:auto;background:#0a1b33;border-radius:8px}.controls{display:flex;align-items:center;gap:14px;margin-top:12px}
input[type=range]{flex:1}button{background:#2b75e6;border:0;border-radius:8px;color:white;padding:8px 15px;cursor:pointer}
#clock{font-variant-numeric:tabular-nums;min-width:74px}.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:0 0 15px}
.stat{background:#193457;border-radius:8px;padding:9px}.stat strong{font-size:21px;display:block;color:#fff}.stat small{color:#b7c6d8}
.vehicles{display:grid;grid-template-columns:1fr 1fr;gap:7px}.vehicle{background:#193457;border-radius:7px;padding:7px;font-size:12px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:6px}.events{max-height:310px;overflow:auto;font-size:12px}
.event{border-bottom:1px solid #315071;padding:5px 0}.muted{color:#93a8c3}h2{font-size:16px;margin:16px 0 8px}
@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style></head><body><main><h1>4 个 RSU · 8 辆请求车</h1>
<p>同一段 NGSIM 观测轨迹；圆形 RSU 覆盖、带宽与计算速率为受控实验参数。每个彩色圆点都有独立工作流请求。</p>
<div class="grid"><section class="panel"><svg id="map" viewBox="0 0 760 650" role="img" aria-label="车辆与RSU接入回放"></svg>
<div class="controls"><button id="play">播放</button><input id="frame" type="range" min="0" max="119" value="0"><span id="clock">0.0 s</span></div></section>
<aside class="panel"><div class="stats"><div class="stat"><strong id="active">8</strong><small>已产生请求</small></div><div class="stat"><strong id="delivered">0</strong><small>已交付</small></div><div class="stat"><strong id="handoffs">0</strong><small>累计切换</small></div><div class="stat"><strong id="busy">0</strong><small>正在计算</small></div></div>
<h2>车辆与接入</h2><div id="vehicles" class="vehicles"></div><h2>当前事件</h2><div id="events" class="events"></div></aside></div>
<p style="margin-top:15px">本图展示局部坐标和工程事件，不代表 GIS 全路网、真实 RSU 站址或真实模型推理。</p>
</main><script id="data" type="application/json">""" + payload + """</script><script>
const d=JSON.parse(document.getElementById('data').textContent),svg=document.getElementById('map');
const ids=Object.keys(d.traces),colors=['#ff685b','#5bd5f6','#f9c74f','#9bdb82','#d79afa','#fb8cba','#82a8ff','#ffab5c'];
const all=ids.flatMap(id=>d.traces[id]),rs=d.rsus;
const x0=Math.min(...all.map(p=>p.x),...rs.map(r=>r.x-r.r))-10,x1=Math.max(...all.map(p=>p.x),...rs.map(r=>r.x+r.r))+10;
const y0=Math.min(...all.map(p=>p.y),...rs.map(r=>r.y-r.r))-10,y1=Math.max(...all.map(p=>p.y),...rs.map(r=>r.y+r.r))+10;
const scale=Math.min(700/(x1-x0),590/(y1-y0));const X=x=>30+(x-x0)*scale,Y=y=>620-(y-y0)*scale;
function circle(x,y,r,fill,stroke='',width=1){return `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${width}"/>`}
function draw(i){const t=d.timeline[i].time_s;document.getElementById('clock').textContent=t.toFixed(1)+' s';
let marks='';for(const r of rs){marks+=circle(X(r.x),Y(r.y),r.r*scale,'#2d63a526','#5c92ce88',1.2);marks+=circle(X(r.x),Y(r.y),6,'#b7d9ff');marks+=`<text x="${X(r.x)+11}" y="${Y(r.y)-9}" fill="#d4e7ff" font-size="14">${r.id}</text>`}
ids.forEach((id,k)=>{const pts=d.traces[id],p=pts[i];marks+=`<polyline points="${pts.map(q=>X(q.x)+','+Y(q.y)).join(' ')}" fill="none" stroke="${colors[k]}" stroke-width="1.1" opacity="0.55"/>`;marks+=circle(X(p.x),Y(p.y),6.5,colors[k],'#fff',1.5);marks+=`<text x="${X(p.x)+9}" y="${Y(p.y)+4}" fill="#fff" font-size="12">${id.split(':')[1]}</text>`});svg.innerHTML=marks;
let delivered=0,handoffs=0,busy=0,active=0;const running=new Set();for(const e of d.events){if(e.time_s>t+1e-9)break;if(e.event==='request_arrived')active++;if(e.event==='workflow_delivered')delivered++;if(e.event==='handoff')handoffs++;if(e.event==='compute_started')running.add(e.vehicle_id+'|'+e.node_id);if(e.event==='compute_completed')running.delete(e.vehicle_id+'|'+e.node_id)}busy=running.size;
document.getElementById('active').textContent=active;document.getElementById('delivered').textContent=delivered;document.getElementById('handoffs').textContent=handoffs;document.getElementById('busy').textContent=busy;
document.getElementById('vehicles').innerHTML=ids.map((id,k)=>`<div class="vehicle"><span class="dot" style="background:${colors[k]}"></span>${id} <span class="muted">→ ${d.timeline[i].attached[id]||'断覆盖'}</span></div>`).join('');
const current=d.events.filter(e=>e.time_s<=t+1e-9&&e.time_s>=Math.max(0,t-0.3)-1e-9).slice(-16).reverse();
document.getElementById('events').innerHTML=current.map(e=>`<div class="event"><span class="muted">${e.time_s.toFixed(1)}s</span> ${e.event} ${e.vehicle_id||''} ${e.node_id||''}</div>`).join('')||'<span class="muted">无事件</span>'}
const slider=document.getElementById('frame');slider.addEventListener('input',()=>draw(Number(slider.value)));let timer=null;
document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='播放';return}document.getElementById('play').textContent='暂停';timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%d.timeline.length;draw(Number(slider.value))},100)};draw(0);
</script></body></html>"""
