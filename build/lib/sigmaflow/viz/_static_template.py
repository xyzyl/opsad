"""HTML/JS template for the static dashboard export.

The single placeholder ``__MANIFEST__`` is replaced with a JSON manifest
(signal list, detector explainers, narrative vocabularies) at export
time. All interactivity — thresholding, metrics, table, narrative — runs
client-side against precomputed scores.
"""

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sigmaflow dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root {
    --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --baseline: #c3c2b7;
    --series: #2a78d6; --critical: #d03b3b; --truth: #9ec5f4;
    --border: rgba(11,11,11,0.10);
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--ink);
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         padding: 20px 24px; }
  .header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 16px; }
  .header .sigma { font-size: 1.6rem; color: var(--series); font-weight: 700; }
  .header h1 { font-size: 1.15rem; font-weight: 650; margin: 0; }
  .header .sub { font-size: .85rem; color: var(--muted); }
  .row { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: 8px; padding: 16px; }
  .sidebar { width: 300px; flex-shrink: 0; }
  .main { flex: 1 1 480px; min-width: 0; display: flex; flex-direction: column; gap: 16px; }
  .label { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em;
           color: var(--muted); margin: 12px 0 2px; }
  .label:first-child { margin-top: 0; }
  select, input[type=number] {
    width: 100%; padding: 6px 8px; font-size: .85rem; color: var(--ink);
    background: #fff; border: 1px solid var(--baseline); border-radius: 6px;
  }
  .hint { font-size: .7rem; color: var(--muted); margin-top: 3px; }
  .note { font-size: .75rem; color: var(--ink2); background: var(--page);
          border: 1px solid var(--grid); border-radius: 6px; padding: 8px 10px;
          margin-top: 14px; line-height: 1.45; }
  .check { margin-top: 14px; font-size: .85rem; color: var(--ink2);
           display: flex; gap: 6px; align-items: center; }
  .tiles { display: flex; gap: 12px; flex-wrap: wrap; }
  .tile { background: var(--surface); border: 1px solid var(--border);
          border-radius: 8px; padding: 10px 18px; min-width: 120px; }
  .tile .t-label { font-size: .68rem; text-transform: uppercase;
                   letter-spacing: .06em; color: var(--muted); }
  .tile .t-value { font-size: 1.35rem; font-weight: 650; color: var(--ink); }
  #narrative { font-size: .88rem; line-height: 1.55; color: var(--ink2); max-width: 72ch; }
  #narrative p { margin: 0 0 8px; }
  .split { display: flex; gap: 16px; align-items: stretch; flex-wrap: wrap; }
  .split .card { min-width: 0; }
  .grow40 { flex: 1 1 320px; } .grow60 { flex: 1 1 420px; overflow: auto; }
  table { border-collapse: collapse; width: 100%; font-size: .8rem; color: var(--ink2); }
  th { color: var(--muted); font-weight: 600; text-transform: uppercase;
       font-size: .7rem; text-align: left; padding: 6px 10px;
       border-bottom: 1px solid var(--grid); }
  td { padding: 6px 10px; border-bottom: 1px solid var(--grid); }
  tbody tr { cursor: pointer; }
  tbody tr:hover { background: var(--page); }
  tbody tr.active { background: #f9dcdc; }
  .table-note { font-size: .7rem; color: var(--muted); margin-top: 6px; }
</style>
</head>
<body>
<div class="header">
  <span class="sigma">&sigma;</span>
  <h1>sigmaflow dashboard</h1>
  <span class="sub">interactive anomaly detection on live open data</span>
</div>
<div class="row">
  <div class="card sidebar">
    <div class="label">signal</div>
    <select id="signal-select"></select>
    <div class="label">channel</div>
    <select id="channel-select"></select>
    <div class="label">detector</div>
    <select id="detector-select"></select>
    <div class="hint" id="params-hint"></div>
    <div class="label">threshold method</div>
    <select id="threshold-method">
      <option value="auto">auto</option>
      <option value="percentile">percentile</option>
      <option value="sigma">sigma</option>
      <option value="fixed">fixed</option>
    </select>
    <div class="label">threshold value</div>
    <input id="threshold-value" type="number" step="any"
           placeholder="used by percentile / sigma / fixed">
    <label class="check"><input id="show-truth" type="checkbox" checked>
      show ground-truth bands</label>
    <div class="note">Every signal here is <strong>real instrument data</strong> from a
      public source, captured when this page was built (see the note under the
      interpretation). Detection scores are precomputed; the threshold, table, and
      interpretation update live in your browser. Live detector <em>re-fitting</em>
      (changing contamination, window size&hellip;) needs the local app:
      <code>pip install sigmaflow[dashboard]</code> then <code>sigmaflow dashboard</code>.</div>
  </div>
  <div class="main">
    <div class="tiles" id="tiles"></div>
    <div class="card">
      <div class="label">what is this showing?</div>
      <div id="narrative"></div>
      <div class="hint" id="source-note" style="margin-top:8px"></div>
    </div>
    <div class="card"><div id="main-graph" style="width:100%;height:520px"></div></div>
    <div class="split">
      <div class="card grow40"><div id="hist-graph" style="width:100%;height:260px"></div></div>
      <div class="card grow60">
        <div class="label">detected anomalies &mdash; click a row to zoom</div>
        <table>
          <thead><tr><th>#</th><th>start</th><th>end</th>
            <th>duration (s)</th><th>peak score</th></tr></thead>
          <tbody id="anomaly-rows"></tbody>
        </table>
        <div class="table-note" id="table-note"></div>
      </div>
    </div>
  </div>
</div>

<script>
const MANIFEST = __MANIFEST__;
const C = { surface:"#fcfcfb", ink2:"#52514e", muted:"#898781", grid:"#e1e0d9",
            baseline:"#c3c2b7", series:"#2a78d6", critical:"#d03b3b", truth:"#9ec5f4" };
const FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif';

const state = { data:null, channel:null, detector:MANIFEST.detectors.includes("isolation_forest")
                ? "isolation_forest" : MANIFEST.detectors[0],
                method:"auto", value:null, showTruth:true };

// ------------------------------------------------------------ helpers
const $ = id => document.getElementById(id);
const mean = a => a.reduce((s,x)=>s+x,0)/a.length;
const std = a => { const m = mean(a); return Math.sqrt(mean(a.map(x=>(x-m)*(x-m)))); };
function percentile(arr, p){
  const s=[...arr].sort((a,b)=>a-b), i=(p/100)*(s.length-1),
        lo=Math.floor(i), hi=Math.ceil(i);
  return s[lo]+(s[hi]-s[lo])*(i-lo);
}
function runs(labels){
  const out=[]; let i=0;
  while(i<labels.length){
    if(labels[i]){ let j=i; while(j+1<labels.length&&labels[j+1])j++; out.push([i,j]); i=j+1; }
    else i++;
  }
  return out;
}
const overlaps=(a,b)=>a[0]<=b[1]&&b[0]<=a[1];
function fmtDuration(s){
  if(s>=172800) return (s/86400).toFixed(0)+" days";
  if(s>=5400) return +(s/3600).toPrecision(3)+" hours";
  if(s>=120) return +(s/60).toPrecision(3)+" minutes";
  return +s.toPrecision(4)+" seconds";
}
function fmtTime(t,isDt){
  if(isDt) return new Date(t).toLocaleString(undefined,
    {month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"});
  return "t = "+(+Number(t).toPrecision(4))+" s";
}
function characterize(values,i0,i1,median,scale){
  const seg=values.slice(i0,i1+1);
  const mx=Math.max(...seg), mn=Math.min(...seg);
  if(seg.length>=4 && (mx-mn)<1e-12*Math.max(1,Math.abs(seg[0]))) return "flat";
  if(seg.length<=3) return "spike";
  if(std(seg)>3*scale) return "noisy";
  return "shift";
}
function secondsOf(t,isDt){ return isDt ? t/1000 : t; }

// ------------------------------------------------------------ state -> derived
function currentThreshold(scores, auto){
  const v = state.value;
  if(state.method==="auto") return auto;
  if(state.method==="percentile") return percentile(scores, v==null?99:v);
  if(state.method==="sigma"){ const m=mean(scores); return m+(v==null?3:v)*std(scores); }
  return v==null ? auto : v;   // fixed
}
function truthArray(){
  const n=state.data.n_samples, t=new Uint8Array(n);
  for(const [i0,i1] of state.data.truth_intervals)
    for(let i=i0;i<=i1;i++) t[i]=1;
  return t;
}

// ------------------------------------------------------------ narrative
function buildNarrative(labels, scores, threshold, detEvents){
  const d=state.data, ch=d.channels[state.channel];
  const isDt = d.time_kind==="datetime";
  const paras=[ch.para1];
  const pct=(100*labels.reduce((s,x)=>s+x,0)/labels.length).toFixed(2);
  const detLabel=state.detector.replace(/_/g," ");
  paras.push(`The ${detLabel} detector ${MANIFEST.explainers[state.detector]}. `+
    `The dashed red line in the lower chart is the alarm bar: every moment scoring above it `+
    `is declared an anomaly. At the current setting, ${pct}% of all readings clear it, `+
    `grouped into ${detEvents.length} distinct event${detEvents.length===1?"":"s"}.`);
  if(detEvents.length){
    let best=detEvents[0], bestScore=-Infinity;
    for(const ev of detEvents){
      let peak=-Infinity;
      for(let i=ev[0];i<=ev[1];i++) peak=Math.max(peak,scores[i]);
      if(peak>bestScore){ bestScore=peak; best=ev; }
    }
    const character=characterize(ch.values,best[0],best[1],ch.median,ch.scale);
    const when=fmtTime(d.time[best[0]],isDt);
    const span=secondsOf(d.time[best[1]],isDt)-secondsOf(d.time[best[0]],isDt);
    const spanStr=span>0?` for about ${fmtDuration(span)}`:"";
    const domain=(d.domain && MANIFEST.domainMeaning[d.domain])?d.domain:"generic";
    const meaning=MANIFEST.domainMeaning[domain][character];
    const domainName=domain==="generic"?"general instrumentation":domain;
    paras.push(`The strongest event is around ${when}, where the signal `+
      `${MANIFEST.characterPhrases[character]}${spanStr}. In ${domainName} terms, that `+
      `pattern most often means ${meaning}. ${MANIFEST.crossDomain[character]}`);
  } else {
    paras.push("Nothing currently clears the alarm bar — either this stretch of data is "+
      "genuinely unremarkable, or the bar is set too high. Try the percentile threshold "+
      "method with a value like 99 to surface the most unusual 1%.");
  }
  if(d.truth_intervals.length){
    const truth=truthArray();
    const truthRuns=d.truth_intervals;
    const caught=truthRuns.filter(t=>detEvents.some(p=>overlaps(p,t))).length;
    let tp=0,fp=0,fn=0,tn=0;
    for(let i=0;i<labels.length;i++){
      if(labels[i]&&truth[i])tp++; else if(labels[i])fp++;
      else if(truth[i])fn++; else tn++;
    }
    const fpr=fp+tn?fp/(fp+tn):0;
    let s=`This signal comes with an answer key: ${truthRuns.length} genuine `+
      `anomal${truthRuns.length===1?"y was":"ies were"} planted in it, and the detector `+
      `currently catches ${caught} of ${truthRuns.length} (the pale blue bands mark where `+
      `they really are).`;
    if(caught<truthRuns.length) s+=" Lowering the alarm bar would catch more — at the "+
      "price of more false alarms on ordinary wiggles.";
    else if(fpr>0.05) s+=" It also flags a fair number of ordinary moments, though — "+
      "raising the alarm bar would cut those false alarms.";
    paras.push(s);
  }
  return paras;
}

// ------------------------------------------------------------ rendering
function baseLayout(extra){
  return Object.assign({
    paper_bgcolor:C.surface, plot_bgcolor:C.surface,
    font:{family:FONT,color:C.ink2,size:12},
    margin:{l:56,r:16,t:36,b:40}, hovermode:"x unified", dragmode:"zoom",
  }, extra);
}
const axisStyle = {gridcolor:C.grid, linecolor:C.baseline,
                   tickfont:{color:C.muted}, zeroline:false};

function render(){
  const d=state.data; if(!d) return;
  const ch=d.channels[state.channel];
  const det=d.detectors[state.detector];
  const scores=det.scores;
  const threshold=currentThreshold(scores,det.auto_threshold);
  const labels=scores.map(s=>s>threshold?1:0);
  const detEvents=runs(labels);
  const isDt=d.time_kind==="datetime";

  // tiles
  const nFlag=labels.reduce((s,x)=>s+x,0);
  const tiles=[["anomalies",String(detEvents.length)],
    ["flagged samples",(100*nFlag/labels.length).toFixed(2)+"%"],
    ["max score",String(+Math.max(...scores).toPrecision(3))]];
  if(d.truth_intervals.length){
    const truth=truthArray();
    let tp=0,fp=0,fn=0,tn=0;
    for(let i=0;i<labels.length;i++){
      if(labels[i]&&truth[i])tp++; else if(labels[i])fp++;
      else if(truth[i])fn++; else tn++;
    }
    const prec=tp+fp?tp/(tp+fp):0, rec=tp+fn?tp/(tp+fn):0;
    const f1=prec+rec?2*prec*rec/(prec+rec):0;
    const evRec=d.truth_intervals.length ?
      d.truth_intervals.filter(t=>detEvents.some(p=>overlaps(p,t))).length /
      d.truth_intervals.length : 0;
    tiles.push(["F1 vs truth",f1.toFixed(3)],["event recall",evRec.toFixed(2)],
      ["false pos. rate",(fp+tn?fp/(fp+tn):0).toFixed(4)]);
  }
  $("tiles").innerHTML=tiles.map(([l,v])=>
    `<div class="tile"><div class="t-label">${l}</div><div class="t-value">${v}</div></div>`
  ).join("");

  // narrative + data provenance
  $("narrative").innerHTML=buildNarrative(labels,scores,threshold,detEvents)
    .map(p=>`<p>${p}</p>`).join("");
  $("source-note").textContent=d.source_note||"";

  // main figure: signal (top) + score (bottom), shared x
  const traces=[{x:d.time,y:ch.values,type:"scattergl",mode:"lines",name:state.channel,
    line:{color:C.series,width:1.6},yaxis:"y",hovertemplate:"%{y:.6g}<extra></extra>"}];
  const fi=[],fv=[];
  for(let i=0;i<labels.length;i++) if(labels[i]){ fi.push(d.time[i]); fv.push(ch.values[i]); }
  if(fi.length) traces.push({x:fi,y:fv,type:"scattergl",mode:"markers",
    name:"detected anomaly",marker:{color:C.critical,size:5,line:{color:C.surface,width:1}},
    yaxis:"y",hovertemplate:"anomaly %{y:.6g}<extra></extra>"});
  traces.push({x:d.time,y:scores,type:"scattergl",mode:"lines",name:"anomaly score",
    showlegend:false,line:{color:C.series,width:1.6},yaxis:"y2",
    hovertemplate:"%{y:.4g}<extra></extra>"});

  const shapes=[{type:"line",xref:"paper",x0:0,x1:1,yref:"y2",y0:threshold,y1:threshold,
    line:{color:C.critical,width:1.5,dash:"dash"}}];
  if(state.showTruth){
    const span=secondsOf(d.time[d.time.length-1],isDt)-secondsOf(d.time[0],isDt);
    const pad=(isDt?1000:1)*span/500/2;
    for(const [i0,i1] of d.truth_intervals)
      shapes.push({type:"rect",xref:"x",yref:"y domain",y0:0,y1:1,
        x0:d.time[i0]-pad,x1:d.time[i1]+pad,fillcolor:C.truth,opacity:0.4,
        line:{width:0},layer:"below"});
  }
  const unit=ch.unit?` [${ch.unit}]`:"";
  Plotly.react("main-graph",traces,baseLayout({
    xaxis:Object.assign({anchor:"y2",type:isDt?"date":"linear"},axisStyle),
    yaxis:Object.assign({domain:[0.44,1],title:{text:state.channel+unit,
      font:{size:12,color:C.ink2}}},axisStyle),
    yaxis2:Object.assign({domain:[0,0.36],title:{text:"score",
      font:{size:12,color:C.ink2}}},axisStyle),
    shapes,
    legend:{orientation:"h",yanchor:"bottom",y:1.02,x:0,font:{size:11,color:C.ink2}},
    annotations:[{xref:"paper",x:1,yref:"y2",y:threshold,text:"threshold "+
      (+threshold.toPrecision(3)),showarrow:false,xanchor:"right",yanchor:"bottom",
      font:{color:C.critical,size:11}}],
  }),{displaylogo:false,responsive:true});

  // histogram
  Plotly.react("hist-graph",[{x:scores,type:"histogram",nbinsx:60,
    marker:{color:C.series,line:{color:C.surface,width:1}},
    hovertemplate:"score %{x}<br>count %{y}<extra></extra>"}],
    baseLayout({title:{text:"score distribution (log count)",
      font:{size:13,color:C.ink2}},showlegend:false,
      xaxis:axisStyle,yaxis:Object.assign({type:"log"},axisStyle),
      shapes:[{type:"line",xref:"x",x0:threshold,x1:threshold,yref:"paper",y0:0,y1:1,
        line:{color:C.critical,width:1.5,dash:"dash"}}]}),
    {displaylogo:false,responsive:true});

  // table (cap rows to keep the DOM light)
  const MAXROWS=200;
  const rows=detEvents.slice(0,MAXROWS).map((ev,k)=>{
    let peak=-Infinity;
    for(let i=ev[0];i<=ev[1];i++) peak=Math.max(peak,scores[i]);
    const dur=secondsOf(d.time[ev[1]],isDt)-secondsOf(d.time[ev[0]],isDt);
    const f=t=>isDt?new Date(t).toLocaleString(undefined,
      {month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"}):(+Number(t).toPrecision(6));
    return `<tr data-i0="${ev[0]}" data-i1="${ev[1]}"><td>${k+1}</td><td>${f(d.time[ev[0]])}</td>`+
      `<td>${f(d.time[ev[1]])}</td><td>${+dur.toPrecision(4)}</td>`+
      `<td>${+peak.toPrecision(4)}</td></tr>`;
  });
  $("anomaly-rows").innerHTML=rows.join("");
  $("table-note").textContent=detEvents.length>MAXROWS?
    `showing the first ${MAXROWS} of ${detEvents.length} events`:"";
  for(const tr of $("anomaly-rows").children){
    tr.onclick=()=>{
      for(const o of $("anomaly-rows").children) o.classList.remove("active");
      tr.classList.add("active");
      const i0=+tr.dataset.i0, i1=+tr.dataset.i1;
      const s0=secondsOf(d.time[i0],isDt), s1=secondsOf(d.time[i1],isDt);
      const total=secondsOf(d.time[d.time.length-1],isDt)-secondsOf(d.time[0],isDt);
      const span=Math.max(s1-s0,total/100), k=isDt?1000:1;
      Plotly.relayout("main-graph",
        {"xaxis.range":[d.time[i0]-2*span*k, d.time[i1]+2*span*k]});
      window.scrollTo({top:$("main-graph").offsetTop-80,behavior:"smooth"});
    };
  }
}

// ------------------------------------------------------------ wiring
async function loadSignal(slug){
  const res=await fetch(`data/${slug}.json`);
  state.data=await res.json();
  const chans=Object.keys(state.data.channels);
  $("channel-select").innerHTML=chans.map(c=>`<option>${c}</option>`).join("");
  state.channel=chans[0];
  updateParamsHint();
  render();
}
function updateParamsHint(){
  if(!state.data) return;
  const p=state.data.detectors[state.detector].params;
  const txt=Object.entries(p).map(([k,v])=>`${k}=${v}`).join(", ");
  $("params-hint").textContent=txt?`precomputed with ${txt}`:"precomputed with defaults";
}
function init(){
  $("signal-select").innerHTML=MANIFEST.signals
    .map(s=>`<option value="${s.slug}">${s.name}</option>`).join("");
  $("detector-select").innerHTML=MANIFEST.detectors
    .map(m=>`<option ${m===state.detector?"selected":""}>${m}</option>`).join("");
  $("signal-select").onchange=e=>loadSignal(e.target.value);
  $("channel-select").onchange=e=>{state.channel=e.target.value;render();};
  $("detector-select").onchange=e=>{state.detector=e.target.value;updateParamsHint();render();};
  $("threshold-method").onchange=e=>{state.method=e.target.value;render();};
  $("threshold-value").oninput=e=>{
    state.value=e.target.value===""?null:+e.target.value;render();};
  $("show-truth").onchange=e=>{state.showTruth=e.target.checked;render();};
  loadSignal(MANIFEST.signals[0].slug);
}
init();
</script>
</body>
</html>
"""
