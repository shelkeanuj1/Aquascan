"""Per-photo report like a results card: measured values + every model's prediction, in original units (pH, NTU, mg/L).
Needs oof_*.csv (written by train.py) and merged_dataset/. Writes report/image_report.html and report/per_image_wide.csv"""
import base64, glob, io, json, os, numpy as np, pandas as pd
from PIL import Image
T, LOG = ["ph", "turbidity", "bod", "cod"], {"turbidity", "bod", "cod"}
ORDER = ["CNN", "EfficientNetB0", "ResNet50", "MobileNetV2", "VGG16", "VGG19", "Xception"]
m = pd.read_csv("merged_dataset/master.csv").set_index("id"); os.makedirs("report", exist_ok=True)
conv = lambda t, v: round(max(float(np.expm1(v) if t in LOG else v), 0.0), 2)        # undo log, no negative values
preds = {os.path.basename(f)[4:-4]: pd.read_csv(f).set_index("id") for f in glob.glob("oof_*.csv")}
models = [x for x in ORDER if x in preds] + [x for x in preds if x not in ORDER]
data, wide = [], []
for i in m.index[m.is_synthetic == 0]:
    buf = io.BytesIO(); im = Image.open(f"merged_dataset/{m.loc[i, 'file']}"); im.thumbnail((360, 360)); im.save(buf, "JPEG", quality=75)
    actual = {t: (None if pd.isna(m.loc[i, t]) else float(m.loc[i, t])) for t in T}
    pr = {n: {t: conv(t, preds[n].loc[i, f"pred_{t}"]) for t in T} for n in models}
    data.append(dict(id=i, img="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), actual=actual, pred=pr))
    wide.append({"image": i, **{f"actual_{t}": actual[t] for t in T}, **{f"{n}_{t}": pr[n][t] for n in models for t in T}})
pd.DataFrame(wide).to_csv("report/per_image_wide.csv", index=False)
HTML = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Per-photo model comparison</title><style>
:root{--bg:#fff;--ink:#14212b;--mut:#66788a;--line:#e1e8ee;--ok:#1f8a4c;--bad:#c0392b;--card:#f6f9fb}
@media(prefers-color-scheme:dark){:root{--bg:#0d1418;--ink:#e8eef2;--mut:#8aa0b0;--line:#22313b;--ok:#4cc380;--bad:#f0776b;--card:#141f26}}
body{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--ink)}main{max-width:820px;margin:0 auto;padding:20px}
.bar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}select,input,button{font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink)}select{flex:1;min-width:180px}
img{width:100%;max-width:360px;border-radius:12px;display:block}.row{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--line)}
h2{font-size:12px;letter-spacing:.06em;color:var(--mut);margin:22px 0 6px}table{width:100%;border-collapse:collapse}th,td{padding:10px 8px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad)}small{color:var(--mut)}.wrap{overflow-x:auto}</style>
<main><h1 style="font-size:22px">Per-photo model comparison</h1><div class="bar"><input id="q" placeholder="Search photo"><select id="s"></select><button id="p">Prev</button><button id="n">Next</button></div>
<div id="v"></div><p><small>Green = close to the measured value (pH within ±1; turbidity, BOD and COD within 0.5x to 2x). Every prediction comes from a model that did not train on this photo.
Units: pH, turbidity in NTU, BOD and COD in mg/L. Values below 0 are shown as 0.</small></p></main>
<script>const D=__DATA__,M=__MODELS__,T=[["ph","pH"],["turbidity","Turbidity"],["bod","BOD"],["cod","COD"]];let cur=0;
const s=document.getElementById("s"),q=document.getElementById("q"),v=document.getElementById("v");
const good=(t,a,p)=>a==null?null:t=="ph"?Math.abs(p-a)<=1:((p+1)/(a+1)>=.5&&(p+1)/(a+1)<=2);
function fill(){const f=q.value.toLowerCase();s.innerHTML="";D.forEach((d,i)=>{if(d.id.toLowerCase().includes(f)){const o=document.createElement("option");o.value=i;o.textContent=d.id;s.appendChild(o)}});if(s.options.length){cur=+s.value;show()}else v.innerHTML="<p>No match.</p>"}
function show(){const d=D[cur];let h=`<img src="${d.img}" alt="${d.id}"><h2>MEASURED VALUES</h2>`;
T.forEach(([k,n])=>h+=`<div class="row"><span>${n}</span><span>${d.actual[k]==null?"not measured":d.actual[k]}</span></div>`);
h+=`<h2>PREDICTIONS FROM EACH MODEL</h2><div class="wrap"><table><tr><th>Model</th>${T.map(([k,n])=>`<th>Predicted ${n}</th>`).join("")}</tr>`;
M.forEach(m=>{h+=`<tr><td>${m}</td>`+T.map(([k])=>{const p=d.pred[m][k],g=good(k,d.actual[k],p);return `<td class="${g===true?"ok":g===false?"bad":""}">${p}</td>`}).join("")+"</tr>"});v.innerHTML=h+"</table></div>"}
s.onchange=()=>{cur=+s.value;show()};q.oninput=fill;
document.getElementById("n").onclick=()=>{if(s.selectedIndex<s.options.length-1){s.selectedIndex++;s.onchange()}};
document.getElementById("p").onclick=()=>{if(s.selectedIndex>0){s.selectedIndex--;s.onchange()}};fill();</script></html>"""
open("report/image_report.html", "w", encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(data)).replace("__MODELS__", json.dumps(models)))
print(f"{len(data)} photos, {len(models)} models -> report/image_report.html and report/per_image_wide.csv")
