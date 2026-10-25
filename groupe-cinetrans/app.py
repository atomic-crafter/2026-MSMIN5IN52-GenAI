"""Interface web du pipeline complet (Flask) :
extractsrt.py (transcription) -> translate_json.py (traduction) -> dubbing.py (vidéo doublée).

docker compose up --build   puis   http://localhost:8000
Fonctionne à l'identique sur Windows, macOS et Linux : seul un navigateur est nécessaire.
"""
import json
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from dubbing import LANG_MAP as DUBBING_LANGS  # langues pour lesquelles on a une voix Kokoro

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"

LANGUAGES = {
    "fr": "Français", "en": "Anglais", "es": "Espagnol", "de": "Allemand",
    "it": "Italien", "pt": "Portugais", "nl": "Néerlandais", "ru": "Russe",
    "ar": "Arabe", "zh": "Chinois", "ja": "Japonais", "ko": "Coréen",
}

WHISPER_MODELS = {  # nom faster-whisper -> (libellé, indication de poids)
    "tiny": ("Tiny", "~75 Mo · le plus rapide, approximatif"),
    "base": ("Base", "~140 Mo · rapide"),
    "small": ("Small", "~460 Mo · bon compromis"),
    "medium": ("Medium", "~1,5 Go · précis, lent sur CPU"),
    "large-v3-turbo": ("Large-v3 Turbo", "~1,6 Go · presque aussi précis que large, ~2× plus rapide"),
    "large-v3": ("Large-v3", "~3 Go · le plus précis, très lent sur CPU"),
}
EMOTION_LEVELS = {
    1: ("Léger", "wav2vec2-base · ~95 M paramètres · 4 émotions"),
    2: ("Standard", "wav2vec2-large XLSR · ~316 M · 8 émotions"),
    3: ("Précis", "Whisper-large-v3 (encodeur) · ~637 M · 7 émotions · le plus lent"),
}

app = Flask(__name__)
jobs = {}  # id -> {"state": running|done|error, "log": str, "files": [..]}
run_lock = threading.Lock()  # un seul traitement à la fois (modèles lourds)


def stream(job, cmd):
    """Lance une commande, ajoute sa sortie au journal du job, retourne son code de retour."""
    try:
        proc = subprocess.Popen(cmd, cwd=BASE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            job["log"] += line
        return proc.wait()
    except OSError as e:
        job["log"] += f"Impossible de lancer le traitement : {e}\n"
        return 1


def run_job(job_id, video, out_dir, args, targets):
    job = jobs[job_id]
    with run_lock:
        job["state"] = "running"
        code = stream(job, [sys.executable, "-u", "extractsrt.py", "--video", str(video),
                            "--out-dir", str(out_dir), *args])
        if code == 0 and targets:
            code = translate_and_dub(job, video, out_dir, targets)
    job["files"] = sorted(p.name for p in out_dir.iterdir()
                          if p.suffix in {".srt", ".json", ".wav", ".mp4"}
                          and not p.name.endswith(".report.json"))
    job["state"] = "done" if code == 0 else "error"


def translate_and_dub(job, video, out_dir, targets):
    source_json = out_dir / f"{video.stem}.json"
    skipped = [t for t in targets if t not in DUBBING_LANGS]
    if skipped:
        job["log"] += f"Pas de voix de synthèse pour : {', '.join(skipped)} (ignoré)\n"
    targets = [t for t in targets if t in DUBBING_LANGS]
    if not targets:
        return 1

    job["log"] += "[4/5] Traduction…\n"
    for lang in targets:
        code = stream(job, [sys.executable, "-u", "translate_json.py", str(source_json),
                            "--target-lang", lang])
        if code != 0:
            return code

    job["log"] += "[5/5] Doublage…\n"
    failed = 0
    for lang in targets:
        # le JSON traduit a target_languages = [lang] : dubbing.py ne synthétise que cette langue
        failed += stream(job, [sys.executable, "-u", "dubbing.py",
                               str(out_dir / f"segment_{lang}.json"),
                               "--video", str(video), "--output-dir", str(out_dir)]) != 0
    return 1 if failed else 0


@app.post("/run")
def run():
    file = request.files.get("video")
    if not file or not file.filename:
        return jsonify(error="Aucune vidéo envoyée"), 400
    source = request.form.get("source_lang", "auto")
    if source != "auto" and source not in LANGUAGES:
        return jsonify(error="Langue source invalide"), 400
    targets = [t for t in request.form.getlist("targets") if t in LANGUAGES]

    whisper_model = request.form.get("whisper_model", "large-v3")
    if whisper_model not in WHISPER_MODELS:
        return jsonify(error="Modèle Whisper invalide"), 400
    try:
        emotion_level = int(request.form.get("emotion_level", 2))
        sensitivity = int(request.form.get("speaker_sensitivity", 2))
        num_speakers = request.form.get("num_speakers", "").strip()
        num_speakers = int(num_speakers) if num_speakers else None
    except ValueError:
        return jsonify(error="Valeur numérique invalide"), 400
    if emotion_level not in EMOTION_LEVELS or sensitivity not in range(1, 6) \
            or (num_speakers is not None and not 1 <= num_speakers <= 12):
        return jsonify(error="Réglage hors limites"), 400

    job_id = uuid.uuid4().hex[:8]
    name = secure_filename(file.filename) or "video.mp4"
    (UPLOADS / job_id).mkdir(parents=True, exist_ok=True)
    out_dir = OUTPUTS / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    video = UPLOADS / job_id / name
    file.save(video)

    args = ["--source-lang", source, "--target-langs", ",".join(targets),
            "--whisper-model", whisper_model, "--emotion-level", str(emotion_level),
            "--speaker-sensitivity", str(sensitivity)]
    if num_speakers:
        args += ["--num-speakers", str(num_speakers)]
    if request.form.get("multilingual"):
        args.append("--multilingual")
    jobs[job_id] = {"state": "queued", "log": "", "files": []}
    threading.Thread(target=run_job, args=(job_id, video, out_dir, args, targets), daemon=True).start()
    return jsonify(id=job_id)


@app.get("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if job is None:
        abort(404)
    return jsonify(job)


@app.get("/download/<job_id>/<name>")
def download(job_id, name):
    if job_id not in jobs:
        abort(404)
    return send_from_directory(OUTPUTS / job_id, name, as_attachment=True)


@app.get("/")
def index():
    langs = "".join(f'<option value="{c}">{n} ({c})</option>' for c, n in LANGUAGES.items())
    checks = "".join(
        f'<label class="chip"><input type="checkbox" name="targets" value="{c}"><span>{n}</span></label>'
        for c, n in LANGUAGES.items()
    )
    models = "".join(
        f'<option value="{k}" data-hint="{hint}"{" selected" if k == "large-v3" else ""}>{label}</option>'
        for k, (label, hint) in WHISPER_MODELS.items()
    )
    hints = {str(k): f"{label} — {hint}" for k, (label, hint) in EMOTION_LEVELS.items()}
    return (PAGE.replace("{{LANGS}}", langs).replace("{{CHECKS}}", checks)
            .replace("{{MODELS}}", models).replace("{{EMOTION_HINTS}}", json.dumps(hints)))


PAGE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CineTrans</title>
<style>
:root{--bg:#f6f7fb;--card:#fff;--ink:#1c2030;--muted:#6a7189;--line:#e3e6ef;--accent:#5b5bf0;--accent-ink:#fff;--ok:#1f9d6b;--bad:#d64545;--code:#0f1220;--codeink:#cfd4ea}
@media (prefers-color-scheme:dark){:root{--bg:#0e1019;--card:#171a27;--ink:#e8eaf4;--muted:#8b92ad;--line:#262a3c;--accent:#7c7cff;--accent-ink:#0e1019;--code:#090b13}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:32px 16px 64px}
header{display:flex;align-items:baseline;gap:12px;margin-bottom:24px}
header h1{margin:0;font-size:26px;letter-spacing:-.02em}
header span{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:16px}
.card h2{margin:0 0 4px;font-size:15px}
.card p.sub{margin:0 0 14px;color:var(--muted);font-size:13px}
.drop{display:block;border:2px dashed var(--line);border-radius:12px;padding:28px;text-align:center;cursor:pointer;transition:.15s}
.drop:hover,.drop.over{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 6%,transparent)}
.drop strong{display:block;font-size:16px}.drop small{color:var(--muted)}
.drop.has{border-style:solid;border-color:var(--ok)}
input[type=file]{display:none}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:640px){.grid2{grid-template-columns:1fr}}
label.f{display:block;font-weight:600;font-size:13px;margin-bottom:6px}
select,input[type=number]{width:100%;padding:9px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);font:inherit}
.hint{margin-top:6px;color:var(--muted);font-size:12.5px;min-height:18px}
input[type=range]{width:100%;accent-color:var(--accent)}
.ticks{display:flex;justify-content:space-between;color:var(--muted);font-size:12px}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip input{display:none}
.chip span{display:inline-block;padding:6px 12px;border:1px solid var(--line);border-radius:999px;cursor:pointer;font-size:13px;user-select:none}
.chip input:checked+span{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
.check{display:flex;gap:8px;align-items:center;margin-top:12px;font-size:14px}
button.go{width:100%;padding:13px;border:0;border-radius:10px;background:var(--accent);color:var(--accent-ink);font-family:inherit;font-weight:600;font-size:16px;cursor:pointer}
button.go:disabled{opacity:.5;cursor:default}
.steps{display:flex;gap:8px;margin:4px 0 14px}
.step{flex:1;padding:8px 10px;border-radius:8px;background:var(--bg);border:1px solid var(--line);font-size:12.5px;color:var(--muted)}
.step.on{border-color:var(--accent);color:var(--ink);font-weight:600}
.step.done{border-color:var(--ok);color:var(--ok)}
.status{display:flex;justify-content:space-between;font-size:13px;color:var(--muted);margin-bottom:8px}
pre{background:var(--code);color:var(--codeink);padding:14px;border-radius:10px;max-height:300px;overflow:auto;white-space:pre-wrap;font:12.5px/1.45 ui-monospace,Menlo,Consolas,monospace;margin:0}
.files a{display:inline-block;margin:4px 8px 0 0;padding:8px 14px;border:1px solid var(--line);border-radius:8px;color:var(--accent);text-decoration:none;font-weight:600}
.files a:hover{border-color:var(--accent)}
.err{color:var(--bad);font-weight:600}
</style></head><body><div class="wrap">
<header><h1>CineTrans</h1><span>vidéo → SRT avec locuteurs, émotions et langues</span></header>

<form id="f">
 <div class="card">
  <h2>Vidéo</h2><p class="sub">Glissez un fichier ou cliquez pour parcourir.</p>
  <label class="drop" id="drop"><input type="file" name="video" id="video" accept="video/*" required>
   <strong id="dropTitle">Choisir une vidéo</strong><small id="dropSub">mp4, mkv, mov, avi, webm…</small></label>
 </div>

 <div class="card">
  <h2>Langues</h2><p class="sub">Langue parlée dans la vidéo et langues de destination.</p>
  <div class="grid2">
   <div><label class="f" for="src">Langue de la vidéo</label>
    <select name="source_lang" id="src"><option value="auto">Auto-détection</option>{{LANGS}}</select>
    <label class="check"><input type="checkbox" name="multilingual" id="multi" value="1"> Plusieurs langues dans la vidéo</label></div>
   <div><label class="f">Traduire vers</label><div class="chips">{{CHECKS}}</div></div>
  </div>
 </div>

 <div class="card">
  <h2>Qualité &amp; puissance</h2><p class="sub">Plus le modèle est gros, plus il est précis… et lent (surtout sans GPU).</p>
  <div class="grid2">
   <div><label class="f" for="wm">Transcription (Whisper)</label>
    <select name="whisper_model" id="wm">{{MODELS}}</select><div class="hint" id="wmHint"></div></div>
   <div><label class="f" for="em">Émotions</label>
    <input type="range" name="emotion_level" id="em" min="1" max="3" value="2">
    <div class="ticks"><span>Léger</span><span>Standard</span><span>Précis</span></div>
    <div class="hint" id="emHint"></div></div>
  </div>
 </div>

 <div class="card">
  <h2>Locuteurs</h2><p class="sub">Si les voix se confondent, indiquez le nombre exact ou augmentez la séparation.</p>
  <div class="grid2">
   <div><label class="f" for="ns">Nombre de locuteurs</label>
    <input type="number" name="num_speakers" id="ns" min="1" max="12" placeholder="Auto">
    <div class="hint">Laisser vide pour une détection automatique. Le plus efficace si vous le connaissez.</div></div>
   <div><label class="f" for="ss">Séparation des voix</label>
    <input type="range" name="speaker_sensitivity" id="ss" min="1" max="5" value="2">
    <div class="ticks"><span>Fusionne</span><span>Défaut</span><span>Sépare plus</span></div>
    <div class="hint">Montez si deux personnes sont prises pour une seule ; baissez si une personne est coupée en deux.</div></div>
  </div>
 </div>

 <button class="go" id="go">Lancer le traitement</button>
</form>

<div class="card" id="run" style="display:none;margin-top:16px">
 <div class="steps"><div class="step" id="s1">1 · Audio</div><div class="step" id="s2">2 · Texte &amp; locuteurs</div><div class="step" id="s3">3 · Émotions</div><div class="step" id="s4">4 · Traduction</div><div class="step" id="s5">5 · Doublage</div></div>
 <div class="status"><span id="state">En attente…</span><span id="timer">0:00</span></div>
 <pre id="log"></pre>
 <div class="files" id="files"></div>
</div>
</div>
<script>
const $=id=>document.getElementById(id), EM={{EMOTION_HINTS}};
const fileIn=$('video'),drop=$('drop');
function showFile(){const f=fileIn.files[0];drop.classList.toggle('has',!!f);if(f){$('dropTitle').textContent=f.name;$('dropSub').textContent=(f.size/1048576).toFixed(1)+' Mo — cliquer pour changer'}}
fileIn.onchange=showFile;
['dragenter','dragover'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',ev=>{if(ev.dataTransfer.files.length){fileIn.files=ev.dataTransfer.files;showFile()}});
const wm=$('wm');function wmHint(){$('wmHint').textContent=wm.selectedOptions[0].dataset.hint}wm.onchange=wmHint;wmHint();
const em=$('em');function emHint(){$('emHint').textContent=EM[em.value]}em.oninput=emHint;emHint();
$('multi').onchange=e=>{if(e.target.checked)$('src').value='auto';$('src').disabled=e.target.checked};

let timer,t0;
function tick(){const s=Math.floor((Date.now()-t0)/1000);$('timer').textContent=Math.floor(s/60)+':'+String(s%60).padStart(2,'0')}
function steps(log){
 const n=[...log.matchAll(/\[(\d)\/[35]\]/g)].map(m=>+m[1]).pop()||0;
 [1,2,3,4,5].forEach(i=>{$('s'+i).className='step'+(i<n?' done':i==n?' on':'')});
 return n;
}
$('f').onsubmit=async e=>{
 e.preventDefault();
 const fd=new FormData($('f'));if($('src').disabled)fd.set('source_lang','auto');
 $('go').disabled=true;$('run').style.display='block';$('files').innerHTML='';
 $('log').textContent='Envoi de la vidéo…';$('state').textContent='Envoi…';steps('');
 t0=Date.now();clearInterval(timer);timer=setInterval(tick,1000);
 let j;try{const r=await fetch('/run',{method:'POST',body:fd});j=await r.json();if(!r.ok)throw new Error(j.error)}
 catch(err){$('state').innerHTML='<span class="err">'+err.message+'</span>';$('go').disabled=false;clearInterval(timer);return}
 const poll=setInterval(async()=>{
  const s=await (await fetch('/status/'+j.id)).json();
  $('log').textContent=s.log;$('log').scrollTop=1e9;steps(s.log);
  $('state').textContent={queued:'En file d\'attente…',running:'Traitement en cours…'}[s.state]||'';
  if(s.state=='done'||s.state=='error'){
   clearInterval(poll);clearInterval(timer);$('go').disabled=false;
   if(s.state=='error'){$('state').innerHTML='<span class="err">Erreur — voir le journal</span>'}
   else{$('state').textContent='Terminé';[1,2,3,4,5].slice(0,fd.getAll('targets').length?5:3).forEach(i=>$('s'+i).className='step done')}
   $('files').innerHTML=s.files.map(n=>`<a href="/download/${j.id}/${n}">${n}</a>`).join('');
  }
 },1500);
};
</script></body></html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
