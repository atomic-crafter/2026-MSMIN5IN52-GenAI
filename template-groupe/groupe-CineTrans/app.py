"""Interface web minimale pour extractsrt.py (Flask).

docker compose up --build   puis   http://localhost:8000
Fonctionne à l'identique sur Windows, macOS et Linux : seul un navigateur est nécessaire.
"""
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"

LANGUAGES = {
    "fr": "Français", "en": "Anglais", "es": "Espagnol", "de": "Allemand",
    "it": "Italien", "pt": "Portugais", "nl": "Néerlandais", "ru": "Russe",
    "ar": "Arabe", "zh": "Chinois", "ja": "Japonais", "ko": "Coréen",
}

app = Flask(__name__)
jobs = {}  # id -> {"state": running|done|error, "log": str, "files": [..]}
run_lock = threading.Lock()  # un seul traitement à la fois (modèles lourds)


def run_job(job_id, video, out_dir, args):
    job = jobs[job_id]
    with run_lock:
        job["state"] = "running"
        cmd = [sys.executable, "-u", "extractsrt.py", "--video", str(video),
               "--out-dir", str(out_dir), *args]
        proc = subprocess.Popen(cmd, cwd=BASE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            job["log"] += line
        code = proc.wait()
    job["files"] = sorted(p.name for p in out_dir.iterdir() if p.suffix in {".srt", ".json", ".wav"})
    job["state"] = "done" if code == 0 else "error"


@app.post("/run")
def run():
    file = request.files.get("video")
    if not file or not file.filename:
        return jsonify(error="Aucune vidéo envoyée"), 400
    source = request.form.get("source_lang", "auto")
    if source != "auto" and source not in LANGUAGES:
        return jsonify(error="Langue source invalide"), 400
    targets = [t for t in request.form.getlist("targets") if t in LANGUAGES]

    job_id = uuid.uuid4().hex[:8]
    name = secure_filename(file.filename) or "video.mp4"
    (UPLOADS / job_id).mkdir(parents=True, exist_ok=True)
    out_dir = OUTPUTS / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    video = UPLOADS / job_id / name
    file.save(video)

    args = ["--source-lang", source, "--target-langs", ",".join(targets)]
    if request.form.get("multilingual"):
        args.append("--multilingual")
    jobs[job_id] = {"state": "queued", "log": "", "files": []}
    threading.Thread(target=run_job, args=(job_id, video, out_dir, args), daemon=True).start()
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
    options = "".join(f'<option value="{c}">{n} ({c})</option>' for c, n in LANGUAGES.items())
    checks = "".join(
        f'<label><input type="checkbox" name="targets" value="{c}"> {n}</label>'
        for c, n in LANGUAGES.items()
    )
    return PAGE.replace("{{OPTIONS}}", options).replace("{{CHECKS}}", checks)


PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CineTrans</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#222}
 fieldset{margin:1rem 0;border:1px solid #ccc;border-radius:6px}
 .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.3rem}
 button{padding:.6rem 1.2rem;font-size:1rem;cursor:pointer}
 pre{background:#111;color:#ddd;padding:1rem;border-radius:6px;max-height:320px;overflow:auto;white-space:pre-wrap}
</style></head><body>
<h1>CineTrans — vidéo → SRT</h1>
<form id="f">
 <p><label>Vidéo : <input type="file" name="video" accept="video/*" required></label></p>
 <p><label>Langue de la vidéo :
  <select name="source_lang" id="src"><option value="auto">Auto-détection</option>{{OPTIONS}}</select></label></p>
 <p><label><input type="checkbox" name="multilingual" id="multi" value="1">
  Plusieurs langues dans la vidéo (détection par segment)</label></p>
 <fieldset><legend>Traduire vers</legend><div class="grid">{{CHECKS}}</div></fieldset>
 <button id="go">Lancer</button>
</form>
<h3>Résultats</h3><div id="files">—</div>
<pre id="log"></pre>
<script>
const f=document.getElementById('f'),log=document.getElementById('log'),files=document.getElementById('files'),go=document.getElementById('go');
document.getElementById('multi').onchange=e=>{const s=document.getElementById('src');if(e.target.checked)s.value='auto';s.disabled=e.target.checked};
f.onsubmit=async e=>{
 e.preventDefault();
 const fd=new FormData(f); if(document.getElementById('src').disabled) fd.set('source_lang','auto');
 go.disabled=true; log.textContent='Envoi de la vidéo…'; files.textContent='—';
 const r=await fetch('/run',{method:'POST',body:fd}); const j=await r.json();
 if(!r.ok){log.textContent=j.error;go.disabled=false;return}
 const t=setInterval(async()=>{
  const s=await (await fetch('/status/'+j.id)).json();
  log.textContent=s.log; log.scrollTop=log.scrollHeight;
  if(s.state=='done'||s.state=='error'){
   clearInterval(t); go.disabled=false;
   if(s.state=='error') log.textContent+='\\n== ERREUR ==';
   files.innerHTML=s.files.map(n=>`<a href="/download/${j.id}/${n}">${n}</a>`).join(' · ')||'—';
  }
 },1500);
};
</script></body></html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
