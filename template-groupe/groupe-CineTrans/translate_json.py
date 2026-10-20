#!/usr/bin/env python3
"""
Traduction d'un JSON de segments pour le projet de doublage multi-locuteurs.

Entrée  : liste de segments {"start", "end", "text", "speaker", "language", "emotion"}
Sortie  : MÊME structure ; seuls "text" et "language" changent.

Le LLM ne voit jamais le JSON complet : on lui envoie seulement (id, texte, locuteur,
émotion, longueur max) et on réinjecte ses réponses dans le JSON d'origine. Les champs
start / end / speaker / emotion ne peuvent donc pas être modifiés par erreur.

Configuration : bloc CONFIGURATION en haut du fichier, ou variables d'environnement, ou options :
    API_KEY       clé fournie par le professeur
    API_BASE_URL  URL de l'API compatible OpenAI (ex. https://.../v1)
    LLM_MODEL     nom exact du modèle (ex. celui de Qwen3.8 27B dans la liste du prof)

Sortie : sans nom de sortie, le fichier s'appelle segment_<langue>.json (ex. segment_fr.json),
dans le dossier du fichier d'entrée (motif modifiable : OUTPUT_PATTERN). Un rapport
segment_<langue>.report.json est écrit à côté (longueurs et accélération nécessaire par segment).

Longueur : la fidélité au sens prime. Une traduction un peu plus longue que l'original est
gardée telle quelle (la vidéo sera accélérée) ; --shorten force une 2e passe de raccourcissement.
"""
import argparse
import contextlib
import copy
import json
import os
import re
import sys
import threading
import time
import urllib.request
import requests

# =============================== CONFIGURATION ===============================
API_KEY = os.getenv("OPENROUTER_API_KEY")
API_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = "deepseek/deepseek-v4.1-flash"
TARGET_LANG = "fr"
OUTPUT_PATTERN = "segment_{lang}.json"
# =============================================================================

LANG_NAMES = {
    "fr": "français", "en": "anglais", "es": "espagnol", "de": "allemand",
    "it": "italien", "pt": "portugais", "nl": "néerlandais", "ru": "russe",
    "ja": "japonais", "zh": "chinois", "ko": "coréen", "ar": "arabe",
    "hi": "hindi", "tr": "turc", "pl": "polonais",
}
REQUIRED_KEYS = {"start", "end", "text", "speaker", "language", "emotion"}
MIN_CHARS = 12
CHANGEABLE_KEYS = {"text", "language"}

SYSTEM_PROMPT = """Tu es traducteur professionnel spécialisé dans le doublage de dialogues.
Tu reçois des segments de parole extraits d'une vidéo, au format JSON. Pour chaque segment :
- traduis "text" en __TGT__ de façon naturelle et orale, comme quelqu'un qui parle ;
- respecte le registre, le ton et l'émotion indiqués dans "emotion" ;
- reste fidèle au sens, sans ajouter ni omettre d'information importante ;
- la FIDÉLITÉ AU SENS passe avant tout : ne supprime et ne résume jamais une idée pour gagner \
de la place ; la vidéo sera simplement accélérée si la traduction est un peu plus longue ;
- "longueur_indicative_car" donne un ordre de grandeur de la longueur souhaitée ("duree_s" est \
la durée de parole en secondes) : vise cette longueur quand une formulation naturelle le permet, \
mais ce n'est PAS une limite stricte, un léger dépassement est acceptable ;
- pas d'émojis, pas de didascalies, pas de notes, pas de guillemets ajoutés ;
- les noms propres ne se traduisent pas ;
- un même locuteur ("speaker") garde le même registre (tutoiement/vouvoiement, vocabulaire) ;
- les segments sont dans l'ordre de la conversation ; "contexte_precedent" sert uniquement \
à comprendre la suite, ne le traduis pas.
Réponds UNIQUEMENT par un tableau JSON : [{"id": <id>, "text": "<traduction>"}, ...] \
avec exactement les mêmes id, sans aucun commentaire."""

SHORTEN_PROMPT = """Tu es traducteur professionnel spécialisé dans le doublage, vers le __TGT__.
Les traductions ("current") des segments suivants sont trop longues pour le temps de parole.
Pour chaque segment, réécris "current" plus court (au plus "max_chars" caractères), en gardant \
le sens essentiel, un ton oral naturel et l'émotion indiquée. "text" est le texte source, \
donné pour référence.
Réponds UNIQUEMENT par un tableau JSON : [{"id": <id>, "text": "<version courte>"}, ...] \
avec exactement les mêmes id, sans aucun commentaire."""

def to_seconds(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    try:
        total = 0.0
        for part in str(v).replace(",", ".").split(":"):
            total = total * 60 + float(part)
        return total
    except ValueError:
        return None

def extract_json(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    a, b = raw.find("["), raw.rfind("]")
    if a == -1 or b == -1 or b < a:
        raise ValueError("aucun tableau JSON dans la réponse")
    return json.loads(raw[a:b + 1])

class FatalAPIError(Exception):
    pass
class ConnectionProblem(Exception):
    pass
class RateLimited(Exception):
    def __init__(self, msg, wait=None):
        super().__init__(msg)
        self.wait = wait

FATAL_HINTS = (
    "Pistes :\n"
    "  1) python translate_json.py --check   -> teste la connexion\n"
    "  2) vérifie API_BASE_URL, la clé et l'identifiant du modèle\n"
    "  3) essaie --no-proxy (proxy d'école, antivirus, VPN)\n"
    "  4) essaie un autre réseau"
)

_session = None

def get_session(cfg):
    global _session
    if _session is None:
        _session = requests.Session()
        if cfg.get("no_proxy"):
            _session.trust_env = False
    return _session

def call_llm(cfg, system, user):
    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    try:
        r = get_session(cfg).post(
            url,
            headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
            json={"model": cfg["model"], "temperature": 0.2,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=180,
        )
    except requests.exceptions.RequestException as e:
        raise ConnectionProblem(f"{type(e).__name__} en appelant {url} : {e}") from e
    if r.status_code in (400, 401, 403, 404):
        raise FatalAPIError(f"HTTP {r.status_code} sur {url} : {r.text[:300]}")
    if r.status_code == 429:
        try:
            wait = float(r.headers.get("Retry-After", ""))
        except ValueError:
            wait = None
        raise RateLimited(f"HTTP 429 : {r.text[:200]}", wait)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} : {r.text[:200]}")
    data = r.json()
    if "choices" not in data:
        raise RuntimeError(f"réponse inattendue : {str(data)[:300]}")
    content = data["choices"][0]["message"].get("content")
    if not content:
        raise RuntimeError("réponse vide du modèle")
    return content

def say(cfg, msg="", detail=False):
    if detail and cfg.get("quiet"):
        return
    print(msg, flush=True)

def fmt_time(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"

@contextlib.contextmanager
def heartbeat(cfg, label, interval=10):
    stop = threading.Event()
    t0 = time.monotonic()
    def run():
        while not stop.wait(interval):
            say(cfg, f"    ... {label} ({int(time.monotonic() - t0)} s)", detail=True)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()

def ask(cfg, items, context, mode):
    if cfg["dry_run"]:
        if mode == "translate":
            return {it["id"]: f"[{cfg['tgt']}] {it['text']}" for it in items}
        return {it["id"]: it["current"] for it in items}
    template = SYSTEM_PROMPT if mode == "translate" else SHORTEN_PROMPT
    system = template.replace("__TGT__", cfg["tgt_name"])
    if cfg.get("context"):
        system += f"\nContexte de la vidéo : {cfg['context']}"
    payload = {"segments": items}
    if mode == "translate":
        payload["contexte_precedent"] = context
    t0 = time.monotonic()
    with heartbeat(cfg, "toujours en attente de la réponse du modèle"):
        raw = call_llm(cfg, system, json.dumps(payload, ensure_ascii=False))
    say(cfg, f"    réponse reçue en {time.monotonic() - t0:.1f} s")
    data = extract_json(raw)
    out = {}
    for d in data:
        out[int(d["id"])] = str(d["text"]).strip()
    expected = {it["id"] for it in items}
    if set(out) != expected:
        raise ValueError(f"ids reçus {sorted(out)} != attendus {sorted(expected)}")
    if any(not t for t in out.values()):
        raise ValueError("une traduction est vide")
    return out

def ask_with_retry(cfg, items, context, mode, retries=3):
    net_failures = rate_failures = 0
    for attempt in range(retries):
        try:
            return ask(cfg, items, context, mode)
        except FatalAPIError:
            raise
        except RateLimited as e:
            rate_failures += 1
            pause = min(60, e.wait if e.wait else 5 * 3 ** attempt)
            print(f"  ! essai {attempt + 1}/{retries} : limite de débit, nouvelle tentative dans {pause:.0f} s ({e})", file=sys.stderr, flush=True)
            time.sleep(pause)
        except ConnectionProblem as e:
            net_failures += 1
            print(f"  ! essai {attempt + 1}/{retries} échoué (réseau) : {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  ! essai {attempt + 1}/{retries} échoué : {e}", file=sys.stderr)
            time.sleep(2 ** attempt if not cfg["dry_run"] else 0)
    if net_failures == retries:
        raise FatalAPIError(f"connexion impossible après {retries} essais")
    if rate_failures == retries:
        raise FatalAPIError(f"limite de débit (HTTP 429) atteinte {retries} fois de suite")
    if len(items) > 1:
        mid = len(items) // 2
        res = ask_with_retry(cfg, items[:mid], context, mode, retries)
        res.update(ask_with_retry(cfg, items[mid:], context, mode, retries))
        return res
    return {}

def duration(seg):
    s, e = to_seconds(seg["start"]), to_seconds(seg["end"])
    return (e - s) if (s is not None and e is not None and e > s) else None

def translate_segments(segs, cfg):
    idx = [i for i, s in enumerate(segs) if str(s["text"]).strip()]
    results = {}
    bs = cfg["batch_size"]
    t_start = time.monotonic()

    def limit_for(i):
        lim = int(len(segs[i]["text"]) * cfg["max_ratio"])
        dur = duration(segs[i])
        if dur:
            lim = min(lim, int(dur * cfg["max_cps"]))
        return max(MIN_CHARS, lim)

    def show_pair(i, text):
        s = segs[i]
        dur = duration(s)
        meta = f"{s['speaker']} | {s['emotion']}" + (f" | {dur:.1f} s" if dur else "")
        flag = (f"   (plus long que le créneau : {len(text)} car. pour ~{limit_for(i)} -> à accélérer)"
                if len(text) > limit_for(i) else "")
        say(cfg, f"  #{i} [{meta}]", detail=True)
        say(cfg, f"      {str(s['language']).upper():<3}: {s['text']}", detail=True)
        say(cfg, f"      {cfg['tgt'].upper():<3}: {text}{flag}", detail=True)

    n_batches = -(-len(idx) // bs)
    say(cfg, f"{len(idx)} segment(s) à traduire vers '{cfg['tgt']}' en {n_batches} lot(s) de {bs} maximum")
    for b in range(0, len(idx), bs):
        chunk = idx[b:b + bs]
        num = b // bs + 1
        items = [{
            "id": i, "speaker": segs[i]["speaker"], "emotion": segs[i]["emotion"],
            "text": segs[i]["text"],
            "longueur_indicative_car": limit_for(i),
            "duree_s": round(duration(segs[i]), 2) if duration(segs[i]) else None,
        } for i in chunk]
        prev = [j for j in idx if j < chunk[0]][-3:]
        context = [{"speaker": segs[j]["speaker"], "text": segs[j]["text"]} for j in prev]
        say(cfg, f"\n[Lot {num}/{n_batches}] {len(items)} segments -> envoi au modèle...")
        got = ask_with_retry(cfg, items, context, "translate")
        results.update(got)
        for i in chunk:
            if i in got:
                show_pair(i, got[i])
            else:
                say(cfg, f"  #{i} NON TRADUIT (échec du modèle), texte source conservé")
        done = min(b + bs, len(idx))
        elapsed = time.monotonic() - t_start
        eta = elapsed / num * (n_batches - num)
        say(cfg, f"  Progression : {done}/{len(idx)} segments ({100 * done // len(idx)} %), "
             f"écoulé {fmt_time(elapsed)}, reste environ {fmt_time(eta)}")

    over = [i for i in results if len(results[i]) > limit_for(i)]
    if cfg.get("shorten"):
        too_long = over
    else:
        too_long = []
        if over:
            say(cfg, f"\n{len(over)} segment(s) dépassent un peu leur créneau : traduction gardée (vidéo accélérée). Segments : {over}")

    if too_long:
        say(cfg, f"\n[Raccourcissement] {len(too_long)} segment(s) trop long(s) -> envoi au modèle...")
    for b in range(0, len(too_long), bs):
        chunk = too_long[b:b + bs]
        items = [{"id": i, "speaker": segs[i]["speaker"], "emotion": segs[i]["emotion"],
                  "text": segs[i]["text"], "current": results[i],
                  "max_chars": limit_for(i)} for i in chunk]
        short = ask_with_retry(cfg, items, [], "shorten")
        for i, t in short.items():
            if t and len(t) < len(results[i]):
                say(cfg, f"  #{i} raccourci : {len(results[i])} -> {len(t)} car. (max {limit_for(i)})", detail=True)
                say(cfg, f"      {cfg['tgt'].upper():<3}: {t}", detail=True)
                results[i] = t
            else:
                say(cfg, f"  #{i} déjà aussi court que possible", detail=True)
    say(cfg, f"\nTraduction terminée en {fmt_time(time.monotonic() - t_start)}")
    return results

def check_integrity(original, final):
    assert len(original) == len(final), "nombre de segments différent"
    for n, (a, b) in enumerate(zip(original, final)):
        assert set(a) == set(b), f"segment {n} : clés différentes"
        for k in a:
            if k not in CHANGEABLE_KEYS and a[k] != b[k]:
                raise AssertionError(f"segment {n} : le champ '{k}' a été modifié")

def build_report(original, final, results, cfg):
    rows = []
    for i, (a, b) in enumerate(zip(original, final)):
        dur = duration(a)
        src_c, tgt_c = len(str(a["text"])), len(str(b["text"]))
        translated = i in results
        rows.append({
            "id": i, "speaker": a["speaker"], "translated": translated,
            "duration_s": round(dur, 2) if dur else None,
            "src_chars": src_c, "tgt_chars": tgt_c,
            "ratio": round(tgt_c / src_c, 2) if src_c else None,
            "cps": round(tgt_c / dur, 1) if dur else None,
            "too_dense": bool(dur and tgt_c / dur > cfg["max_cps"]),
            "speedup": round(max(1.0, tgt_c / (dur * cfg["max_cps"])), 2) if dur else None,
        })
    return rows

def check_connection(cfg):
    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    key = cfg["api_key"]
    proxies = {} if cfg.get("no_proxy") else urllib.request.getproxies()
    proxies = {k: re.sub(r"//[^@/]*@", "//***@", v) for k, v in proxies.items() if k in ("http", "https", "all")}
    print(f"URL appelée   : {url}")
    print(f"Modèle        : {cfg['model']}")
    print(f"Clé           : {key[:4]}... ({len(key)} caractères)")
    print(f"Proxy détecté : {proxies or 'aucun'}" + (" (ignoré : --no-proxy)" if cfg.get("no_proxy") else ""))
    t0 = time.monotonic()
    try:
        txt = call_llm(cfg, "Tu es un assistant concis.", "Réponds uniquement par le mot : ok")
    except Exception as e:
        print(f"\nÉCHEC : {e}\n\n{FATAL_HINTS}")
        return 1
    dt = time.monotonic() - t0
    print(f"\nConnexion OK en {dt:.1f} s. Réponse du modèle : {txt.strip()[:80]}")
    if dt > 8:
        print("Remarque : c'est déjà lent. Modèle « à raisonnement » ou file d'attente ?")
    return 0

def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--target-lang", default=None, help="code langue cible, ex. fr, en, es")
    ap.add_argument("--model", default=os.getenv("LLM_MODEL") or LLM_MODEL)
    ap.add_argument("--base-url", default=os.getenv("API_BASE_URL") or API_BASE_URL)
    ap.add_argument("--api-key", default=os.getenv("API_KEY") or API_KEY)
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--max-ratio", type=float, default=1.25)
    ap.add_argument("--context", default=None)
    ap.add_argument("--max-cps", type=float, default=17.0)
    ap.add_argument("--report", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-proxy", action="store_true")
    ap.add_argument("--shorten", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.dry_run and not (args.model and args.base_url and args.api_key):
        sys.exit("Il manque API_KEY, API_BASE_URL ou LLM_MODEL")
    if args.check:
        sys.exit(check_connection({"model": args.model, "base_url": args.base_url,
                                   "api_key": args.api_key, "no_proxy": args.no_proxy}))
    if not args.input:
        ap.error("le fichier d'entrée est obligatoire (sauf avec --check)")

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    wrapper = None
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        wrapper, segs_in = data, data["segments"]
    elif isinstance(data, list):
        segs_in = data
    else:
        sys.exit("Format inattendu : il faut une liste de segments (ou {'segments': [...]}).")
    for n, s in enumerate(segs_in):
        missing = REQUIRED_KEYS - set(s)
        if missing:
            sys.exit(f"Segment {n} : clés manquantes {sorted(missing)}")

    if args.target_lang:
        targets, origin = [args.target_lang.lower()], "option --target-lang"
    elif wrapper is not None and wrapper.get("target_languages"):
        targets = [str(l).lower() for l in wrapper["target_languages"]]
        origin = "champ target_languages du JSON d'entrée"
    else:
        targets, origin = [TARGET_LANG.lower()], "constante TARGET_LANG"
    targets = list(dict.fromkeys(targets))
    print(f"Langue(s) cible(s) : {', '.join(targets)} (source : {origin})")

    for tgt in targets:
        if args.output:
            base, ext = os.path.splitext(args.output)
            out_path = args.output if len(targets) == 1 else f"{base}.{tgt}{ext or '.json'}"
        else:
            out_dir = os.path.dirname(os.path.abspath(args.input))
            out_path = os.path.join(out_dir, OUTPUT_PATTERN.format(lang=tgt))
        if os.path.abspath(out_path) == os.path.abspath(args.input):
            sys.exit(f"Le fichier de sortie ({out_path}) est le même que l'entrée")
        cfg = {
            "tgt": tgt, "tgt_name": LANG_NAMES.get(tgt, tgt), "model": args.model,
            "base_url": args.base_url, "api_key": args.api_key, "dry_run": args.dry_run,
            "batch_size": args.batch_size, "max_ratio": args.max_ratio, "max_cps": args.max_cps,
            "context": args.context, "no_proxy": args.no_proxy, "quiet": args.quiet,
            "shorten": args.shorten,
        }
        segs_out = copy.deepcopy(segs_in)
        results = translate_segments(segs_out, cfg)
        for i, t in results.items():
            segs_out[i]["text"] = t
            segs_out[i]["language"] = tgt
        check_integrity(segs_in, segs_out)

        out_data = segs_out
        if wrapper is not None:
            out_data = {**wrapper, "segments": segs_out, "target_languages": [tgt]}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

        report = build_report(segs_in, segs_out, results, cfg)
        rpath = (args.report if (args.report and len(targets) == 1)
                 else os.path.splitext(out_path)[0] + ".report.json")
        with open(rpath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        n_tr = sum(r["translated"] for r in report)
        n_skip = len(report) - n_tr
        dense = [r["id"] for r in report if r["too_dense"]]
        ratios = [r["ratio"] for r in report if r["translated"] and r["ratio"]]
        speeds = [r["speedup"] for r in report if r["translated"] and r["speedup"]]
        print(f"\nTerminé [{tgt}] : {n_tr}/{len(report)} segments traduits -> {out_path}")
        if ratios:
            print(f"Ratio de longueur moyen : {sum(ratios) / len(ratios):.2f}")
        if speeds:
            print(f"Accélération nécessaire : jusqu'à x{max(speeds):.2f} (moyenne x{sum(speeds) / len(speeds):.2f}, {len(dense)} segment(s))")
        if n_skip:
            print(f"ATTENTION : {n_skip} segment(s) non traduit(s)")
        print(f"Rapport : {rpath}")

if __name__ == "__main__":
    try:
        main()
    except FatalAPIError as e:
        sys.exit(f"\nERREUR API : {e}\n\n{FATAL_HINTS}")
