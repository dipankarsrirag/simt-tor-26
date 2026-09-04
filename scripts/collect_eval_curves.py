# Collect BLEU/AL for all four systems from the eval JSONs (2B from the Hub,
# the rest local) into one facets file for plotting.
import json, os
from huggingface_hub import HfApi, hf_hub_download

LAT = ["low", "low-medium", "medium", "medium-high", "high"]
CELLS = [("WMT15 de-en", "wmt15", "de-en"), ("WMT22 de-en", "wmt22", "de-en"),
         ("WMT22 en-de", "wmt22", "en-de"), ("WMT22 ru-en", "wmt22", "ru-en"),
         ("WMT22 en-ru", "wmt22", "en-ru"), ("IWSLT17 de-en", "iwslt17", "de-en"),
         ("IWSLT17 en-de", "iwslt17", "en-de"), ("IWSLT17 ar-en", "iwslt17", "ar-en"),
         ("IWSLT17 en-ar", "iwslt17", "en-ar"), ("IWSLT15 vi-en", "iwslt15", "vi-en"),
         ("IWSLT15 en-vi", "iwslt15", "en-vi")]
TAGS = ["gemma_4b_curated", "gemma_4b_from_2b_annot", "east_8b_from_2b_annot"]

api = HfApi()
b2 = {}
for f in api.list_repo_files("unswnlporg/tor-simt-gemma-2b-curated"):
    if "htgt" in f and ("iwslt" in f or "wmt" in f):
        try:
            d = json.load(open(hf_hub_download("unswnlporg/tor-simt-gemma-2b-curated", f)))
        except Exception:
            continue
        n = f.split("/")[-1].replace("_stream_v6bv2balv3htgt_checkargmax", "").rsplit("_n", 1)[0]
        n = n.replace("_low_medium_", "_low-medium_").replace("_medium_high_", "_medium-high_")
        b2[n] = [d["al_mean"], d["bleu"]]

def local(tag, ts, lat, pair):
    p = f"results/eval/{tag}/{ts}_stream_{tag}_check_argmax_{lat}_{pair}.json"
    d = json.load(open(p))
    return [d["al_mean"], d["bleu"]]

out, missing = [], 0
for disp, ts, pair in CELLS:
    series = []
    for tag in [None] + TAGS:
        pts = []
        for lat in LAT:
            try:
                pts.append(b2[f"{ts}_{lat}_{pair}"] if tag is None else local(tag, ts, lat, pair))
            except Exception:
                missing += 1
        series.append(pts)
    out.append([disp] + series)
json.dump(out, open("/srv/scratch/z5531827/facets4_fresh.json", "w"), ensure_ascii=False)
print("facets:", len(out), "missing cells:", missing)
print("counts:", [[len(s) for s in row[1:]] for row in out][:3])
