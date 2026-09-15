"""EXP_6 server watchdog (python): sequence LL then LSDA-Long on GPU0, restart on death.

Runs on the generation server. Resumable jobs; logs a status line every 180s.
"""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/science/wx/pry/MMDIT")
EXP = ROOT / "experiments/2026_9_15_EXP_6_KNOW_INJECT"
OUTLL = ROOT / "data/KNOW_INJECT/2026_9_15_EXP_6/LL"
OUTLS = ROOT / "data/KNOW_INJECT/2026_9_15_EXP_6/LSDA_Long"
PY = "/science/wx/pry/.venv/bin/python"
MD = "/science/wx/pry/models"
LOG = EXP / "watchdog.log"
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUDA_VISIBLE_DEVICES": "0"}


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg + "\n")


def running(pat: str) -> bool:
    out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.strip()
    return bool(out)


def spawn(cmd: list, logname: str) -> None:
    fh = open(EXP / logname, "a", encoding="utf-8")
    subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True, env=ENV)


def main() -> None:
    log("watchdog(py) start")
    while True:
        nll = len(list(OUTLL.glob("*.png")))
        nls = len(list(OUTLS.glob("runs/*/lsda.png")))
        if nll < 300:
            if not running("gen_ll.py"):
                log(f"restart LL ({nll}/300)")
                spawn([PY, "code/culture_comp/gen_ll.py", "--jobs", str(EXP / "jobs/ll_jobs.json"),
                       "--out", str(OUTLL), "--model-dir", f"{MD}/stable-diffusion-3.5-large"], "ll.log")
        else:
            if nls < 250 and not running("run_census_lsda.py"):
                log(f"start LSDA-Long on GPU0 ({nls}/250)")
                time.sleep(20)
                spawn([PY, "code/lsda/run_census_lsda.py", "--jobs", str(EXP / "jobs/lsdalong_jobs.json"),
                       "--images-dir", str(ROOT / "experiments/2026_9_12_EXP_3_KA_ME/census/images"),
                       "--out", str(OUTLS), "--model-dir", f"{MD}/stable-diffusion-3.5-large",
                       "--sam-model-dir", f"{MD}/sam-vit-base", "--rect-pad", "16"], "lsdalong.log")
        log(f"LL={nll}/300 LSDA={nls}/250")
        time.sleep(180)


if __name__ == "__main__":
    main()
