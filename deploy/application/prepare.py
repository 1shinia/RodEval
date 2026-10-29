"""Create a clean source context and dependency lock from the validated environment."""
import importlib.metadata as md
import json
import shutil
import subprocess
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('/data/evalperf-build')
CTX = OUT / 'context'
CTX.mkdir(parents=True, exist_ok=True)
roots = []
for name in ['framework', 'service', 'perf', 'rag', 'sandbox', 'aigc']:
    for line in (ROOT / f'requirements/{name}.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            roots.append(Requirement(line))
for name in ['mutagen', 'websockets', 'jiwer', 'psutil', 'tiktoken', 'swebench', 'harbor', 'langdetect', 'emoji', 'syllapy', 'apted', 'distance', 'Polygon3', 'zss', 'python-Levenshtein', 'lxml', 'beautifulsoup4', 'shapely', 'seaborn', 'pythainlp']:
    roots.append(Requirement(name))
queue = roots[:]
extras_seen = {}
versions = {}
missing = set()
while queue:
    req = queue.pop()
    key = canonicalize_name(req.name)
    extras = set(req.extras) | {''}
    if key in extras_seen and extras <= extras_seen[key]:
        continue
    extras_seen.setdefault(key, set()).update(extras)
    try:
        dist = md.distribution(req.name)
    except md.PackageNotFoundError:
        missing.add(req.name)
        continue
    versions[key] = dist.version
    for text in dist.requires or []:
        child = Requirement(text)
        if child.marker and not any(child.marker.evaluate({'extra': e}) for e in extras_seen[key]):
            continue
        if canonicalize_name(child.name).startswith(('nvidia-', 'triton')):
            continue
        queue.append(child)
versions['torch'] = md.version('torch').split('+')[0] + '+cpu'
# Service enables CORS opportunistically; include the declared runtime dependency.
versions['flask-cors'] = '6.0.2'
# The source environment has an existing protobuf conflict (google-api-core/grpcio-status).
# Use the shared compatible range, without modifying the working Conda environment.
if 'protobuf' in versions:
    versions['protobuf'] = '5.29.6'
(CTX / 'requirements.lock').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(versions.items()))+'\n')
(OUT / 'dependency-baseline.json').write_text(json.dumps({'versions': versions, 'missing_in_hermes': sorted(missing)}, indent=2))
files = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
for name in files:
    if not name or not (name.startswith(('evalscope/', 'requirements/', 'leaderboard_data/')) or name in ['pyproject.toml','setup.py','README.md','LICENSE','MANIFEST.in']):
        continue
    if any(part in {'node_modules', 'outputs', '__pycache__', '.env', '.git'} for part in Path(name).parts):
        continue
    src = ROOT / name
    if src.is_file():
        dst=CTX / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src,dst)
for name in ['Dockerfile','entrypoint.py']:
    shutil.copy2(ROOT / 'deploy/application' / name, CTX / name)
print(json.dumps({'context':str(CTX),'locked_packages':len(versions),'missing_in_hermes':sorted(missing)},ensure_ascii=False))
