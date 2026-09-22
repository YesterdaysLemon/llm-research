import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
names=subprocess.check_output(['git','ls-files','experiments/E006-heart-wall-agent-sampling','experiments/E009-interiority-ablation/cloud','experiments/E010-kv-policy-compression'],cwd=root,text=True).splitlines()
files={name:{'bytes':(root/name).stat().st_size,'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest()} for name in names if (root/name).is_file()}
(root/'public-release-manifest.json').write_text(json.dumps({'edition':'Lyrebird 2026-09-22','files':files,'note':'Hashes identify this public source edition. Original private provenance remains in the project archive; public metadata redactions are explained in PUBLICATION.md.'},indent=2),encoding='utf-8',newline='\n')
print('Public evidence manifest:',len(files),'files')
