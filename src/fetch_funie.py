import io
import urllib.request
import zipfile
from pathlib import Path

SHA = "8f934c834c94e007b00866186b9ee624dc2b7b69"
URL = f"https://github.com/xahidbuffon/FUnIE-GAN/archive/{SHA}.zip"
WANT = ("PyTorch/", "TF-Keras/", "Evaluation/", "data/test/", "LICENSE", "README.md")
OUT = Path("third_party/FUnIE-GAN")

# Extract only chosen paths, since the repo has data/con.jpg and Windows
# refuses to create files named after reserved device names like CON.
data = urllib.request.urlopen(URL).read()
with zipfile.ZipFile(io.BytesIO(data)) as z:
    for member in z.namelist():
        rel = member.split("/", 1)[1] if "/" in member else ""
        if not rel or rel.endswith("/") or not rel.startswith(WANT):
            continue
        target = OUT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(z.read(member))
        print(rel)