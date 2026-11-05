"""Export the OpenAPI contract as YAML (docs/API_CONTRACT.yaml)."""
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "docs/API_CONTRACT.yaml"
with open(out, "w") as fh:
    yaml.safe_dump(json.loads(json.dumps(app.openapi())), fh, sort_keys=False)
print("wrote", out, len(app.openapi()["paths"]), "paths")
