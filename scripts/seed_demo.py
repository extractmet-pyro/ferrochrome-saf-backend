"""Create tables, seed role accounts, write demo data files (plant config, materials, tags, nominal scenario)."""
import csv
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.config import settings  # noqa: E402
from app.db.repositories import init_db  # noqa: E402
from app.domain import plant_config as pc  # noqa: E402
from app.domain.material_streams import DEFAULT_RECIPE, default_materials  # noqa: E402
from app.integrations.synthetic_gateway import tags_csv  # noqa: E402

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "data", "demo")
init_db(settings.demo_password)
cfg = pc.build_default_config(45.0, "closed")
with open(os.path.join(D, "plant_config.yaml"), "w") as fh:
    yaml.safe_dump(json.loads(json.dumps(cfg)), fh, sort_keys=False)
mats = default_materials()
keys = sorted({k for m in mats.values() for k in m.assay_dry_wt_pct})
with open(os.path.join(D, "raw_materials.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["key", "name", "category", "moisture_wet_wt_pct", "price_inr_per_t", "d50_mm", "fines_pct", "bulk_density_t_m3",
                "prereduction_fe", "prereduction_cr"] + [f"{k}_dry_wt_pct" for k in keys])
    for m in mats.values():
        w.writerow([m.key, m.name, m.category, m.moisture_wet_wt_pct, m.price_inr_per_t, m.d50_mm, m.fines_pct, m.bulk_density_t_m3,
                    m.prereduction_fe, m.prereduction_cr] + [m.assay_dry_wt_pct.get(k, 0.0) for k in keys])
with open(os.path.join(D, "tags.csv"), "w") as fh:
    fh.write(tags_csv())
with open(os.path.join(D, "recipe_nominal.json"), "w") as fh:
    json.dump(DEFAULT_RECIPE, fh, indent=1)
print("seeded users and wrote", os.listdir(D))
