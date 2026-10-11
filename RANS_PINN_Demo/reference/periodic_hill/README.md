# Periodic-hill reference data

The 2-D periodic-hill experiment uses the **independent** LES reference from the
ERCOFTAC Classic Collection:

- Case: **081**, “Flow over Periodic Hills”
- Authors: L. Temmerman and M. A. Leschziner (2001)
- Type: wall-resolving LES with a dynamic subgrid-scale model
- Source page: http://cfd.mace.manchester.ac.uk/ercoftac/doku.php?id=cases:case081
- Archive: `perh-allfiles.zip`
- SHA-256: `61ca8053a187c67d914f1ab951a003136bd92c4b61e7de7b359d18002978030b`

The archive is **not** committed to this MIT-licensed repository.  Download it
with the project helper:

```bash
python -m RANS_PINN_Demo.fetch_periodic_hill_reference --destination RANS_PINN_Demo/reference/periodic_hill
```

The helper writes `RANS_PINN_Demo/reference/periodic_hill/perh-allfiles.zip` and
extracts `data-001.dat` … `data-010.dat` plus `geom.dat` into `extracted/`.

Data columns are `Y/h`, `U/U_b`, `V/U_b`, `uu/U_b^2`, `vv/U_b^2`, `uv/U_b^2`,
`k/U_b^2`.  The reference is used only for sparse observations and held-out
evaluation; it is never passed to the RANS solver or to the closure model as an
input field.

License note: the ERCOFTAC Classic Collection data is distributed under
CC BY-NC-SA 4.0.  Keep the attribution above with any redistributed copy.
