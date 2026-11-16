# Table 3: UCI real-data sanity check

`run_uci_table3.py` is the single canonical UCI runner retained from the
supplied `uci_related.zip`.  The duplicate local/self-contained/screening
variants are intentionally omitted.

Its default dataset list is restricted to the three datasets reported in the
paper:

- Auto MPG
- Parkinsons Motor UPDRS
- Air Quality NOx

and the default seed list is 0--9, matching the ten random splits in Table 3.

Run:

```bash
python run_uci_table3.py --outdir results
```

The script downloads/loads UCI data, preprocesses them, and compares augmented
RFF-DGP, regular RFF-DGP, exact RBF GP and linear regression.  Internet access
may be required for dataset acquisition.

The manuscript reports only the final Table-3 comparison, not every
implementation hyperparameter.  The retained defaults are those of the
supplied full-baseline candidate runner (eta 1.0,0.6,0.3; 1000 epochs; 96/192
RFF features; max training size 800).
