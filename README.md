# Pedrógão Grande Reburn Susceptibility

Source code for the CASA0025 group project Earth Engine application that maps reburn susceptibility within the 2017 Pedrógão Grande fire footprint ahead of the 2026 fire season.

## Live application

https://exalted-country-485019-c8.projects.earthengine.app/view/pedrogao-reburn

## How to run the source

Open https://code.earthengine.google.com and sign in with any account that has Earth Engine access. Paste the contents of pedrogao_reburn_app.js into a new script, click Save, then click Run. The map and sidebar appear within 5 to 10 seconds. The six required assets are shared as Anyone can read so no extra access request is needed.

## Contribution map

Work was divided across the three categories defined in the Assessment Guidelines: Pre-processing, Analysis, and Visualization.

| Category | Member | Section ownership |
|----------|--------|-------------------|
| Pre-processing | Xiaozhu Yang | Config flags, ICNF perimeter filter and reburn label raster, Sentinel 2 dNBR with CloudScore plus pixel masking |
| Pre-processing | Chendie Wu | NBR linear fit 2018 to 2020, 2025 NDVI NDMI LST composites, Copernicus DEM elevation and aspect, 8 band predictor stack and Stage A export task |
| Analysis | Yifan Zhang | Spatial holdout train test split, Random Forest training and hyperparameter scan rationale, client side Mann Whitney U for AUC under the user memory ceiling, Youden J threshold scan and confusion matrix at J star, four partial dependence plots, 5 class quantile classification, município aggregation, Stage B and C and D exports, diag_v6 CSV encoded property design |
| Visualization | Wei Chen | LAYER_CONFIG palette and legend per layer, sidebar header with four fire badges and SE drop shadow focus, refreshLegend and refreshMap eight layer composition, click inspector with boundary check and NBR recovery trajectory chart |
| Visualization | Chenwei Fang | Top 5 priority concelhos card, treatment priority percentile filter, side by side split view via ui.SplitPanel and ui.Map.Linker, diagnostics panel rendered from diag_v6 in a single round trip, CSV export task of priority pixel centroids to Google Drive, USE_ASSET false fallback preview |

The five members each authored their assigned sections directly through their own GitHub accounts. Each pull request and merged commit on master shows the responsible contributor.

## Asset paths

The application reads from six Earth Engine assets owned by the deployment account.

```
projects/exalted-country-485019-c8/assets/icnf_2017_centre
projects/exalted-country-485019-c8/assets/icnf_reburns_2018_2025
projects/exalted-country-485019-c8/assets/predictors_stack_icnf_v5
projects/exalted-country-485019-c8/assets/reburn_prob_v5
projects/exalted-country-485019-c8/assets/diag_v6
projects/exalted-country-485019-c8/assets/muni_stats_v6
```

The first two are uploaded ICNF official perimeter data. The other four are precomputed outputs from a four stage caching pipeline that drops application cold start from roughly 120 seconds when computing live to roughly 10 seconds when reading from cache.

## Caching strategy

Three boolean flags at the top of the script control which precomputed asset is used.

`USE_ASSET` reads the eight band predictor stack from asset instead of building it from Sentinel 2 and Landsat composites at runtime. `USE_PROB_ASSET` reads the Random Forest probability surface from asset instead of running RF inference live across the burn footprint. `USE_DIAG_ASSET` reads every diagnostic value and the nine percentile breaks from a single one row FeatureCollection asset instead of running the diagnostic graph live.

All three flags are set to true in the deployed version. To rebuild any asset from scratch, flip the corresponding flag to false, save the script, run it once to register the export task in the Tasks tab, click Run on the task, wait for it to turn green, then flip the flag back to true.
