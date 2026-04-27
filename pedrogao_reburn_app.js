// Pedrógão Grande Reburn Susceptibility
// CASA0025 final project

var USE_ASSET = true; // stage A: flip to true after predictors_export_icnf finishes
var USE_PROB_ASSET = true; // stage C: flip to true after reburn_prob_export finishes (huge speedup)
var USE_DIAG_ASSET = true; // stage D: flip to true after diag_export_v6 finishes (skips runtime RF inference for diagnostics)
// Stage E: flip to true after class5_rgb_export + prob_rgb_export finish.
// Pre-baked RGB int8 visualisations let GEE serve plain pixel tiles instead
// of running classify + palette + encode per tile, the dominant cost of
// cold-start tile rendering.
var USE_RGB_ASSET = false;

var ASSET_PATH = 'projects/exalted-country-485019-c8/assets/predictors_stack_icnf_v5';
var REBURN_PROB_ASSET = 'projects/exalted-country-485019-c8/assets/reburn_prob_v5';
var DIAG_ASSET = 'projects/exalted-country-485019-c8/assets/diag_v6';
var MUNI_STATS_ASSET = 'projects/exalted-country-485019-c8/assets/muni_stats_v6';
var ICNF_2017_ASSET = 'projects/exalted-country-485019-c8/assets/icnf_2017_centre';
var ICNF_REBURN_ASSET = 'projects/exalted-country-485019-c8/assets/icnf_reburns_2018_2025';
var CLASS5_RGB_ASSET = 'projects/exalted-country-485019-c8/assets/class5_rgb_v6';
var PROB_RGB_ASSET = 'projects/exalted-country-485019-c8/assets/prob_rgb_v6';

var GRID_SCALE = 100;
var TARGET_PROJ = 'EPSG:3763';

// 2017 burn footprint from ICNF
// only Jun-Jul 2017 fires > 500 ha. October fires fall after the post-fire imagery window.
var burn_vector_2017 = ee.FeatureCollection(ICNF_2017_ASSET)
  .filter(ee.Filter.and(
    ee.Filter.gte('DH_Inicio', ee.Date('2017-06-01').millis()),
    ee.Filter.lt('DH_Inicio', ee.Date('2017-08-01').millis()),
    ee.Filter.gte('AreaHaPoly', 500)
  ));

var aoi_burn = burn_vector_2017.geometry();
var aoi_buffered = aoi_burn.buffer(5000);

// reburn labels: 1 where ICNF 2018-2025 intersects the 2017 footprint, else 0
var icnf_reburn = ee.FeatureCollection(ICNF_REBURN_ASSET).filterBounds(aoi_burn);
var burned_after = ee.Image().byte()
  .paint(icnf_reburn, 1).unmask(0).clip(aoi_burn).rename('reburn');

print('polygons kept:', burn_vector_2017.size());
print('areas (ha):', burn_vector_2017.aggregate_array('AreaHaPoly'));

// 2017 dNBR from Sentinel-2
var S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var CSPLUS = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
// CloudScore+ moderate cutoff (Pasquarella et al. 2023). Stricter values
// drop too many summer pixels in cloud-prone central Portugal.
var CS_THRESHOLD = 0.60;

function maskAndScale(img) {
  var clear = img.select('cs_cdf').gte(CS_THRESHOLD);
  return img.updateMask(clear).select(['B2','B3','B4','B8','B11','B12']).divide(10000);
}
function addNBR(img) { return img.addBands(img.normalizedDifference(['B8','B12']).rename('NBR')); }
function addNDMI(img) { return img.addBands(img.normalizedDifference(['B8','B11']).rename('NDMI')); }
function addNDVI(img) { return img.addBands(img.normalizedDifference(['B8','B4']).rename('NDVI')); }

// metadata pre-filter: drop scenes that are >80% cloudy before per-pixel
// CloudScore+ masking, to avoid wasted compute on useless tiles.
var s2_with_cs = S2.linkCollection(CSPLUS, ['cs','cs_cdf'])
  .filterBounds(aoi_burn)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80));

var nbr_pre = s2_with_cs.filterDate('2016-06-01','2016-09-30')
  .map(maskAndScale).map(addNBR).select('NBR').median().rename('NBR_pre');
var nbr_post = s2_with_cs.filterDate('2017-08-01','2017-10-31')
  .map(maskAndScale).map(addNBR).select('NBR').median().rename('NBR_post');
var dNBR_2017 = nbr_pre.subtract(nbr_post).multiply(1000).rename('dNBR_2017');

// predictors
function summerNBR(year) {
  var nbr = s2_with_cs.filterDate(year + '-06-01', year + '-09-30')
    .map(maskAndScale).map(addNBR).select('NBR').median();
  return ee.Image.constant(year).toFloat().addBands(nbr).rename(['year','NBR']);
}

// slope fit 2018-2020: bulk Portuguese reburns happened in the 2022-2023 drought,
// so a slope fitted across the full window would be partly driven by NBR collapses
// caused by the very reburn events the model is predicting (temporal leakage).
var slopeYears = [2018, 2019, 2020];
var nbrSeries = ee.ImageCollection(slopeYears.map(summerNBR));
var recoveryFit = nbrSeries.reduce(ee.Reducer.linearFit());
var NBR_slope = recoveryFit.select('scale').rename('NBR_slope');
var NBR_offset = recoveryFit.select('offset').rename('NBR_offset');
var s2_2025 = s2_with_cs.filterDate('2025-06-01','2025-09-30')
  .map(maskAndScale).map(addNBR).map(addNDMI).map(addNDVI);
var NDVI_2025 = s2_2025.select('NDVI').median().rename('NDVI_2025');
var NDMI_2025_min = s2_2025.select('NDMI').min().rename('NDMI_2025_min');

function landsatLST(img) {
  var qa = img.select('QA_PIXEL');
  // mask cloud bit 3 and cloud shadow bit 4. Cirrus bit 2 kept because its
  // effect on ST_B10 is minimal and aggressive QA masking drops valid pixels.
  var clear = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));
  return img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
    .updateMask(clear).rename('LST');
}
var L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2');
var L9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2');
// .max captures fire-danger heat extremes (Yebra 2013), not summer-average
// warming. CLOUD_COVER<60 prefilter on Landsat C2.
var LST_2025_max = L8.merge(L9)
  .filterDate('2025-06-01','2025-09-30').filterBounds(aoi_burn)
  .filter(ee.Filter.lt('CLOUD_COVER', 60))
  .map(landsatLST).max().rename('LST_2025_max');
var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').mosaic().select('DEM').rename('elevation');
var terrain = ee.Terrain.products(dem);
// slope dropped: zero RF importance at 100 m resampling smoothing (Strobl et al. 2007)
var aspect = terrain.select('aspect').unmask(0).rename('aspect');
var elevation = dem.unmask(0);

// dist_settlement dropped in v2: zero RF importance at 100 m
// rural footprint, uniform settlement distance, signal absorbed by elevation
function clipLocal(img) { return img.clip(aoi_buffered); }

// Stage A: bundle 8 predictor bands and export to asset.
var predictors_raw = ee.Image.cat([
  clipLocal(dNBR_2017), clipLocal(NBR_slope), clipLocal(NBR_offset),
  clipLocal(NDVI_2025), clipLocal(NDMI_2025_min), clipLocal(LST_2025_max),
  clipLocal(aspect), clipLocal(elevation)
]);

Export.image.toAsset({
  image: predictors_raw.toFloat(),
  description: 'predictors_export_icnf',
  assetId: ASSET_PATH,
  region: aoi_burn, scale: GRID_SCALE, crs: TARGET_PROJ, maxPixels: 1e10
});

var predictors = USE_ASSET ? ee.Image(ASSET_PATH).clip(aoi_burn) : predictors_raw.clip(aoi_burn);
var label_100m = burned_after.clip(aoi_burn).rename('reburn');
// Each team member appends their assigned section below in order.
// Refer to TASK_SPLIT.md for the section ownership map and the code for each Part.

// model
if (USE_ASSET) {

  var SAMPLES_PER_CLASS = 2000;
  var SEED = 42;

  // Stratified sample fixes per-class count, otherwise the ~22% positive
  // prevalence would let random sampling under-represent reburn cells.
  var allLabelPoints = label_100m.stratifiedSample({
    numPoints: SAMPLES_PER_CLASS, classBand: 'reburn',
    region: aoi_burn, scale: 100, projection: TARGET_PROJ,
    tileScale: 8, geometries: true, seed: SEED
  });

  var BANDS = predictors.bandNames();

  // Spatial holdout: hold out Pedrógão Grande, train on the other three polygons.
  // Random 70/30 splits inflate AUC under spatial autocorrelation because
  // neighbouring 100 m pixels carry near-identical predictor signatures
  // (Roberts et al. 2017; Bastos Moroz and Thieken 2026).
  var nPoly = burn_vector_2017.size();
  var sortedList = burn_vector_2017.sort('AreaHaPoly', false).toList(nPoly);
  var burnIds = ee.FeatureCollection(ee.List.sequence(0, nPoly.subtract(1)).map(function(i){
    return ee.Feature(sortedList.get(i)).set('poly_id', i);
  }));
  var polyIdImage = ee.Image().byte().paint(burnIds, 'poly_id')
    .rename('poly_id').clip(aoi_burn);

  var allData = predictors.addBands(polyIdImage).sampleRegions({
    collection: allLabelPoints, properties: ['reburn'],
    scale: 100, projection: TARGET_PROJ,
    tileScale: 16, geometries: false
  }).filter(ee.Filter.notNull(BANDS));

  print('positives per poly:', allData.filter(ee.Filter.eq('reburn',1)).aggregate_histogram('poly_id'));
  print('negatives per poly:', allData.filter(ee.Filter.eq('reburn',0)).aggregate_histogram('poly_id'));

  // poly_id by descending area: 0=Mação, 1=Pedrógão, 2=Góis, 3=Abrantes
  var TEST_POLY_ID = 1;
  var trainData = allData.filter(ee.Filter.neq('poly_id', TEST_POLY_ID));
  var validData = allData.filter(ee.Filter.eq('poly_id', TEST_POLY_ID));

  // PROBABILITY-mode RF. Confusion matrix at the Youden J* threshold below
  // is the reported operating point. Kappa at 0.5 is uninformative because
  // class imbalance compresses output probabilities into a low range.
  // numberOfTrees=100: hyperparameter scan tested {50,100,200,300}, AUC plateaued at 100.
  // minLeafPopulation=10: scan tested {1,5,10,20}; values <10 produced
  // visibly wider train-test AUC gaps consistent with overfitting.
  var rfProb = ee.Classifier.smileRandomForest({numberOfTrees: 100, minLeafPopulation: 10, seed: SEED})
    .setOutputMode('PROBABILITY')
    .train({features: trainData, classProperty:'reburn', inputProperties: BANDS});

  var validWithProb = validData.classify(rfProb, 'predicted_prob');

  var brierFC = validWithProb.map(function(f){
    var p = ee.Number(f.get('predicted_prob'));
    var y = ee.Number(f.get('reburn'));
    return f.set('sq_err', p.subtract(y).pow(2));
  });
  var brierScore = brierFC.aggregate_mean('sq_err');

  // Server returns paired [prob, label] rows via a single reduceColumns call.
  // AUC is then computed client-side with the Mann-Whitney U formula. The
  // previous server-side 51-threshold sweep blew the user memory limit on
  // the deployed App.
  var aucValRows = validWithProb.reduceColumns(
    ee.Reducer.toList(2), ['predicted_prob', 'reburn']);
  var aucTrainRows = trainData.classify(rfProb, 'predicted_prob').reduceColumns(
    ee.Reducer.toList(2), ['predicted_prob', 'reburn']);

  // Youden J threshold (Youden 1950).
  // Kappa at 0.5 is uninformative for compressed probabilities under class
  // imbalance, so we scan thresholds from 0.02 to 0.50 in 0.02 steps and
  // report the confusion matrix at J* = argmax(TPR - FPR).
  function youdensJ(fc, probField, truthField){
    var ts = ee.List.sequence(0.02, 0.5, 0.02);
    var pts = ee.FeatureCollection(ts.map(function(t){
      t = ee.Number(t);
      var tp = fc.filter(ee.Filter.and(ee.Filter.gte(probField,t), ee.Filter.eq(truthField,1))).size();
      var fp = fc.filter(ee.Filter.and(ee.Filter.gte(probField,t), ee.Filter.eq(truthField,0))).size();
      var fn = fc.filter(ee.Filter.and(ee.Filter.lt(probField,t), ee.Filter.eq(truthField,1))).size();
      var tn = fc.filter(ee.Filter.and(ee.Filter.lt(probField,t), ee.Filter.eq(truthField,0))).size();
      var tpr = ee.Number(tp).divide(ee.Number(tp).add(fn).max(1));
      var fpr = ee.Number(fp).divide(ee.Number(fp).add(tn).max(1));
      return ee.Feature(null, {threshold: t, J: tpr.subtract(fpr)});
    }));
    return ee.Number(pts.sort('J', false).first().get('threshold'));
  }

  var optThreshold = youdensJ(validWithProb, 'predicted_prob', 'reburn');
  var validClassOpt = validWithProb.map(function(f){
    return f.set('pred_opt', ee.Number(f.get('predicted_prob')).gte(optThreshold).toInt());
  });
  var confMatrixOpt = validClassOpt.errorMatrix('reburn', 'pred_opt');

  // Hyperparameter tuning. Set DEBUG_MODE=true to scan trees and minLeafPopulation.
  // Adds ~1-2 minutes; only useful when retraining on a new label set.
  var DEBUG_MODE = false;
  if (DEBUG_MODE) {
    [50, 100, 200, 300].forEach(function(nT){
      var rf = ee.Classifier.smileRandomForest({numberOfTrees: nT, seed: SEED})
        .setOutputMode('PROBABILITY')
        .train({features: trainData, classProperty:'reburn', inputProperties: BANDS});
      print('trees=' + nT + ' val rows count:',
        validData.classify(rf,'predicted_prob').reduceColumns(ee.Reducer.toList(2), ['predicted_prob', 'reburn']));
    });
    [1, 5, 10, 20].forEach(function(mlp){
      var rf = ee.Classifier.smileRandomForest({numberOfTrees: 100, minLeafPopulation: mlp, seed: SEED})
        .setOutputMode('PROBABILITY')
        .train({features: trainData, classProperty:'reburn', inputProperties: BANDS});
      print('minLeaf=' + mlp + ' val rows count:',
        validData.classify(rf,'predicted_prob').reduceColumns(ee.Reducer.toList(2), ['predicted_prob', 'reburn']));
    });
  }

  // PDP: mean predicted prob binned by each predictor (10 bins).
  // Curves answer Q1 (recovery slope vs reburn) and Q2 (severity vs reburn).
  var allDataWithProb = allData.classify(rfProb, 'predicted_prob');

  function binnedPDP(fc, varName, nBins) {
    var stats = fc.reduceColumns(ee.Reducer.minMax(), [varName]);
    var vmin = ee.Number(stats.get('min'));
    var vmax = ee.Number(stats.get('max'));
    var step = vmax.subtract(vmin).divide(nBins);
    var bins = ee.List.sequence(0, nBins - 1).map(function(i){
      i = ee.Number(i);
      var lo = vmin.add(step.multiply(i));
      var hi = vmin.add(step.multiply(i.add(1)));
      var subset = fc.filter(ee.Filter.and(
        ee.Filter.gte(varName, lo),
        ee.Filter.lt(varName, hi)
      ));
      return ee.Feature(null, {
        bin_center: lo.add(hi).divide(2),
        mean_prob: ee.Algorithms.If(subset.size().gt(0), subset.aggregate_mean('predicted_prob'), null),
        count: subset.size()
      });
    });
    return ee.FeatureCollection(bins).filter(ee.Filter.notNull(['mean_prob']));
  }

  var pdp_NBRslope = binnedPDP(allDataWithProb, 'NBR_slope', 10);
  var pdp_dNBR = binnedPDP(allDataWithProb, 'dNBR_2017', 10);
  var pdp_NDVI = binnedPDP(allDataWithProb, 'NDVI_2025', 10);
  var pdp_LST = binnedPDP(allDataWithProb, 'LST_2025_max', 10);

  // Stage B: cache RF probability surface to asset. RF inference over the
  // ~83,000-pixel footprint is the slowest step (~30-60 s). Cached at startup
  // when USE_PROB_ASSET is true, dropping cold start.
  var reburnProbLive = predictors.classify(rfProb).rename('reburn_prob');

  Export.image.toAsset({
    image: reburnProbLive.toFloat(),
    description: 'reburn_prob_export',
    assetId: REBURN_PROB_ASSET,
    region: aoi_burn, scale: GRID_SCALE, crs: TARGET_PROJ, maxPixels: 1e10
  });

  // Stage D: bundle every diagnostic value into a single one-row FeatureCollection.
  // GEE Asset Feature properties only accept scalar Number/String/Date.
  // List<Float>, nested lists, ee.Array, and ee.Dictionary all fail at
  // export. Encode every non-scalar value as a CSV/pipe-delimited String;
  // the App parses back via String.split + parseFloat.
  var rfExplain = ee.Dictionary(rfProb.explain());
  function listFloatToCsv(eeList) {
    return ee.String(ee.List(eeList).map(function(n){
      return ee.Number(n).format('%.10g');
    }).join(','));
  }
  function rows2dToCsv(eeListOfRows) {
    return ee.String(ee.List(eeListOfRows).map(function(row){
      return ee.List(row).map(function(n){
        return ee.Number(n).format('%.10g');
      }).join('|');
    }).join(','));
  }
  var impDict = ee.Dictionary(rfExplain.get('importance'));

  // Percentiles for the 5-class break and the priority filter, captured
  // server-side at export so the App reads them as scalar Numbers and
  // skips the cold-start reduceRegion.
  var probForDiag = USE_PROB_ASSET
    ? ee.Image(REBURN_PROB_ASSET).clip(aoi_burn).rename('reburn_prob')
    : reburnProbLive;
  var probPctDiag = probForDiag.reduceRegion({
    reducer: ee.Reducer.percentile([20,40,60,80]),
    geometry: aoi_burn, scale: 100, crs: TARGET_PROJ,
    tileScale: 8, maxPixels: 1e10, bestEffort: true
  });
  var priorityPctDiag = probForDiag.reduceRegion({
    reducer: ee.Reducer.percentile([50,70,80,90,95]),
    geometry: aoi_burn, scale: 100, crs: TARGET_PROJ,
    tileScale: 8, maxPixels: 1e10, bestEffort: true
  });
  var diagFeat = ee.Feature(ee.Geometry.Point([0, 0]), {
    j: optThreshold,
    acc: confMatrixOpt.accuracy(),
    kappa: confMatrixOpt.kappa(),
    prod: rows2dToCsv(confMatrixOpt.producersAccuracy().toList()),
    cons: rows2dToCsv(confMatrixOpt.consumersAccuracy().toList()),
    oob: rfExplain.get('outOfBagErrorEstimate'),
    imp_keys: ee.String(impDict.keys().join(',')),
    imp_vals: listFloatToCsv(impDict.values()),
    aucValRows: rows2dToCsv(ee.List(ee.Dictionary(aucValRows).get('list'))),
    aucTrainRows: rows2dToCsv(ee.List(ee.Dictionary(aucTrainRows).get('list'))),
    brier: brierScore,
    ns_x: listFloatToCsv(pdp_NBRslope.aggregate_array('bin_center')),
    ns_y: listFloatToCsv(pdp_NBRslope.aggregate_array('mean_prob')),
    dn_x: listFloatToCsv(pdp_dNBR.aggregate_array('bin_center')),
    dn_y: listFloatToCsv(pdp_dNBR.aggregate_array('mean_prob')),
    nd_x: listFloatToCsv(pdp_NDVI.aggregate_array('bin_center')),
    nd_y: listFloatToCsv(pdp_NDVI.aggregate_array('mean_prob')),
    ls_x: listFloatToCsv(pdp_LST.aggregate_array('bin_center')),
    ls_y: listFloatToCsv(pdp_LST.aggregate_array('mean_prob')),
    p20: probPctDiag.get('reburn_prob_p20'),
    p40: probPctDiag.get('reburn_prob_p40'),
    p60: probPctDiag.get('reburn_prob_p60'),
    p80: probPctDiag.get('reburn_prob_p80'),
    pf50: priorityPctDiag.get('reburn_prob_p50'),
    pf70: priorityPctDiag.get('reburn_prob_p70'),
    pf80: priorityPctDiag.get('reburn_prob_p80'),
    pf90: priorityPctDiag.get('reburn_prob_p90'),
    pf95: priorityPctDiag.get('reburn_prob_p95')
  });
  Export.table.toAsset({
    collection: ee.FeatureCollection([diagFeat]),
    description: 'diag_export_v6',
    assetId: DIAG_ASSET
  });

  var reburnProb = USE_PROB_ASSET
    ? ee.Image(REBURN_PROB_ASSET).clip(aoi_burn).rename('reburn_prob')
    : reburnProbLive;

  // Stage D: when USE_DIAG_ASSET is true, the 4 break percentiles are read
  // as scalar Numbers from diag_v6, eliminating the cold-start reduceRegion.
  var probPercentiles;
  if (USE_DIAG_ASSET) {
    var diagFeatRefP = ee.Feature(ee.FeatureCollection(DIAG_ASSET).first());
    probPercentiles = ee.Dictionary({
      'reburn_prob_p20': diagFeatRefP.get('p20'),
      'reburn_prob_p40': diagFeatRefP.get('p40'),
      'reburn_prob_p60': diagFeatRefP.get('p60'),
      'reburn_prob_p80': diagFeatRefP.get('p80')
    });
  } else {
    probPercentiles = reburnProb.reduceRegion({
      reducer: ee.Reducer.percentile([20,40,60,80]),
      geometry: aoi_burn, scale: 100, crs: TARGET_PROJ,
      tileScale: 8, maxPixels: 1e10, bestEffort: true
    });
  }
  var p20 = ee.Number(probPercentiles.get('reburn_prob_p20'));
  var p40 = ee.Number(probPercentiles.get('reburn_prob_p40'));
  var p60 = ee.Number(probPercentiles.get('reburn_prob_p60'));
  var p80 = ee.Number(probPercentiles.get('reburn_prob_p80'));

  // 5-class quantile breaks. AGIF allocates by top-X% of pixels, so quantile
  // bins map directly onto operational language. Fixed thresholds would not.
  var reburnClass = reburnProb
    .where(reburnProb.lte(p20), 1)
    .where(reburnProb.gt(p20).and(reburnProb.lte(p40)), 2)
    .where(reburnProb.gt(p40).and(reburnProb.lte(p60)), 3)
    .where(reburnProb.gt(p60).and(reburnProb.lte(p80)), 4)
    .where(reburnProb.gt(p80), 5)
    .toByte().rename('reburn_class');

  // Município choropleth (FAO GAUL level 2 = concelho).
  var municipalitiesAll = ee.FeatureCollection('FAO/GAUL/2015/level2')
    .filter(ee.Filter.eq('ADM0_NAME','Portugal'));
  var municipalities = municipalitiesAll.filterBounds(aoi_burn);

  var highMask = reburnClass.gte(4).rename('high_mask');

  // Batch reduceRegions avoids "too many concurrent aggregations" tile errors
  // that arise from looping reduceRegion over 16 polygons.
  var muniIntersected = municipalities.map(function(f){
    var inside = f.geometry().intersection(aoi_burn, 10);
    return f.setGeometry(inside).set('area_in_burn_ha', inside.area(1).divide(10000));
  }).filter(ee.Filter.gt('area_in_burn_ha', 10));

  // Stage D: muniStats touches Top 5 evaluate plus two choropleth tile
  // layers. Cache it as an asset so cold start reads 16 small features
  // for free, instead of running the batch reduceRegions on every visit.
  var muniStats;
  if (USE_DIAG_ASSET) {
    muniStats = ee.FeatureCollection(MUNI_STATS_ASSET);
  } else {
    muniStats = reburnProb.addBands(highMask).reduceRegions({
      collection: muniIntersected,
      reducer: ee.Reducer.mean(),
      scale: 100, crs: TARGET_PROJ, tileScale: 16
    }).map(function(f){
      var pct = ee.Number(f.get('high_mask'));
      var areaHa = ee.Number(f.get('area_in_burn_ha'));
      return f.set('mean_reburn_prob', f.get('reburn_prob'))
              .set('pct_high_class', pct)
              // Operational KPI: priority_ha = high+very-high share x concelho area.
              // Matches AGIF's "treat N hectares this year" budget language.
              .set('priority_ha', pct.multiply(areaHa));
    });
  }
  // Always register the export task so a fresh build can regenerate the asset.
  var muniStatsLive = reburnProb.addBands(highMask).reduceRegions({
    collection: muniIntersected,
    reducer: ee.Reducer.mean(),
    scale: 100, crs: TARGET_PROJ, tileScale: 16
  }).map(function(f){
    var pct = ee.Number(f.get('high_mask'));
    var areaHa = ee.Number(f.get('area_in_burn_ha'));
    return f.set('mean_reburn_prob', f.get('reburn_prob'))
            .set('pct_high_class', pct)
            .set('priority_ha', pct.multiply(areaHa));
  });
  Export.table.toAsset({
    collection: muniStatsLive,
    description: 'muni_stats_export_v6',
    assetId: MUNI_STATS_ASSET
  });

  var muniChoropct = ee.Image().float()
    .paint({featureCollection: muniStats, color:'pct_high_class'})
    .updateMask(ee.Image().byte().paint(muniStats, 1));
  var muniChoroPriority = ee.Image().float()
    .paint({featureCollection: muniStats, color:'priority_ha'})
    .updateMask(ee.Image().byte().paint(muniStats, 1));
  var muniBoundaryOutline = ee.Image().byte()
    .paint({featureCollection: municipalities, color: 1, width: 2}).selfMask();
  var muniOutlineAll = ee.Image().byte()
    .paint({featureCollection: municipalitiesAll, color: 1, width: 1}).selfMask();
  var muniMinusBurn = ee.FeatureCollection([
    ee.Feature(municipalities.geometry().difference(aoi_burn, 10))
  ]);
  var muniIntersectFill = ee.Image().byte().paint(muniMinusBurn, 1).selfMask();

  // Cached per-polygon stats for fire summary popups; one batch reduceRegions.
  var perPolygonStats = reburnProb.addBands(highMask).reduceRegions({
    collection: burn_vector_2017,
    reducer: ee.Reducer.mean(),
    scale: 100, crs: TARGET_PROJ, tileScale: 16
  });

  // Stage D: priority filter percentiles come from diag_v6 so the App
  // avoids a second cold-start reduceRegion.
  var priorityPercentiles;
  if (USE_DIAG_ASSET) {
    var diagFeatRefPF = ee.Feature(ee.FeatureCollection(DIAG_ASSET).first());
    priorityPercentiles = ee.Dictionary({
      'reburn_prob_p50': diagFeatRefPF.get('pf50'),
      'reburn_prob_p70': diagFeatRefPF.get('pf70'),
      'reburn_prob_p80': diagFeatRefPF.get('pf80'),
      'reburn_prob_p90': diagFeatRefPF.get('pf90'),
      'reburn_prob_p95': diagFeatRefPF.get('pf95')
    });
  } else {
    priorityPercentiles = reburnProb.reduceRegion({
      reducer: ee.Reducer.percentile([50, 70, 80, 90, 95]),
      geometry: aoi_burn, scale: 100, crs: TARGET_PROJ,
      tileScale: 8, maxPixels: 1e10, bestEffort: true
    });
  }

  // CSV export of top-5% priority pixel centroids; runs from the Tasks tab.
  // Bridges the App to AGIF's downstream GIS workflow without manual digitising.
  var top5pctThreshold = ee.Number(priorityPercentiles.get('reburn_prob_p95'));
  var top5pctMask = reburnProb.gte(top5pctThreshold).selfMask();
  var top5pctPoints = top5pctMask.reduceToVectors({
    geometry: aoi_burn, scale: 100, crs: TARGET_PROJ,
    geometryType: 'centroid', maxPixels: 1e10, bestEffort: true
  });
  var top5pctEnriched = reburnProb.sampleRegions({
    collection: top5pctPoints, scale: 100, projection: TARGET_PROJ,
    geometries: true
  });
  Export.table.toDrive({
    collection: top5pctEnriched,
    description: 'priority_pixels_top5pct_csv',
    fileFormat: 'CSV'
  });
