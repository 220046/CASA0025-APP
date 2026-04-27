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
   // UI
  // SATELLITE no road labels gives the cleanest backdrop for the data layers.
  Map.setOptions('SATELLITE');
  Map.style().set('cursor','crosshair');
  Map.centerObject(aoi_burn, 9);

  // Hide native controls except zoom and layer-list. Zoom is retained because
  // the four fire polygons are geographically separated and users need to
  // navigate between them.
  Map.setControlVisibility({all: false, layerList: true, zoomControl: true});
  Map.drawingTools().setShown(false);

  // Single white stroke and black halo for all 4 fires.
  // Differentiation lives in the sidebar legend numbered 1-4 by area, not in colour.
  var allFiresFC = burn_vector_2017.sort('AreaHaPoly', false);
  var sortedFireList = allFiresFC.toList(99);
  var firePolyFCs = [0,1,2,3].map(function(i){
    return ee.FeatureCollection([ee.Feature(sortedFireList.get(i))]);
  });

  // Bundled fireGeoms evaluate: pre-evaluate fire geometries to plain JS GeoJSON
  // once at startup so setActiveFire can call ee.Geometry(jsGeom) synchronously
  // at click time. Bundled into ONE evaluate (was 4 separate round trips) so
  // GEE resolves all four geoms in a single compute graph, ~5-10s saved.
  var fireGeoms = [null, null, null, null];
  ee.Dictionary({
    g0: firePolyFCs[0].geometry(),
    g1: firePolyFCs[1].geometry(),
    g2: firePolyFCs[2].geometry(),
    g3: firePolyFCs[3].geometry()
  }).evaluate(function(d, err){
    if (err || !d) return;
    fireGeoms[0] = d.g0; fireGeoms[1] = d.g1;
    fireGeoms[2] = d.g2; fireGeoms[3] = d.g3;
  });
  // Morphological closing +150m / -150m fills internal unburned pockets <300m
  // so the perimeter doesn't paint noisy finger outlines.
  var allFiresFCClean = allFiresFC.map(function(f){
    return f.setGeometry(f.geometry().buffer(150, 30).buffer(-150, 30));
  });
  var firesHalo = ee.Image().byte().paint(allFiresFCClean, 1, 3).selfMask();
  var firesStroke = ee.Image().byte().paint(allFiresFCClean, 1, 1).selfMask();

  // Top-center floating label shows the current focus name (fire or concelho).
  // GEE has no on-map text-at-lat-lng widget.
  var contextLabel = ui.Label('', {
    fontWeight: 'bold', fontSize: '13px', color:'#222',
    backgroundColor: 'rgba(255,255,255,0.92)',
    padding: '6px 14px', margin: '8px', textAlign:'center'
  });
  contextLabel.style().set({position: 'top-center', shown: false});
  Map.add(contextLabel);
  function setContextLabel(text){
    if (text) {
      contextLabel.setValue(text);
      contextLabel.style().set('shown', true);
    } else {
      contextLabel.style().set('shown', false);
    }
  }

  // 5-class warm-orange ramp cream -> burnt orange.
  // Single hue, brighter = more risk, harmonises with olive satellite tiles.
  // Inferno was tried first but its deep purple low values were misread as high
  // intensity in lay user tests (Crameri et al. 2020 colour misuse warning).
  var HAZ5 = ['#fff4d6','#ffcb7d','#fb9646','#e85d04','#b54200'];

  // Stage E swap: when USE_RGB_ASSET is true the two heaviest hazard layers
  // are served as pre-baked RGB Uint8 tiles. GEE skips the per-tile classify
  // + palette + encode pipeline, the dominant cold-start cost.
  var class5_display = USE_RGB_ASSET
    ? ee.Image(CLASS5_RGB_ASSET)
    : reburnClass.clip(aoi_burn);
  var class5_vis = USE_RGB_ASSET ? {} : {min:1, max:5, palette: HAZ5};
  var prob_display = USE_RGB_ASSET
    ? ee.Image(PROB_RGB_ASSET)
    : reburnProb.clip(aoi_burn);
  var prob_vis = USE_RGB_ASSET ? {} : {min:0, max:1, palette: HAZ5};

  var LAYER_CONFIG = {
    'class': {
      name:'Reburn susceptibility class',
      image: class5_display, vis: class5_vis,
      legendLabels:['Very low','Low','Moderate','High','Very high'],
      legendPalette: HAZ5
    },
    'prob': {
      name:'Reburn probability (0-1)',
      image: prob_display, vis: prob_vis,
      legendLabels:['0.0','0.25','0.5','0.75','1.0'],
      legendPalette: HAZ5
    },
    'dnbr': {
      // ColorBrewer Oranges, separate hue family from the model-output Reds
      // so users see at a glance this is an INPUT, not the prediction.
      name:'2017 burn severity (dNBR)',
      image: dNBR_2017.clip(aoi_burn),
      vis:{min:0, max:1000, palette:['#feedde','#fdbe85','#fd8d3c','#e6550d','#a63603']},
      legendLabels:['0','250','500','750','1000'],
      legendPalette:['#feedde','#fdbe85','#fd8d3c','#e6550d','#a63603']
    },
    'slope_rec': {
      // Diverging RdYlGn: green = vegetation regrowth, red = degradation.
      name:'NBR recovery slope 2018-2020',
      image: NBR_slope.clip(aoi_burn),
      vis:{min:-0.05, max:0.05, palette:['#d7191c','#fdae61','#ffffbf','#a6d96a','#1a9850']},
      legendLabels:['Degradation','','No change','','Recovery'],
      legendPalette:['#d7191c','#fdae61','#ffffbf','#a6d96a','#1a9850']
    },
    'ndmi': {
      // BrBG diverging. Brown = dry, teal = wet.
      name:'2025 fuel moisture (NDMI min)',
      image: NDMI_2025_min.clip(aoi_burn),
      vis:{min:-0.2, max:0.4, palette:['#8c510a','#d8b365','#f6e8c3','#5ab4ac','#01665e']},
      legendLabels:['Dry','','Mid','','Wet'],
      legendPalette:['#8c510a','#d8b365','#f6e8c3','#5ab4ac','#01665e']
    },
    'lst': {
      // ColorBrewer Reds. heat=red is unambiguous and decouples LST from the
      // hazard layers visually.
      name:'2025 maximum LST (C)',
      image: LST_2025_max.clip(aoi_burn),
      vis:{min:25, max:50, palette:['#fee5d9','#fcae91','#fb6a4a','#de2d26','#a50f15']},
      legendLabels:['25','31','37','43','50'],
      legendPalette:['#fee5d9','#fcae91','#fb6a4a','#de2d26','#a50f15']
    },
    'muni_priority_ha': {
      name:'Município priority area (ha)',
      image: muniChoroPriority,
      vis:{min:0, max:2000, palette: HAZ5},
      legendLabels:['0','500','1000','1500','2000+'],
      legendPalette: HAZ5
    },
    'muni_pct': {
      name:'Municipality % High+VeryHigh cells',
      image: muniChoropct,
      vis:{min:0, max:1, palette: HAZ5},
      legendLabels:['0%','25%','50%','75%','100%'],
      legendPalette: HAZ5
    }
  };

  // Sidebar
  var sidebar = ui.Panel({style:{width:'400px', padding:'12px'}});

  sidebar.add(ui.Label('Pedrógão Grande Reburn Susceptibility',{
    fontSize:'20px', fontWeight:'bold', margin:'0 0 4px 0'
  }));
  sidebar.add(ui.Label('CASA0025 final project | central Portugal, 2017 footprint',{
    fontSize:'11px', color:'#666', margin:'0 0 10px 0'
  }));
  sidebar.add(ui.Label(
    'Maps reburn susceptibility within the 2017 Pedrógão Grande fire ' +
    'footprint ahead of the 2026 fire season. A random forest trained on ' +
    'ICNF 2018-2025 reburn perimeters is conditioned on recovery state, ' +
    '2017 burn severity, 2025 fuel state, and topography. ' +
    'Intended user: AGIF fuel-reduction prioritisation.',
    {fontSize:'11px', margin:'0 0 12px 0'}
  ));

  sidebar.add(ui.Label('Four 2017 fire events',{
    fontWeight:'bold', fontSize:'11px', margin:'0 0 2px 0'
  }));
  sidebar.add(ui.Label('Click a row to spotlight that fire; click again to clear.',{
    fontSize:'9px', color:'#888', margin:'0 0 4px 0', fontStyle:'italic'
  }));
  var firesPanel = ui.Panel({style:{padding:'4px 6px', backgroundColor:'#f7f7f7', margin:'0 0 10px 0'}});

  // Numbered by area rank (1 = largest). Differentiation via badge, name, area,
  // not via colour. All four polygons share one neutral white stroke on the map.
  var FIRE_ROWS = [
    [0, 'Mação (Sobreira Formosa)','33,712 ha','23 Jul 2017'],
    [1, 'Pedrógão Grande (main)', '30,618 ha','17 Jun 2017'],
    [2, 'Góis / Arganil', '17,432 ha','17 Jun 2017'],
    [3, 'Abrantes (small)', ' 1,258 ha','17 Jun 2017']
  ];

  var activeFire = -1;
  var fireRowPanels = [];
  var fireRowButtons = [];

  // SE-offset drop shadow inserted between aoiDim and the data layer.
  // The data layer hides the original polygon, leaving only the offset edge.
  // Built by client-side coordinate offset on the JS-evaluated GeoJSON because
  // ee.Geometry.translate is not exposed on chained Geometry in the JS client.
  var focusShadow = null;

  // 180 m SE offset converted to degrees at ~40 N (central Portugal):
  // 1 deg lng ~ 85 km, 1 deg lat ~ 111 km.
  var SHADOW_DLNG = 180 / 85000;
  var SHADOW_DLAT = -180 / 111000;
  function offsetCoords(c) {
    if (typeof c[0] === 'number') return [c[0] + SHADOW_DLNG, c[1] + SHADOW_DLAT];
    return c.map(offsetCoords);
  }
  function offsetGeoJSON(g) {
    if (g.type === 'GeometryCollection') {
      return {type:'GeometryCollection', geometries: g.geometries.map(offsetGeoJSON)};
    }
    return {type: g.type, coordinates: offsetCoords(g.coordinates)};
  }

  function setActiveFire(id) {
    activeFire = (activeFire === id) ? -1 : id;
    fireRowPanels.forEach(function(p, i){
      p.style().set('backgroundColor', (i === activeFire) ? '#FFE0B2' : '#f7f7f7');
    });
    fireRowButtons.forEach(function(b, i){
      b.setLabel((i === activeFire) ? 'reset' : 'focus');
    });
    if (focusShadow) { Map.layers().remove(focusShadow); focusShadow = null; }
    if (activeFire >= 0 && fireGeoms[activeFire]) {
      var jsGeom = fireGeoms[activeFire];
      var geom = ee.Geometry(jsGeom);
      var shadowGeom = ee.Geometry(offsetGeoJSON(jsGeom))
                         .buffer(150, 30).buffer(-150, 30);
      var shadowFC = ee.FeatureCollection([ee.Feature(shadowGeom)]);
      var shadowImg = ee.Image().byte().paint(shadowFC, 1).selfMask();
      focusShadow = ui.Map.Layer(shadowImg, {palette:['#000000']},
        'focus drop shadow', true, 0.45);
      Map.layers().insert(1, focusShadow);
      Map.centerObject(geom, 11);
      setContextLabel('Fire: ' + FIRE_ROWS[activeFire][1]);
    } else {
      Map.centerObject(aoi_burn, 9);
      setContextLabel(null);
    }
  }

  FIRE_ROWS.forEach(function(row){
    var r = ui.Panel({
      layout: ui.Panel.Layout.flow('horizontal'),
      style:{margin:'1px 0', padding:'3px 4px', backgroundColor:'#f7f7f7'}
    });
    r.add(ui.Label((row[0]+1)+'',{
      fontSize:'10px', fontWeight:'bold', color:'#fff', backgroundColor:'#222',
      width:'18px', textAlign:'center',
      margin:'6px 8px 0 0', padding:'2px 0'
    }));
    r.add(ui.Label(row[1],{fontSize:'10px', margin:'0', padding:'7px 0 0 0', width:'150px'}));
    r.add(ui.Label(row[2],{fontSize:'10px', margin:'0', padding:'7px 0 0 0', width:'58px', color:'#555'}));
    r.add(ui.Label(row[3],{fontSize:'10px', margin:'0', padding:'7px 0 0 0', color:'#555'}));
    var btn = ui.Button({
      label: 'focus',
      style:{margin:'2px 0 0 4px', padding:'0 4px'},
      onClick: (function(fid){ return function(){ setActiveFire(fid); }; })(row[0])
    });
    r.add(btn);
    fireRowPanels.push(r);
    fireRowButtons.push(btn);
    firesPanel.add(r);
  });
  sidebar.add(firesPanel);

  // 1. Map view: pixel layer + município overlay dropdowns
  sidebar.add(ui.Label('1. Map view',{
    fontWeight:'bold', fontSize:'13px', margin:'8px 0 4px 0', color:'#333'
  }));
  sidebar.add(ui.Label('1a. Pixel layer (100 m)',{fontWeight:'bold', margin:'4px 0 4px 0'}));
  var layerSelect = ui.Select({
    items:[
      {label:'Reburn susceptibility class', value:'class'},
      {label:'Reburn probability (0-1)', value:'prob'},
      {label:'2017 burn severity (dNBR)', value:'dnbr'},
      {label:'NBR recovery slope', value:'slope_rec'},
      {label:'2025 fuel moisture (NDMI)', value:'ndmi'},
      {label:'2025 LST max', value:'lst'}
    ],
    value:'class', style:{stretch:'horizontal'}
  });
  sidebar.add(layerSelect);
  var layerHint = ui.Label('5-class risk (quantile breaks at p20/40/60/80).',{
    fontSize:'9px', color:'#666', margin:'2px 0 0 4px', fontStyle:'italic'
  });
  sidebar.add(layerHint);

  sidebar.add(ui.Label('1b. Município overlay',{fontWeight:'bold', margin:'8px 0 4px 0'}));
  var muniSelect = ui.Select({
    items:[
      {label:'None (outlines only)', value:'none'},
      {label:'Priority area per concelho (ha) - AGIF KPI', value:'muni_priority_ha'},
      {label:'% high-risk pixels', value:'muni_pct'}
    ],
    value:'none', style:{stretch:'horizontal'}
  });
  sidebar.add(muniSelect);
  var muniHint = ui.Label('Outlines only.',{
    fontSize:'9px', color:'#666', margin:'2px 0 0 4px', fontStyle:'italic'
  });
  sidebar.add(muniHint);

  // refreshLegend rebuilds the legend swatch row from LAYER_CONFIG.
  function refreshLegend(key) {
    legendPanel.clear();
    var cfg = LAYER_CONFIG[key];
    legendPanel.add(ui.Label(cfg.name,{fontWeight:'bold', fontSize:'11px', margin:'0 0 4px 0'}));
    var swatchRow = ui.Panel({layout: ui.Panel.Layout.flow('horizontal')});
    cfg.legendPalette.forEach(function(c){
      swatchRow.add(ui.Label('',{
        backgroundColor:c, padding:'8px', margin:'0 2px 0 0',
        border:'1px solid #999', width:'34px', height:'14px'
      }));
    });
    legendPanel.add(swatchRow);
    var labelRow = ui.Panel({layout: ui.Panel.Layout.flow('horizontal')});
    cfg.legendLabels.forEach(function(l){
      labelRow.add(ui.Label(l,{fontSize:'9px', margin:'0 6px 0 0', width:'34px', textAlign:'center'}));
    });
    legendPanel.add(labelRow);
  }

  // AOI dim mask: 30% black inside the burn footprint to lift data contrast
  // against the satellite green basemap.
  var aoiDim = ee.Image().byte().paint(burn_vector_2017, 1).selfMask();

  function refreshMap() {
    Map.layers().reset();
    var pixelKey = layerSelect.getValue();
    var muniKey = muniSelect.getValue();
    var cfg = LAYER_CONFIG[pixelKey];

    // 1. AOI dim mask
    Map.addLayer(aoiDim, {palette:['#000000']}, 'AOI dim', true, 0.30);

    // 2. Focus drop-shadow (sits BELOW data so only SE-offset edge is visible)
    if (focusShadow) Map.layers().add(focusShadow);

    // 3. Main pixel data layer
    Map.addLayer(cfg.image, cfg.vis, cfg.name, true, 0.82);

    // 4. Optional município choropleth overlay
    if (muniKey !== 'none') {
      var mcfg = LAYER_CONFIG[muniKey];
      Map.addLayer(mcfg.image, mcfg.vis, mcfg.name, true, 0.50);
    }

    // 5. Faint full-Portugal concelho outlines for low-zoom orientation
    Map.addLayer(muniOutlineAll, {palette:['#cccccc']},
      'All Portugal concelhos (faint)', true, 0.45);

    // 6. Soft warm fill on burn-adjacent concelho parts outside the burn
    Map.addLayer(muniIntersectFill, {palette:['#FFCC80']},
      'Burn-adjacent concelhos (fill)', true, 0.32);

    // 7. Burn-adjacent concelhos drawn at full polygon extent boundary
    Map.addLayer(muniBoundaryOutline, {palette:['#1a1a1a']},
      'Burn-adjacent concelhos (outline)', true, 0.85);

    // 8. 2017 fire perimeters: black halo first, white stroke on top.
    Map.addLayer(firesHalo, {palette:['#000000']}, '2017 fire perimeters (halo)', true, 0.55);
    Map.addLayer(firesStroke, {palette:['#ffffff']}, '2017 fire perimeters', true, 0.95);

    // 9. Treatment-priority Top-X% mask if active
    if (priorityLayer) Map.layers().add(priorityLayer);
  }

  var LAYER_HINTS = {
    'class': '5-class risk (quantile breaks at p20/40/60/80).',
    'prob': 'Random-forest output probability, 0 to 1.',
    'dnbr': 'Initial burn severity: high = trees killed in 2017.',
    'slope_rec': 'Vegetation recovery rate 2018-2020 (pre-drought window).',
    'ndmi': 'Fuel moisture in 2025 summer minimum; lower = drier.',
    'lst': '2025 summer maximum land-surface temperature.'
  };
  var MUNI_HINTS = {
    'none': 'Outlines only.',
    'muni_priority_ha': 'Priority hectares per concelho - matches AGIF "treat N ha" budget language.',
    'muni_pct': 'Share of pixels classified High or Very High.'
  };

  // Click inspector panel
  sidebar.add(ui.Label('4. Inspect a location',{
    fontWeight:'bold', fontSize:'13px', margin:'14px 0 4px 0', color:'#333'
  }));
  sidebar.add(ui.Label('Click anywhere on the map. A cyan marker highlights the pixel; summary + values update in the cards below.',{
    fontSize:'10px', color:'#666', margin:'0 0 6px 0'
  }));

  var clickInfo = ui.Panel({style:{padding:'6px', backgroundColor:'#f4f4f4', margin:'0 0 8px 0'}});
  clickInfo.add(ui.Label('(click a point to see pixel values)',{
    fontSize:'10px', color:'#999', margin:'0'
  }));
  sidebar.add(clickInfo);

  var chartPanel = ui.Panel({style:{margin:'0 0 10px 0'}});
  sidebar.add(chartPanel);

  var clickMarker = null;

  var resetBtn = ui.Button({
    label: 'Clear inspection',
    onClick: function(){
      clickInfo.clear();
      clickInfo.add(ui.Label('(click a point to see pixel values)',{
        fontSize:'10px', color:'#999', margin:'0'
      }));
      chartPanel.clear();
      if (clickMarker) { Map.layers().remove(clickMarker); clickMarker = null; }
      setContextLabel(null);
    },
    style: {stretch:'horizontal', margin:'0 0 10px 0'}
  });
  sidebar.add(resetBtn);

  // Full 8-year window for the trajectory chart. slope fit stays 2018-2020
  // to avoid temporal leakage from drought-year reburns.
  var trajectoryYears = [2018,2019,2020,2021,2022,2023,2024,2025];
  var nbrCol = ee.ImageCollection(trajectoryYears.map(function(y){
    var nbr = s2_with_cs.filterDate(y + '-06-01', y + '-09-30')
      .map(maskAndScale).map(addNBR).select('NBR').median().rename('NBR');
    return nbr.set('system:time_start', ee.Date(y + '-08-01').millis());
  }));

  Map.onClick(function(coords){
    var point = ee.Geometry.Point([coords.lon, coords.lat]);

    if (clickMarker) Map.layers().remove(clickMarker);
    clickMarker = Map.addLayer(point.buffer(100), {color:'00FFFF'}, 'click-marker');

    clickInfo.clear();
    chartPanel.clear();
    clickInfo.add(ui.Label('Checking location...',{color:'#666', fontSize:'11px'}));

    // Boundary check: stop early if outside the 2017 footprint, otherwise
    // the user gets silent failure and assumes the App is broken.
    aoi_burn.contains(point, 1).evaluate(function(inside){
      if (!inside) {
        clickInfo.clear();
        clickInfo.add(ui.Label('Click outside the 2017 burn footprint.',{
          color:'#c00', fontSize:'11px', fontWeight:'bold', margin:'0 0 4px 0'
        }));
        clickInfo.add(ui.Label('The model only applies inside the coloured polygons.',{
          color:'#666', fontSize:'10px'
        }));
        return;
      }
      runPixelInspector(point, coords);
    });
  });

  function runPixelInspector(point, coords) {
    clickInfo.clear();
    clickInfo.add(ui.Label('Pixel values at click:',{
      fontWeight:'bold', fontSize:'10px', margin:'0 0 4px 0', color:'#333'
    }));

    var sampleImg = predictors.addBands(reburnProb).addBands(reburnClass);
    sampleImg.reduceRegion({
      reducer: ee.Reducer.first(),
      geometry: point, scale: 100, crs: TARGET_PROJ
    }).evaluate(function(res){
      if (!res) {
        clickInfo.add(ui.Label('No pixel data here (likely cloud-masked).',{color:'#c00', fontSize:'11px'}));
        return;
      }
      var classNames = ['n/a','Very low','Low','Moderate','High','Very high'];
      function fmt(v,d){ return (v==null) ? 'n/a' : Number(v).toFixed(d); }
      var rows = [
        ['Lat, Lon', coords.lat.toFixed(4) + ', ' + coords.lon.toFixed(4)],
        ['Município', '...'],
        ['Reburn probability', fmt(res.reburn_prob, 3)],
        ['Susceptibility class', res.reburn_class != null ? classNames[Math.round(res.reburn_class)] : 'n/a'],
        ['2017 burn severity', fmt(res.dNBR_2017, 0)],
        ['Recovery slope', fmt(res.NBR_slope, 4)],
        ['2025 NDVI', fmt(res.NDVI_2025, 3)],
        ['2025 fuel moisture', fmt(res.NDMI_2025_min, 3)],
        ['2025 max LST (C)', fmt(res.LST_2025_max, 1)],
        ['Aspect (deg)', fmt(res.aspect, 1)],
        ['Elevation (m)', fmt(res.elevation, 0)]
      ];
      var muniValueLabels = [];
      rows.forEach(function(r, idx){
        var row = ui.Panel({layout: ui.Panel.Layout.flow('horizontal'), style:{margin:'0'}});
        row.add(ui.Label(r[0] + ':',{fontSize:'10px', margin:'0', width:'150px', color:'#333'}));
        var valLbl = ui.Label(r[1],{fontSize:'10px', margin:'0', fontWeight:'bold'});
        if (r[0] === 'Município') {
          valLbl.style().set({color:'#e65100'});
          muniValueLabels.push(valLbl);
        }
        row.add(valLbl);
        clickInfo.add(row);
      });

      muniStats.filterBounds(point).first().evaluate(function(m){
        var name = (m && m.properties && m.properties.ADM2_NAME) || 'outside any concelho';
        muniValueLabels.forEach(function(l){ l.setValue(name); });
      });
    });

    chartPanel.clear();
    chartPanel.add(ui.Label('NBR recovery trajectory 2018-2025',{
      fontWeight:'bold', fontSize:'11px', margin:'8px 0 4px 0'
    }));
    chartPanel.add(ui.Label('(building chart, ~3s)',{
      color:'#999', fontSize:'10px', margin:'0 0 4px 0'
    }));

    var chart = ui.Chart.image.series({
      imageCollection: nbrCol, region: point,
      reducer: ee.Reducer.first(), scale: 100,
      xProperty: 'system:time_start'
    }).setOptions({
      title: 'NBR at clicked pixel',
      vAxis: {title:'NBR', viewWindow:{min:-0.5, max:1}},
      hAxis: {title:''},
      legend: {position:'none'},
      lineWidth: 2, pointSize: 5,
      colors: ['#1a9850'],
      height: 180, chartArea: {left:50, top:30, width:'70%', height:'60%'}
    });
    chartPanel.widgets().reset([
      ui.Label('NBR recovery trajectory 2018-2025',{
        fontWeight:'bold', fontSize:'11px', margin:'8px 0 4px 0'
      }),
      chart
    ]);
  }

  // Top 5 priority concelhos
    sidebar.add(ui.Label('2. AGIF decision support',{
      fontWeight:'bold', fontSize:'13px', margin:'14px 0 4px 0', color:'#333'
    }));
    sidebar.add(ui.Label('2a. Top 5 priority concelhos',{
      fontWeight:'bold', margin:'4px 0 4px 0'
    }));
    sidebar.add(ui.Label('Ranked by priority hectares (high+very-high share x concelho area).',{
      fontSize:'9px', color:'#888', margin:'0 0 4px 0', fontStyle:'italic'
    }));
    var top5Panel = ui.Panel({
      style:{padding:'4px', backgroundColor:'#ffffff',
             border:'1px solid #ddd', margin:'0 0 4px 0'}
    });
    top5Panel.add(ui.Label('loading ranking...',{fontSize:'10px', color:'#999', margin:'0'}));
    sidebar.add(top5Panel);
  
    // Shared summary card: populated when user clicks a Top 5 row, also when
    // the pixel inspector resolves a concelho name. Single card avoids two
    // cards showing different concelhos at once.
    var concelhoSummaryCard = ui.Panel({
      style:{padding:'8px 10px', backgroundColor:'#fff8e1',
             border:'2px solid #ff9800', margin:'0 0 8px 0'}
    });
    concelhoSummaryCard.add(ui.Label('Click a row above to see its operational summary.',{
      fontSize:'10px', color:'#888', margin:'0'
    }));
    sidebar.add(concelhoSummaryCard);
  
    function fillConcelhoSummary(p, name){
      concelhoSummaryCard.clear();
      var areaHa = Math.round(p.area_in_burn_ha || 0);
      var meanProb = p.mean_reburn_prob != null ? p.mean_reburn_prob.toFixed(3) : 'n/a';
      var pctHigh = p.pct_high_class != null ? p.pct_high_class : 0;
      var priorityHa = Math.round(pctHigh * areaHa);
      concelhoSummaryCard.add(ui.Label('Concelho: ' + name,{
        fontWeight:'bold', fontSize:'12px', color:'#e65100', margin:'0 0 4px 0'
      }));
      concelhoSummaryCard.add(ui.Label('Area in 2017 burn footprint: ' + areaHa + ' ha',{
        fontSize:'11px', margin:'0'
      }));
      concelhoSummaryCard.add(ui.Label('Mean reburn probability: ' + meanProb,{
        fontSize:'11px', margin:'0'
      }));
      concelhoSummaryCard.add(ui.Label('High/Very-High share: ' + (pctHigh*100).toFixed(1) + '%',{
        fontSize:'11px', margin:'0'
      }));
      concelhoSummaryCard.add(ui.Label('~' + priorityHa + ' ha for priority fuel-reduction treatment',{
        fontSize:'11px', margin:'4px 0 0 0', fontWeight:'bold', color:'#bf360c'
      }));
    }
  
    muniStats.sort('priority_ha', false).limit(5).evaluate(function(fc){
      top5Panel.clear();
      if (!fc || !fc.features || fc.features.length === 0) {
        top5Panel.add(ui.Label('No data.',{fontSize:'10px', color:'#999'}));
        return;
      }
      fc.features.forEach(function(f, i){
        var p = f.properties;
        var name = p.ADM2_NAME || 'Unknown';
        var ha = Math.round(p.priority_ha || 0);
        var bg = (i === 0) ? '#ffe0b2' : (i === 1 ? '#fff3e0' : '#fafafa');
        var row = ui.Panel({
          layout: ui.Panel.Layout.flow('horizontal'),
          style:{margin:'1px 0', padding:'2px 4px', backgroundColor: bg}
        });
        row.add(ui.Label('#' + (i+1),{
          fontWeight:'bold', fontSize:'10px', width:'24px', color:'#bf360c'
        }));
        row.add(ui.Label(name,{fontSize:'10px', margin:'0', width:'160px'}));
        row.add(ui.Label(ha + ' ha',{
          fontSize:'10px', margin:'0', width:'58px', color:'#555', fontWeight:'bold'
        }));
        var btn = ui.Button({
          label: 'view',
          style:{margin:'0 0 0 4px', padding:'0 4px'},
          onClick: (function(featRef, nm){
            return function(){
              Map.centerObject(ee.Geometry(featRef.geometry), 12);
              fillConcelhoSummary(featRef.properties, nm);
              setContextLabel('Concelho: ' + nm);
            };
          })(f, name)
        });
        row.add(btn);
        top5Panel.add(row);
      });
    });
  
    // Treatment priority filter (Top X% via discrete Select)
    // Discrete percentile filter 5/10/20/30/50% for budget-sized AOI subsets.
    sidebar.add(ui.Label('2b. Treatment priority filter',{
      fontWeight:'bold', margin:'10px 0 4px 0'
    }));
    sidebar.add(ui.Label('Mask all but the top X% of pixels - sized to your annual budget.',{
      fontSize:'9px', color:'#888', margin:'0 0 4px 0', fontStyle:'italic'
    }));
    var priorityLayer = null;
    var priorityHintLbl = ui.Label('No filter - full pixel layer shown.',{
      fontSize:'9px', color:'#666', margin:'2px 0 0 4px', fontStyle:'italic'
    });
    var prioritySelect = ui.Select({
      items: [
        {label: 'No filter (full map)', value: '0'},
        {label: 'Top 5% (highest priority)', value: '5'},
        {label: 'Top 10%', value: '10'},
        {label: 'Top 20%', value: '20'},
        {label: 'Top 30%', value: '30'},
        {label: 'Top 50%', value: '50'}
      ],
      value: '0',
      style: {stretch:'horizontal'},
      onChange: function(pctStr){
        if (priorityLayer) {
          Map.layers().remove(priorityLayer);
          priorityLayer = null;
        }
        var pct = Number(pctStr);
        if (pct === 0) {
          priorityHintLbl.setValue('No filter - full pixel layer shown.');
          return;
        }
        var pKey = 'reburn_prob_p' + (100 - pct);
        var thresh = ee.Number(priorityPercentiles.get(pKey));
        var mask = reburnProb.gte(thresh).selfMask();
        priorityLayer = ui.Map.Layer(mask,
          {palette:['#bf360c']},
          'Top ' + pct + '% priority',
          true, 0.92);
        Map.layers().add(priorityLayer);
        var approxHa = Math.round(pct / 100 * 83000);
        priorityHintLbl.setValue(
          '~ ' + approxHa + ' ha highlighted - top ' + pct + '% of model probability.'
        );
      }
    });
    sidebar.add(prioritySelect);
    sidebar.add(priorityHintLbl);
  
    // Legend container (referenced by refreshLegend in Part 4)
    var legendPanel = ui.Panel({style:{margin:'8px 0', padding:'6px', backgroundColor:'#fafafa'}});
    sidebar.add(legendPanel);
  
    // Split-view: two predictors side-by-side; default = recovery slope vs prob.
    var splitView = null;
    var splitContainer = null;
    var splitLeftMap = null;
    var splitRightMap = null;
    var splitLeftSelect = null;
    var splitRightSelect = null;
  
    // pixel-level layers eligible for split comparison; skip muni_* aggregates.
    var SPLIT_KEYS = ['prob', 'class', 'dnbr', 'slope_rec', 'ndmi', 'lst'];
  
    // Static layers (AOI dim, fire halo, fire stroke) are key-independent and
    // built once per map. Only the predictor layer at index 1 changes on
    // dropdown / preset clicks. Same-key calls are no-ops to avoid redundant
    // tile generation.
    function refreshSplitMap(m, key){
      var cfg = LAYER_CONFIG[key];
      if (m.layers().length() === 0) {
        m.addLayer(aoiDim, {palette:['#000000']}, 'AOI dim', true, 0.30);
        m.addLayer(cfg.image, cfg.vis, cfg.name, true, 0.85);
        m.addLayer(firesHalo, {palette:['#000000']}, 'halo', true, 0.55);
        m.addLayer(firesStroke, {palette:['#ffffff']}, 'fires', true, 0.95);
        m._splitKey = key;
        return;
      }
      if (m._splitKey === key) return;
      m.layers().set(1, ui.Map.Layer(cfg.image, cfg.vis, cfg.name, true, 0.85));
      m._splitKey = key;
    }
  
    function buildSplitView(){
      splitLeftMap = ui.Map();
      splitRightMap = ui.Map();
      ui.Map.Linker([splitLeftMap, splitRightMap]);
  
      [splitLeftMap, splitRightMap].forEach(function(m){
        m.setOptions('SATELLITE');
        m.style().set('cursor','crosshair');
        m.setControlVisibility({all:false, zoomControl:true});
        m.drawingTools().setShown(false);
      });
  
      var items = SPLIT_KEYS.map(function(k){
        return {label: LAYER_CONFIG[k].name, value: k};
      });
  
      splitLeftSelect = ui.Select({
        items: items, value: 'slope_rec',
        style: {margin:'0 12px 0 0'},
        onChange: function(k){ refreshSplitMap(splitLeftMap, k); }
      });
      splitRightSelect = ui.Select({
        items: items, value: 'prob',
        style: {margin:'0 12px 0 0'},
        onChange: function(k){ refreshSplitMap(splitRightMap, k); }
      });
  
      var backBtn = ui.Button({
        label: 'Back',
        style: {margin:'0 16px 0 0', backgroundColor:'#ffffff', fontWeight:'bold'},
        onClick: function(){
          ui.root.widgets().reset(origRootWidgets);
          Map.centerObject(aoi_burn, 9);
        }
      });
  
      var splitHeader = ui.Panel({
        layout: ui.Panel.Layout.flow('horizontal'),
        style: {backgroundColor:'#222', padding:'8px 12px', stretch:'horizontal'},
        widgets: [
          backBtn,
          ui.Label('LEFT',{
            color:'#fff', fontSize:'11px', fontWeight:'bold',
            margin:'5px 4px 0 0', backgroundColor:'#222'
          }),
          splitLeftSelect,
          ui.Label('vs',{
            color:'#888', fontSize:'11px',
            margin:'5px 10px 0 4px', backgroundColor:'#222'
          }),
          ui.Label('RIGHT',{
            color:'#fff', fontSize:'11px', fontWeight:'bold',
            margin:'5px 4px 0 0', backgroundColor:'#222'
          }),
          splitRightSelect,
          ui.Label(' Drag the divider to compare any two predictors pixel-by-pixel.',{
            color:'#bbb', fontSize:'10px', fontStyle:'italic',
            margin:'5px 0 0 12px', backgroundColor:'#222'
          })
        ]
      });
  
      splitView = ui.SplitPanel({
        firstPanel: splitLeftMap, secondPanel: splitRightMap,
        orientation: 'horizontal', wipe: true,
        style: {stretch:'both'}
      });
  
      splitContainer = ui.Panel({
        widgets: [splitHeader, splitView],
        layout: ui.Panel.Layout.flow('vertical'),
        style: {stretch:'both', padding:'0', margin:'0'}
      });
  
      refreshSplitMap(splitLeftMap, 'slope_rec');
      refreshSplitMap(splitRightMap, 'prob');
    }
  
    var splitCentered = false;
  
    function enterSplitView(leftKey, rightKey){
      if (!splitContainer) buildSplitView();
      ui.root.widgets().reset([splitContainer]);
      if (leftKey) { splitLeftSelect.setValue(leftKey, true); refreshSplitMap(splitLeftMap, leftKey); }
      if (rightKey) { splitRightSelect.setValue(rightKey, true); refreshSplitMap(splitRightMap, rightKey); }
      if (!splitCentered) {
        splitLeftMap.centerObject(aoi_burn, 9);
        splitRightMap.centerObject(aoi_burn, 9);
        splitCentered = true;
      }
    }
  
    // Single entry into split view. Inside the panel, the LEFT and RIGHT
    // dropdowns let the planner pick any layer pair freely.
    sidebar.add(ui.Label('3. Compare two layers (split view)',{
      fontWeight:'bold', fontSize:'13px', margin:'14px 0 4px 0', color:'#333'
    }));
    sidebar.add(ui.Label('Opens a side-by-side wipe view. Dropdowns at the top of the ' +
      'split panel let you pick any layer pair pixel-by-pixel.',{
      fontSize:'9px', color:'#888', margin:'0 0 4px 0', fontStyle:'italic'
    }));
  
    sidebar.add(ui.Button({
      label: 'Open split view',
      style: {stretch:'horizontal', margin:'2px 0 8px 0', backgroundColor:'#fff3e0'},
      onClick: function(){ enterSplitView('slope_rec', 'prob'); }
    }));
  
    // Wire layer + muni dropdown change handlers and initial render.
    layerSelect.onChange(function(key){
      refreshLegend(key); refreshMap();
      layerHint.setValue(LAYER_HINTS[key]);
    });
    muniSelect.onChange(function(key){
      refreshMap();
      muniHint.setValue(MUNI_HINTS[key]);
    });
  
    refreshLegend('class');
    refreshMap();
  
    // 5. Model diagnostics (always expanded, no toggle)
    sidebar.add(ui.Label('5. Model diagnostics',{
      fontWeight:'bold', fontSize:'13px', margin:'14px 0 4px 0', color:'#333'
    }));
  
    // Asset-based diagnostics: when USE_DIAG_ASSET is true, all RF importance,
    // fast metrics, AUC rows, Brier, and PDP arrays are read from the
    // precomputed diag_v6 FeatureCollection asset in ONE round trip. Replaces
    // four separate live evaluate calls each re-running RF inference,
    // dropping App cold start from ~120s to ~5-10s.
    var diagResolved = null;
    var diagWaiters = [];
    function whenDiag(cb) {
      if (diagResolved !== null) cb(diagResolved);
      else diagWaiters.push(cb);
    }
    function parseFloatList(s) {
      if (!s) return [];
      return String(s).split(',').map(function(x){ return parseFloat(x); });
    }
    function parseStringList(s) {
      if (!s) return [];
      return String(s).split(',');
    }
    function parseMatrix(s) {
      if (!s) return [];
      return String(s).split(',').map(function(rowStr){
        return rowStr.split('|').map(function(x){ return parseFloat(x); });
      });
    }
    if (USE_DIAG_ASSET) {
      ee.Feature(ee.FeatureCollection(DIAG_ASSET).first()).toDictionary().evaluate(function(d, err){
        if (err) print('diag asset load error', err);
        diagResolved = err ? {} : (d || {});
        diagWaiters.forEach(function(cb){ cb(diagResolved); });
        diagWaiters = [];
      });
    }
  
    // 5a. Feature importance bar chart
    sidebar.add(ui.Label('5a. Feature importance (RF)',{fontWeight:'bold', margin:'4px 0 4px 0'}));
    var impHost = ui.Panel({style:{margin:'0', padding:'0'}});
    impHost.add(ui.Label('Computing feature importance...',{
      fontSize:'10px', color:'#999', margin:'4px 0', fontStyle:'italic'
    }));
    sidebar.add(impHost);
  
    var PREDICTOR_LABELS = {
      'dNBR_2017': '2017 burn severity',
      'NBR_slope': 'Recovery slope',
      'NBR_offset': 'Recovery offset',
      'NDVI_2025': '2025 NDVI',
      'NDMI_2025_min': '2025 fuel moisture',
      'LST_2025_max': '2025 max LST',
      'aspect': 'Aspect',
      'elevation': 'Elevation'
    };
    function renderImportanceChart(impObj) {
      impHost.clear();
      var dt = [['Predictor', 'Importance']];
      Object.keys(impObj).forEach(function(k){
        var label = PREDICTOR_LABELS[k] || k;
        dt.push([label, Number(impObj[k])]);
      });
      var chart = ui.Chart(dt).setChartType('ColumnChart').setOptions({
        title: 'Mean decrease in impurity',
        hAxis: {title:'', slantedText:true, slantedTextAngle:45, textStyle:{fontSize:9}},
        vAxis: {title:'Importance', textStyle:{fontSize:9}},
        legend: {position:'none'},
        colors: ['#a50026'],
        height: 200, chartArea: {left:50, top:30, width:'80%', height:'55%'}
      });
      impHost.add(chart);
    }
    if (USE_DIAG_ASSET) {
      whenDiag(function(d){
        var keys = parseStringList(d.imp_keys);
        var vals = parseFloatList(d.imp_vals);
        if (!keys.length) {
          impHost.clear();
          impHost.add(ui.Label('Feature importance not available in diag asset.',{
            fontSize:'10px', color:'#c00', margin:'0'
          }));
          return;
        }
        var impObj = {};
        for (var i = 0; i < keys.length; i++) impObj[keys[i]] = vals[i];
        renderImportanceChart(impObj);
      });
    } else {
      ee.Dictionary(rfProb.explain()).get('importance').evaluate(function(impObj, err){
        impHost.clear();
        if (err || !impObj) { return; }
        renderImportanceChart(impObj);
      });
    }
  
    // 5b. Fast metrics: J*, accuracy, kappa, producer's, consumer's, OOB
    sidebar.add(ui.Label('5b. Model performance',{fontWeight:'bold', margin:'10px 0 4px 0'}));
    var metricsPanel = ui.Panel({style:{padding:'6px', backgroundColor:'#f4f4f4', margin:'0 0 8px 0'}});
  
    function metricRow(label){
      var l = ui.Label(label + ': ...',{fontSize:'10px', margin:'0', color:'#999'});
      metricsPanel.add(l);
      return l;
    }
    function fillNum(lbl, prefix, val, digits){
      lbl.setValue(prefix + ': ' + Number(val).toFixed(digits || 3));
      lbl.style().set('color','#222');
    }
  
    var lblJ = metricRow('Youden J threshold');
    var lblAcc = metricRow('Accuracy at J*');
    var lblKappa = metricRow('Kappa at J*');
    var lblProd = metricRow('Producer\'s acc (recall)');
    var lblCons = metricRow('Consumer\'s acc (precision)');
    var lblOOB = metricRow('OOB error estimate');
  
    metricsPanel.add(ui.Label('Train: Mação, Góis, Abrantes. Test: Pedrógão.',{
      fontSize:'9px', color:'#555', margin:'4px 0 0 0'
    }));
    metricsPanel.add(ui.Label('Kappa at 0.5 uninformative for compressed probabilities; J* is the ROC-optimal operating point.',{
      fontSize:'9px', color:'#555', margin:'2px 0 0 0'
    }));
    metricsPanel.add(ui.Label('Class 0 = no reburn, Class 1 = reburn (binary).',{
      fontSize:'9px', color:'#555', margin:'2px 0 0 0'
    }));
  
    function renderFastMetrics(j, acc, kappa, prod, cons, oob) {
      if (j != null) fillNum(lblJ, 'Youden J threshold', j);
      if (acc != null) fillNum(lblAcc, 'Accuracy at J*', acc);
      if (kappa != null) fillNum(lblKappa, 'Kappa at J*', kappa);
      if (prod && prod[0] && prod[1]) {
        lblProd.setValue('Producer\'s acc (recall): no=' + prod[0][0].toFixed(2) + ' / re=' + prod[1][0].toFixed(2));
        lblProd.style().set('color','#222');
      }
      if (cons && cons[0] && cons[0].length >= 2) {
        lblCons.setValue('Consumer\'s acc (precision): no=' + cons[0][0].toFixed(2) + ' / re=' + cons[0][1].toFixed(2));
        lblCons.style().set('color','#222');
      }
      if (oob != null) fillNum(lblOOB, 'OOB error estimate', Number(oob));
    }
    if (USE_DIAG_ASSET) {
      whenDiag(function(d){
        var prodMatrix = parseMatrix(d.prod);
        var consMatrix = parseMatrix(d.cons);
        renderFastMetrics(d.j, d.acc, d.kappa, prodMatrix, consMatrix, d.oob);
      });
    } else {
      // Bundle 6 metrics into one evaluate to avoid GEE's parallel-same-graph
      // rate limiting. Separate requests would stall a few of the metrics.
      var oobErr = ee.Dictionary(rfProb.explain()).get('outOfBagErrorEstimate');
      var fastMetrics = ee.Dictionary({
        j: optThreshold, acc: confMatrixOpt.accuracy(), kappa: confMatrixOpt.kappa(),
        prod: confMatrixOpt.producersAccuracy(), cons: confMatrixOpt.consumersAccuracy(),
        oob: oobErr
      });
      fastMetrics.evaluate(function(d, err){
        if (err || !d) return;
        renderFastMetrics(d.j, d.acc, d.kappa, d.prod, d.cons, d.oob);
      });
    }
  
    sidebar.add(metricsPanel);
  
    // 5c. Probabilistic metrics: AUC test/train + Brier
    sidebar.add(ui.Label('5c. Probabilistic metrics',{fontWeight:'bold', margin:'10px 0 4px 0'}));
    var heavyMetricsPanel = ui.Panel({style:{padding:'6px', backgroundColor:'#f4f4f4', margin:'0 0 8px 0'}});
    var lblAucT = ui.Label('AUC test (spatial holdout): ...',{fontSize:'10px', margin:'0', color:'#0d47a1'});
    var lblAucR = ui.Label('AUC train: ...', {fontSize:'10px', margin:'0', color:'#0d47a1'});
    var lblBrier = ui.Label('Brier score: ...', {fontSize:'10px', margin:'0', color:'#0d47a1'});
    heavyMetricsPanel.add(lblAucT);
    heavyMetricsPanel.add(lblAucR);
    heavyMetricsPanel.add(lblBrier);
    sidebar.add(heavyMetricsPanel);
  
    // Mann-Whitney U formula for AUC, computed in the browser:
    // AUC = (sum_of_ranks_of_positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg).
    // Ties handled with average-rank assignment.
    function clientAUC(rows) {
      if (!rows || !rows.length) return null;
      var indexed = rows.map(function(r){ return [r[0], r[1]]; });
      indexed.sort(function(a, b){ return a[0] - b[0]; });
      var n = indexed.length;
      var ranks = new Array(n);
      var i = 0;
      while (i < n) {
        var j = i;
        while (j < n - 1 && indexed[j+1][0] === indexed[i][0]) j++;
        var avgRank = (i + j) / 2 + 1;
        for (var k = i; k <= j; k++) ranks[k] = avgRank;
        i = j + 1;
      }
      var sumPosRanks = 0, nPos = 0;
      for (var m = 0; m < n; m++) {
        if (indexed[m][1] === 1) { sumPosRanks += ranks[m]; nPos++; }
      }
      var nNeg = n - nPos;
      if (!nPos || !nNeg) return null;
      return (sumPosRanks - nPos * (nPos + 1) / 2) / (nPos * nNeg);
    }
    function renderHeavyMetrics(valRows, trainRows, brier) {
      var vt = clientAUC(valRows);
      var vr = clientAUC(trainRows);
      if (vt != null) lblAucT.setValue('AUC test (spatial holdout): ' + vt.toFixed(3));
      if (vr != null) lblAucR.setValue('AUC train: ' + vr.toFixed(3));
      if (brier != null) lblBrier.setValue('Brier score: ' + Number(brier).toFixed(3));
    }
    if (USE_DIAG_ASSET) {
      whenDiag(function(d){
        var valRows = parseMatrix(d.aucValRows);
        var trainRows = parseMatrix(d.aucTrainRows);
        renderHeavyMetrics(valRows, trainRows, d.brier);
      });
    } else {
      ee.Dictionary({
        val: aucValRows, train: aucTrainRows, brier: brierScore
      }).evaluate(function(d, err){
        if (err || !d) return;
        var valRows = d.val && d.val.list ? d.val.list : d.val;
        var trainRows = d.train && d.train.list ? d.train.list : d.train;
        renderHeavyMetrics(valRows, trainRows, d.brier);
      });
    }
  
    // 5d. Partial dependence plots
    sidebar.add(ui.Label('5d. Variable-probability relationship (PDP)',{
      fontWeight:'bold', margin:'10px 0 4px 0'
    }));
    sidebar.add(ui.Label('Binned mean predicted probability vs each key predictor.',{
      fontSize:'9px', color:'#666', margin:'0 0 4px 0'
    }));
    var pdpHost = ui.Panel({style:{margin:'0', padding:'0'}});
    pdpHost.add(ui.Label('Computing PDP charts...',{
      fontSize:'10px', color:'#999', margin:'4px 0', fontStyle:'italic'
    }));
    sidebar.add(pdpHost);
  
    function pdpLineChart(xArr, yArr, title, xLabel, color){
      if (!xArr || !yArr || xArr.length === 0) return ui.Label('(no data)',{fontSize:'10px', color:'#999'});
      var dt = [[xLabel, 'Mean predicted prob']];
      for (var i = 0; i < xArr.length; i++) {
        dt.push([Number(xArr[i]), Number(yArr[i])]);
      }
      return ui.Chart(dt).setChartType('LineChart').setOptions({
        title: title,
        hAxis: {title: xLabel, textStyle:{fontSize:9}, format:'short'},
        vAxis: {title:'Mean predicted prob', textStyle:{fontSize:9},
                viewWindow:{min:0, max:1}, format:'short'},
        legend: {position:'none'},
        colors: [color], lineWidth: 2, pointSize: 4,
        height: 150, chartArea: {left:50, top:30, width:'80%', height:'55%'}
      });
    }
    function renderPDP(ns_x, ns_y, dn_x, dn_y, nd_x, nd_y, ls_x, ls_y) {
      pdpHost.clear();
      pdpHost.add(pdpLineChart(ns_x, ns_y, 'Recovery slope vs reburn prob', 'Recovery slope', '#1a9850'));
      pdpHost.add(pdpLineChart(dn_x, dn_y, '2017 burn severity vs reburn prob', '2017 burn severity','#a50026'));
      pdpHost.add(pdpLineChart(nd_x, nd_y, '2025 NDVI vs reburn prob', '2025 NDVI', '#66a61e'));
      pdpHost.add(pdpLineChart(ls_x, ls_y, '2025 max LST vs reburn prob', '2025 max LST', '#d73027'));
    }
    if (USE_DIAG_ASSET) {
      whenDiag(function(d){
        renderPDP(
          parseFloatList(d.ns_x), parseFloatList(d.ns_y),
          parseFloatList(d.dn_x), parseFloatList(d.dn_y),
          parseFloatList(d.nd_x), parseFloatList(d.nd_y),
          parseFloatList(d.ls_x), parseFloatList(d.ls_y)
        );
      });
    } else {
      ee.Dictionary({
        ns_x: pdp_NBRslope.aggregate_array('bin_center'),
        ns_y: pdp_NBRslope.aggregate_array('mean_prob'),
        dn_x: pdp_dNBR.aggregate_array('bin_center'),
        dn_y: pdp_dNBR.aggregate_array('mean_prob'),
        nd_x: pdp_NDVI.aggregate_array('bin_center'),
        nd_y: pdp_NDVI.aggregate_array('mean_prob'),
        ls_x: pdp_LST.aggregate_array('bin_center'),
        ls_y: pdp_LST.aggregate_array('mean_prob')
      }).evaluate(function(d, err){
        if (err || !d) return;
        renderPDP(d.ns_x, d.ns_y, d.dn_x, d.dn_y, d.nd_x, d.nd_y, d.ls_x, d.ls_y);
      });
    }
  
    sidebar.add(ui.Label('Data: ICNF fire perimeters; Sentinel-2; Landsat 8/9; Copernicus DEM.',{
      fontSize:'9px', color:'#999', margin:'8px 0 0 0'
    }));
  
    // Bottom spacer to keep the last chart above Google logo overlay.
    sidebar.add(ui.Panel({style:{height:'120px', margin:'0', padding:'0'}}));
  
    // Mount sidebar as Map overlay (top-left).
    // ui.root flex layout was tried but its detach/reattach caused chart
    // library to crash with offsetWidth null on async callbacks. Overlay is stable.
    sidebar.style().set({
      position:'top-left', width:'420px', maxHeight:'100%',
      margin:'0', padding:'10px', backgroundColor:'rgba(255,255,255,0.96)'
    });
    Map.add(sidebar);
  
    // Save ui.root state so the split-view Back button can restore it.
    // ui.root.clear + ui.root.addMap is unreliable: GEE caches Map's render
    // tree but detach/reattach drops widgets, leaving a blank page.
    var origRootWidgets = [];
    var rootList = ui.root.widgets();
    for (var i_root = 0; i_root < rootList.length(); i_root++) {
      origRootWidgets.push(rootList.get(i_root));
    }

  }

  else {
  // First-run preview: USE_ASSET is false, asset hasn't been exported yet.
  // Show three minimal layers so user can verify data wiring before
  // running the export tasks.
  Map.centerObject(aoi_burn, 9);
  Map.addLayer(aoi_burn, {color:'red'}, '2017 Burn Footprint');
  Map.addLayer(burned_after.selfMask(), {palette:['#cc0000']}, 'Reburn 2018-2025');
  Map.addLayer(dNBR_2017.clip(aoi_burn),
    {min:-100, max:1000, palette:['blue','white','yellow','orange','red']}, '2017 dNBR');

  print('USE_ASSET = false. Run predictors_export_icnf, then flip USE_ASSET=true and re-run.');
}
