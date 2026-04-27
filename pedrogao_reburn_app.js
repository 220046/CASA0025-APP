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
