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

// Each team member appends their assigned section below in order.
// Refer to TASK_SPLIT.md for the section ownership map and the code for each Part.
