import argparse
import rasterio
import scipy
import rasterio.mask
import numpy as np
import geopandas as gpd
import pandas as pd

from rasterio.plot import show
from rasterio.features import geometry_mask
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import distance_transform_edt
from shapely.geometry import mapping

"""
            For debug purposes:
            Each function prints an ECHO status to identify that the function has been called correctly, 
            and then also prints another ECHO to acknowledge that the function has operated successfully.

"""

# Converts raster data to match the raster's resolution
def convert_to_rasterio(raster_data, template_raster):
    print("Echo: fetching convert_to_rasterio function...")
    with rasterio.open(template_raster) as template_raster:
        dataset = template_raster
        band1 = dataset.read(1)
        print("Echo: convert_to_rasterio function success!")
        
        return band1, dataset.meta

# Extract values from a raster based on the shapefile object provided
def extract_values_from_raster(raster, shape_object):
    print("Echo: fetching extract_values_from_raster function...")
    with rasterio.open(raster) as src:
        out_image, out_transform = rasterio.mask.mask(src, [mapping(shape_object)], crop=True)
        print("Echo: extract_values_from_raster function success!")
        return out_image[0]


# A function to create and return a random forest classifier
def make_classifier(x, y, verbose=False): 
    X = data_df[["topo", "geo", "lc", "dist_fault", "slope"]]
    y = data_df["landslide"]
    classifier = make_classifier(X, y, verbose=True)
    classifier.fit(X, y)

    print("Echo: fetching make_classifier function...")
    rand_forest = RandomForestClassifier(verbose=verbose)
    
    print("Echo: make_classifier function success!")
    return rand_forest

# Function to make a probability raster
def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
    print("Echo: fetching make_prob_raster_data function...")
    topo_reshaped = topo.flatten()
    geo_reshaped = geo.flatten()
    lc_reshaped = lc.flatten()
    dist_fault_reshaped = dist_fault.flatten()
    slope_reshaped = slope.flatten()

    features = np.stack([topo, geo, lc, dist_fault, slope], axis=-1)
    prob_raster = classifier.predict_proba(features.reshape(-1, features.shape[-1]))[:, 1]
    print("Echo: make_prob_raster_data success!")
    return prob_raster.reshape(topo.shape)

# Creates a dataframe from the raster and shapefile data
def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides):
    print("Echo: fetching create_dataframe function...")
    data = {
        "topo": [],
        "geo": [],
        "lc": [],
        "dist_fault": [],
        "slope": [],
        "landslide": []
    }
    df = pd.DataFrame(data)

    GeodataFrame = gpd.geodataframe.GeoDataFrame(df)
    print("Echo: create_dataframe function success!")
    return GeodataFrame

# Function to calculate the slope from topography data
def calculate_slope(topo_raster):
    print("Echo: fetching calculate_slope function...")
    with rasterio.open(topo_raster) as src:
        elevation = src.read(1)
        x, y = np.gradient(elevation, src.res[0], src.res[1])
        slope = np.sqrt(x**2 + y**2)
        print("Echo: calculate_slope function success!")
    return slope, src.meta

# Function to calculate distance from faults
def calculate_distance_from_faults(faults_shp, template_raster):
    print("Echo: fetching calculate_distance_from_faults function...")
    with rasterio.open(template_raster) as src:
        transform = src.transform
        shape = src.shape
    
    faults = gpd.read_file(faults_shp)
    fault_mask = geometry_mask([mapping(geom) for geom in faults.geometry], transform=transform, invert=True, out_shape=shape)
    
    distance = distance_transform_edt(~fault_mask)
    print("Echo: calculate_distance_from_faults function success!")
    return distance, src.meta


def main():


    parser = argparse.ArgumentParser(
                     prog="Landslide hazard using ML",
                     description="Calculate landslide hazards using simple ML",
                     epilog="Copyright 2024, Jon Hill"
                     )
    parser.add_argument('--topography',
                    required=True,
                    help="topographic raster file")
    parser.add_argument('--geology',
                    required=True,
                    help="geology raster file")
    parser.add_argument('--landcover',
                    required=True,
                    help="landcover raster file")
    parser.add_argument('--faults',
                    required=True,
                    help="fault location shapefile")
    parser.add_argument("landslides",
                    help="the landslide location shapefile")
    parser.add_argument("output",
                    help="the output raster file")
    parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help="Print progress")

    args = parser.parse_args()

    topo = rasterio.open(args.topography)
    geo = rasterio.open(args.geology)
    landc = rasterio.open(args.landcover)
    faultshapefile = gpd.read_file(args.faults)
    landslideshapefile = gpd.read_file(args.landslides)

    # Collecting required files
    topo_raster = "data/AW3D30.tif"
    geo_raster = "data/Geology.tif"
    lc_raster = "data/Lancover.tif"
    faults_shp = "data/Confirmed_faults.shp"
    landslides_shp = "data/landslides.shp"
    landslides = gpd.read_file(landslides_shp)
    
    # Create a slope raster and distance from faults
    slope, slope_meta = calculate_slope(topo_raster)
    dist_fault, dist_meta = calculate_distance_from_faults(faults_shp, topo_raster)
    
    # Convert rasters to numpy arrays matching the template raster
    topo_data, topo_meta = convert_to_rasterio(topo_raster, topo_raster)
    geo_data, geo_meta = convert_to_rasterio(geo_raster, topo_raster)
    lc_data, lc_meta = convert_to_rasterio(lc_raster, topo_raster)
    
    # Create a dataframe
    data_df = create_dataframe(topo_data, geo_data, lc_data, dist_fault, slope, landslides, landslides)
    
if __name__ == '__main__':
    
    main()