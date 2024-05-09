import argparse

# Your job is to write the definitions below so that when you run the code, it gives an output. 
# Currently, if you run python terrain_analysis.py --topography data/AW3D30.tif --geology data/geology_raster.tif --landcover data/Landcover.tif --faults data/Confirmed_faults.shp data/landslides.shp probability.tif
# in the terminal, nothing happens because the definitions below to open the files doesn't exist!!



def convert_to_rasterio(raster_data, template_raster):
    #write stuff here
    return


def extract_values_from_raster(raster, shape_object):
    #write stuff here
    return


def make_classifier(x, y, verbose=False):
    #write stuff here
    return

def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
    #write stuff here
    return

def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides):
    #write stuff here
    return


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


if __name__ == '__main__':
    main()
