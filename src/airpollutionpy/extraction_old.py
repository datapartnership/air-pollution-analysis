"""
NO2 Analysis and Export using Google Earth Engine

This script processes NO2 data from the Sentinel-5P satellite using Google Earth Engine (GEE). 
It includes functions for computing daily and monthly NO2 averages at native resolution and 
aggregated by administrative regions. Data can be exported to Google Drive or Google Cloud Storage.

Dependencies:
- earthengine-api
- geemap
- pandas
- geopandas
"""

# Authenticate and initialize Earth Engine
import ee
import geemap
import pandas as pd
import time
from datetime import datetime, timedelta
import geopandas as gpd

ee.Authenticate()
ee.Initialize()

def get_no2_collection():
    """
    Retrieve the NO2 ImageCollection from Sentinel-5P.
    """
    return ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_NO2").select('NO2_column_number_density')

def split_dates_into_chunks(start_date, end_date, chunk_size=10):
    """
    Split a date range into chunks of specified size.
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    current = start
    chunks = []
    
    while current < end:
        next_chunk = current + timedelta(days=chunk_size)
        chunks.append((current.strftime('%Y-%m-%d'), min(next_chunk, end).strftime('%Y-%m-%d')))
        current = next_chunk
    
    return chunks

def process_no2_data(start_date, end_date, aoi, admin_regions=None, temporal_resolution='daily', spatial_resolution='native'):
    """
    Generalized function to process NO2 data for a given time range.
    
    Parameters:
    - start_date (str): Start date in YYYY-MM-DD format.
    - end_date (str): End date in YYYY-MM-DD format.
    - aoi (ee.Geometry): Area of Interest.
    - admin_regions (ee.FeatureCollection, optional): If provided, computes NO2 over administrative regions.
    - temporal_resolution (str): 'daily' or 'monthly'.
    - spatial_resolution (str): 'native' or 'admin'.
    
    Returns:
    - ee.FeatureCollection: Processed NO2 data.
    """
    NO2Collection = get_no2_collection()
    final_collection = ee.FeatureCollection([])
    
    if temporal_resolution == 'daily':
        date_chunks = split_dates_into_chunks(start_date, end_date)
        
        for chunk_start_date, chunk_end_date in date_chunks:
            current_date = datetime.strptime(chunk_start_date, '%Y-%m-%d')
            end_date_dt = datetime.strptime(chunk_end_date, '%Y-%m-%d')
            
            while current_date <= end_date_dt:
                current_date_str = current_date.strftime('%Y-%m-%d')
                ee_date = ee.Date(current_date_str)
                filtered_day = NO2Collection.filterDate(ee_date, ee_date.advance(1, 'day')).mean()
                
                if spatial_resolution == 'admin' and admin_regions:
                    sampled_pixels = filtered_day.reduceRegions(
                        collection=admin_regions, reducer=ee.Reducer.mean(), scale=1000, crs='EPSG:4326')
                else:
                    sampled_pixels = filtered_day.sample(
                        region=aoi, scale=1000, projection='EPSG:4326', geometries=True)
                
                sampled_pixels = sampled_pixels.map(lambda feature: feature.set('date', current_date_str))
                final_collection = final_collection.merge(sampled_pixels)
                
                current_date += timedelta(days=1)
    
    elif temporal_resolution == 'monthly':
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_date = start_date_dt
        
        while current_date <= end_date_dt:
            year, month = current_date.year, current_date.month
            ee_start_date = ee.Date(f"{year}-{month:02d}-01")
            ee_end_date = ee_start_date.advance(1, 'month')
            
            filtered_month = NO2Collection.filterDate(ee_start_date, ee_end_date).mean()
            
            if spatial_resolution == 'admin' and admin_regions:
                sampled_pixels = filtered_month.reduceRegions(
                    collection=admin_regions, reducer=ee.Reducer.mean(), scale=1000, crs='EPSG:4326')
            else:
                sampled_pixels = filtered_month.sample(
                    region=aoi, scale=1000, projection='EPSG:4326', geometries=True)
            
            sampled_pixels = sampled_pixels.map(lambda feature: feature.set('date', f"{year}-{month:02d}-01"))
            final_collection = final_collection.merge(sampled_pixels)
            
            current_date = (current_date.replace(day=1) + timedelta(days=32)).replace(day=1)
    
    return final_collection

def export_no2_data(collection, description, output_file, export_type, destination):
    """
    Exports processed NO2 data to Google Drive or Google Cloud Storage.
    
    Parameters:
    - collection (ee.FeatureCollection): NO2 data to export.
    - description (str): Description for the export task.
    - output_file (str): Output filename.
    - export_type (str): 'Drive' or 'GCS'.
    - destination (str): Google Drive folder or GCS bucket.
    """
    if export_type == 'GCS':
        export_task = ee.batch.Export.table.toCloudStorage(
            collection=collection,
            description=description,
            bucket=destination,
            fileNamePrefix=output_file.replace('.csv', ''),
            fileFormat="CSV")
    else:
        export_task = ee.batch.Export.table.toDrive(
            collection=collection,
            description=description,
            folder=destination,
            fileNamePrefix=output_file.replace('.csv', ''),
            fileFormat="CSV")
    
    export_task.start()
    
    while export_task.active():
        print(f'Exporting... Task status:', export_task.status()['state'])
        time.sleep(30)
    
    print(f"Export completed: {output_file}" if export_task.status()['state'] == 'COMPLETED' else f"Export failed: {export_task.status()}")

aoi = geemap.geopandas_to_ee(gpd.read_file('data/boundaries/djibouti-addis/ethiopia_adm3_djibouti_addis_outline.shp'))

start_date = '2024-10-01'
end_date = '2024-12-31'
gcs_bucket = 'datalab-air-pollution'

data = process_no2_data(start_date, end_date, aoi, temporal_resolution='daily', spatial_resolution='native')

export_no2_data(data, 
                "NO2_sample_djibouti_addis", 
                "no2_djibouti_addis.csv", 
                "GCS", 
                gcs_bucket)
