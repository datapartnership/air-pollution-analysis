"""
NO2 Analysis and Export using Google Earth Engine

This script processes NO2 data from the Sentinel-5P satellite using Google Earth Engine (GEE). 
It includes functions for computing daily, monthly, and annual NO2 averages at native resolution and 
aggregated by administrative regions. Data can be exported to a DataFrame, Google Drive, or Google Cloud Storage.

Dependencies:
- earthengine-api
- geemap
- pandas
- geopandas
- tqdm
- numpy
"""

# Authenticate and initialize Earth Engine
import ee
import geemap
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import geopandas as gpd
from tqdm import tqdm
import os
import gc
import argparse
import warnings
import traceback
warnings.filterwarnings('ignore')

def initialize_earth_engine():
    """
    Authenticate and initialize Earth Engine.
    """
    try:
        ee.Initialize()
        print("Earth Engine already initialized")
    except Exception:
        ee.Authenticate()
        ee.Initialize()
        print("Earth Engine initialized")

def get_no2_collection(collection_type="OFFL"):
    """
    Retrieve the NO2 ImageCollection from Sentinel-5P.
    
    Parameters:
    - collection_type (str): "NRTI" for Near Real-Time or "OFFL" for Offline (default).
    
    Returns:
    - ee.ImageCollection: Selected NO2 collection
    """
    valid_types = ["NRTI", "OFFL"]
    if collection_type not in valid_types:
        raise ValueError(f"Collection type must be one of {valid_types}")
    
    collection_id = f"COPERNICUS/S5P/{collection_type}/L3_NO2"
    return ee.ImageCollection(collection_id).select('NO2_column_number_density')

def split_dates_into_chunks(start_date, end_date, chunk_size=10):
    """
    Split a date range into chunks of specified size to manage memory usage.
    
    Parameters:
    - start_date (str): Start date in YYYY-MM-DD format.
    - end_date (str): End date in YYYY-MM-DD format.
    - chunk_size (int): Number of days per chunk.
    
    Returns:
    - list: List of tuples containing (chunk_start_date, chunk_end_date).
    """
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current = start
        chunks = []
        
        while current < end:
            next_chunk = current + timedelta(days=chunk_size)
            chunks.append((current.strftime('%Y-%m-%d'), min(next_chunk, end).strftime('%Y-%m-%d')))
            current = next_chunk
        
        return chunks
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected YYYY-MM-DD: {e}")

def process_no2_data(start_date, end_date, aoi, admin_regions=None, 
                    temporal_resolution='daily', spatial_resolution='native',
                    collection_type="OFFL"):
    """
    Process NO2 data for a given time range.
    
    Parameters:
    - start_date (str): Start date in YYYY-MM-DD format.
    - end_date (str): End date in YYYY-MM-DD format.
    - aoi (ee.Geometry): Area of Interest.
    - admin_regions (ee.FeatureCollection, optional): Administrative regions.
    - temporal_resolution (str): 'daily', 'monthly', or 'annual'.
    - spatial_resolution (str): 'native' or 'admin'.
    - collection_type (str): "NRTI" or "OFFL".
    
    Returns:
    - ee.FeatureCollection: Processed NO2 data.
    """
    # Validate inputs
    valid_temporal = ['daily', 'monthly', 'annual']
    valid_spatial = ['native', 'admin']
    
    if temporal_resolution not in valid_temporal:
        raise ValueError(f"Temporal resolution must be one of {valid_temporal}")
    
    if spatial_resolution not in valid_spatial:
        raise ValueError(f"Spatial resolution must be one of {valid_spatial}")
    
    if spatial_resolution == 'admin' and admin_regions is None:
        raise ValueError("Admin regions must be provided when spatial_resolution is 'admin'")
    
    # Print info about processing
    print(f"Processing NO2 data from {start_date} to {end_date}")
    print(f"Temporal resolution: {temporal_resolution}")
    print(f"Spatial resolution: {spatial_resolution}")
    print(f"Collection type: {collection_type}")
    
    # Check admin regions if used
    if spatial_resolution == 'admin' and admin_regions is not None:
        try:
            admin_count = admin_regions.size().getInfo()
            print(f"Admin regions count: {admin_count}")
            if admin_count == 0:
                raise ValueError("Admin regions collection is empty")
        except Exception as e:
            print(f"Warning: Could not verify admin regions: {e}")
    
    NO2Collection = get_no2_collection(collection_type)
    
    # Check data availability
    try:
        available_images = NO2Collection.filterDate(start_date, end_date).size().getInfo()
        print(f"Number of available images in date range: {available_images}")
        if available_images == 0:
            print("Warning: No images found in the specified date range")
            # Return empty collection instead of proceeding
            return ee.FeatureCollection([])
    except Exception as e:
        print(f"Warning: Could not verify data availability: {e}")
    
    final_collection = ee.FeatureCollection([])
    
    if temporal_resolution == 'daily':
        # Process data in chunks to manage memory
        date_chunks = split_dates_into_chunks(start_date, end_date)
        
        for chunk_start_date, chunk_end_date in tqdm(date_chunks, desc="Processing daily chunks"):
            filtered_chunk = NO2Collection.filterDate(chunk_start_date, chunk_end_date)
            
            # Check if we have data for this chunk
            chunk_size = filtered_chunk.size().getInfo()
            if chunk_size == 0:
                print(f"No data available for chunk {chunk_start_date} to {chunk_end_date}. Skipping.")
                continue
                
            # Calculate mean for the chunk
            chunk_mean = filtered_chunk.mean()
            
            if spatial_resolution == 'admin' and admin_regions:
                # Reduce to admin regions
                try:
                    reduced_chunk = chunk_mean.reduceRegions(
                        collection=admin_regions, 
                        reducer=ee.Reducer.mean(), 
                        scale=1000, 
                        crs='EPSG:4326'
                    )
                    
                    # Add date information
                    reduced_chunk = reduced_chunk.map(lambda feature: 
                        feature.set('start_date', chunk_start_date)
                               .set('end_date', chunk_end_date)
                               .set('temporal_res', 'daily')
                    )
                    
                    final_collection = final_collection.merge(reduced_chunk)
                except Exception as e:
                    print(f"Error processing admin regions for chunk {chunk_start_date} to {chunk_end_date}: {e}")
            else:
                # Sample at native resolution
                try:
                    sampled_chunk = chunk_mean.sample(
                        region=aoi, 
                        scale=1000, 
                        projection='EPSG:4326', 
                        geometries=True
                    )
                    
                    # Add date information
                    sampled_chunk = sampled_chunk.map(lambda feature: 
                        feature.set('start_date', chunk_start_date)
                               .set('end_date', chunk_end_date)
                               .set('temporal_res', 'daily')
                    )
                    
                    final_collection = final_collection.merge(sampled_chunk)
                except Exception as e:
                    print(f"Error sampling at native resolution for chunk {chunk_start_date} to {chunk_end_date}: {e}")
    
    elif temporal_resolution == 'monthly':
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_date = start_date_dt.replace(day=1)  # Start at first day of month
        
        while current_date <= end_date_dt:
            year, month = current_date.year, current_date.month
            next_month = datetime(year + (month // 12), (month % 12) + 1, 1)
            
            start_month_str = f"{year}-{month:02d}-01"
            end_month_str = next_month.strftime('%Y-%m-%d')
            
            print(f"Processing month: {start_month_str} to {end_month_str}")
            
            # Filter the collection for the month
            filtered_month = NO2Collection.filterDate(start_month_str, end_month_str)
            
            # Check if we have data for this month
            month_size = filtered_month.size().getInfo()
            if month_size == 0:
                print(f"No data available for month {start_month_str}. Skipping.")
                current_date = next_month
                continue
                
            # Calculate mean for the month
            monthly_mean = filtered_month.mean()
            
            if spatial_resolution == 'admin' and admin_regions:
                # Reduce to admin regions
                try:
                    reduced_month = monthly_mean.reduceRegions(
                        collection=admin_regions, 
                        reducer=ee.Reducer.mean(), 
                        scale=1000, 
                        crs='EPSG:4326'
                    )
                    
                    # Verify we have results
                    reduced_size = reduced_month.size().getInfo()
                    print(f"Month {start_month_str}: Reduced regions result size: {reduced_size}")
                    
                    if reduced_size > 0:
                        # Add date information
                        reduced_month = reduced_month.map(lambda feature: 
                            feature.set('start_date', start_month_str)
                                   .set('end_date', end_month_str)
                                   .set('temporal_res', 'monthly')
                                   .set('month', month)
                                   .set('year', year)
                        )
                        
                        final_collection = final_collection.merge(reduced_month)
                    else:
                        print(f"Warning: No features in reduced regions result for month {start_month_str}")
                except Exception as e:
                    print(f"Error processing admin regions for month {start_month_str}: {e}")
            else:
                # Sample at native resolution
                try:
                    sampled_month = monthly_mean.sample(
                        region=aoi, 
                        scale=1000, 
                        projection='EPSG:4326', 
                        geometries=True
                    )
                    
                    # Add date information
                    sampled_month = sampled_month.map(lambda feature: 
                        feature.set('start_date', start_month_str)
                               .set('end_date', end_month_str)
                               .set('temporal_res', 'monthly')
                               .set('month', month)
                               .set('year', year)
                    )
                    
                    final_collection = final_collection.merge(sampled_month)
                except Exception as e:
                    print(f"Error sampling at native resolution for month {start_month_str}: {e}")
            
            current_date = next_month
    
    elif temporal_resolution == 'annual':
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Get unique years in the date range
        start_year = start_date_dt.year
        end_year = end_date_dt.year
        
        for year in range(start_year, end_year + 1):
            year_start = max(datetime(year, 1, 1), start_date_dt).strftime('%Y-%m-%d')
            year_end = min(datetime(year, 12, 31), end_date_dt).strftime('%Y-%m-%d')
            
            print(f"Processing year: {year_start} to {year_end}")
            
            # Filter the collection for the year
            filtered_year = NO2Collection.filterDate(year_start, year_end)
            
            # Check if we have data for this year
            year_size = filtered_year.size().getInfo()
            if year_size == 0:
                print(f"No data available for year {year}. Skipping.")
                continue
                
            # Calculate mean for the year
            annual_mean = filtered_year.mean()
            
            if spatial_resolution == 'admin' and admin_regions:
                # Reduce to admin regions
                try:
                    reduced_year = annual_mean.reduceRegions(
                        collection=admin_regions, 
                        reducer=ee.Reducer.mean(), 
                        scale=1000, 
                        crs='EPSG:4326'
                    )
                    
                    # Verify we have results
                    reduced_size = reduced_year.size().getInfo()
                    print(f"Year {year}: Reduced regions result size: {reduced_size}")
                    
                    if reduced_size > 0:
                        # Add date information
                        reduced_year = reduced_year.map(lambda feature: 
                            feature.set('start_date', year_start)
                                   .set('end_date', year_end)
                                   .set('temporal_res', 'annual')
                                   .set('year', year)
                        )
                        
                        final_collection = final_collection.merge(reduced_year)
                    else:
                        print(f"Warning: No features in reduced regions result for year {year}")
                except Exception as e:
                    print(f"Error processing admin regions for year {year}: {e}")
            else:
                # Sample at native resolution
                try:
                    sampled_year = annual_mean.sample(
                        region=aoi, 
                        scale=1000, 
                        projection='EPSG:4326', 
                        geometries=True
                    )
                    
                    # Add date information
                    sampled_year = sampled_year.map(lambda feature: 
                        feature.set('start_date', year_start)
                               .set('end_date', year_end)
                               .set('temporal_res', 'annual')
                               .set('year', year)
                    )
                    
                    final_collection = final_collection.merge(sampled_year)
                except Exception as e:
                    print(f"Error sampling at native resolution for year {year}: {e}")
    
    # Check if we have any results
    try:
        final_size = final_collection.size().getInfo()
        print(f"Final collection size: {final_size} features")
    except Exception as e:
        print(f"Warning: Could not get final collection size: {e}")
    
    return final_collection

def export_no2_data_to_dataframe(collection, output_file=None):
    """
    Exports processed NO2 data to a DataFrame and optionally saves locally.
    
    Parameters:
    - collection (ee.FeatureCollection): NO2 data to export.
    - output_file (str, optional): If provided, save DataFrame to this file.
    
    Returns:
    - pd.DataFrame: DataFrame containing the exported data.
    """
    import os
    import pandas as pd
    import traceback
    
    try:
        # Print debug info
        print("Debug: Starting export to DataFrame")
        
        # Get feature collection as GeoJSON with debug info
        try:
            collection_info = collection.getInfo()
            
            if 'features' not in collection_info:
                print("Debug: No 'features' key in collection_info")
                print(f"Debug: Keys found: {list(collection_info.keys())}")
                return pd.DataFrame()
                
            features = collection_info['features']
            print(f"Debug: Found {len(features)} features")
            
            if len(features) == 0:
                print("Debug: Features list is empty")
                return pd.DataFrame()
        except Exception as e:
            print(f"Debug: Error getting collection info: {e}")
            traceback.print_exc()
            return pd.DataFrame()
        
        # Extract properties and geometry with careful handling
        data = []
        for i, feature in enumerate(features):
            try:
                # Debug for specific feature if needed
                if i < 2:  # Print details for first two features
                    print(f"Debug: Processing feature {i}")
                    if 'properties' not in feature:
                        print(f"Debug: No properties in feature {i}")
                    if 'geometry' not in feature:
                        print(f"Debug: No geometry in feature {i}")
                    elif feature['geometry'] is None:
                        print(f"Debug: Geometry is None in feature {i}")
                
                # Extract properties
                if 'properties' in feature:
                    properties = feature['properties']
                else:
                    print(f"Warning: Feature {i} has no properties, skipping")
                    continue
                
                # Safely extract coordinates if geometry exists
                if 'geometry' in feature and feature['geometry'] is not None:
                    try:
                        geom_type = feature['geometry'].get('type')
                        coords = feature['geometry'].get('coordinates')
                        
                        if geom_type == 'Point' and coords and isinstance(coords, list) and len(coords) >= 2:
                            properties['longitude'] = coords[0]
                            properties['latitude'] = coords[1]
                        else:
                            # For non-point geometries or incomplete coordinates, skip adding coords
                            if i < 2:  # Debug for first two features
                                print(f"Debug: Not adding coordinates for feature {i}. Type: {geom_type}")
                                if coords:
                                    print(f"Debug: Coordinates structure: {type(coords)}")
                    except (IndexError, TypeError, KeyError) as e:
                        print(f"Warning: Could not extract coordinates for feature {i}: {e}")
                
                data.append(properties)
            except Exception as e:
                print(f"Warning: Error processing feature {i}: {e}")
                continue
        
        if not data:
            print("Warning: No valid data extracted from features")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        print(f"Data successfully converted to DataFrame with {len(df)} rows.")
            
        # Save locally if output file specified
        if output_file:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            df.to_csv(output_file, index=False)
            print(f"Data saved locally to {output_file}")
        
        return df
    except Exception as e:
        print(f"Error converting to DataFrame: {e}")
        traceback.print_exc()
        return pd.DataFrame()  # Return empty DataFrame on error

def export_no2_data_to_gcs(collection, description, file_prefix, destination, admin_code_field='admin_code', admin_codes=None, temporal_resolution='monthly'):
    """
    Exports processed NO2 data to Google Cloud Storage, splitting by regions and time periods.
    
    Parameters:
    - collection (ee.FeatureCollection): NO2 data to export.
    - description (str): Description for the export task.
    - file_prefix (str): Prefix for the exported file name.
    - destination (str): GCS bucket and path (e.g., 'bucket-name/path/to/folder').
    - admin_code_field (str): The property name containing admin region codes.
    - admin_codes (list): Optional list of admin region codes. If None, will use temporal splitting only.
    - temporal_resolution (str): The temporal resolution used in processing ('daily', 'monthly', 'annual').
    
    Returns:
    - list: List of export tasks started.
    """
    import ee
    import re
    
    if not destination:
        raise ValueError("Destination bucket must be provided for GCS export")
    
    # Parse the destination into bucket and path components
    bucket_parts = destination.split('/')
    bucket_name = bucket_parts[0]  # Just the bucket name
    
    # Create the path prefix (everything after bucket)
    path_prefix = '/'.join(bucket_parts[1:]) if len(bucket_parts) > 1 else ''
    
    tasks = []
    
    # Function to sanitize descriptions for Earth Engine export
    def sanitize_description(desc):
        # Replace any invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9.,;:_-]', '_', desc)
        # Ensure it's not longer than 100 characters
        return sanitized[:100]
    
    # Determine time-based splitting (start with years)
    try:
        # Get unique years from the collection - but don't try to get the whole array
        # This avoids the payload size issue
        distinct_years = collection.distinct('year').limit(10).aggregate_array('year').getInfo()
        if not distinct_years or len(distinct_years) == 0:
            # If no 'year' property exists, use a default range
            years = list(range(2018, 2025))
            print(f"No year property found, using default range: {years}")
        else:
            years = sorted(distinct_years)
            # Extend the years list if we hit the limit of 10
            if len(distinct_years) == 10:
                years = list(range(min(years), 2025))
            print(f"Using years: {years}")
    except Exception as e:
        print(f"Error determining years: {e}")
        years = list(range(2018, 2025))  # Fallback to default
    
    # If admin_codes is provided, use spatial + temporal splitting
    if admin_codes and len(admin_codes) > 0:
        print(f"Splitting export by {len(admin_codes)} admin regions and {len(years)} years")
        
        # For each year and admin code, export a separate file
        for year in years:
            year_filter = ee.Filter.eq('year', year)
            year_collection = collection.filter(year_filter)
            
            for admin_code in admin_codes:
                try:
                    # Filter to this specific admin region
                    admin_filter = ee.Filter.eq(admin_code_field, admin_code)
                    admin_collection = year_collection.filter(admin_filter)
                    
                    # Create admin+year specific file prefix
                    if path_prefix:
                        region_file_prefix = f"{path_prefix}/{file_prefix}_{admin_code}_{year}"
                    else:
                        region_file_prefix = f"{file_prefix}_{admin_code}_{year}"
                    
                    # Create a sanitized description
                    task_description = sanitize_description(f"{description}_{admin_code}_{year}")
                    
                    # Create and start export task for this admin+year
                    year_task = ee.batch.Export.table.toCloudStorage(
                        collection=admin_collection,
                        description=task_description,
                        bucket=bucket_name,
                        fileNamePrefix=region_file_prefix,
                        fileFormat="CSV")
                    
                    year_task.start()
                    tasks.append(year_task)
                    print(f"Started export task for {admin_code} in {year}")
                except Exception as e:
                    print(f"Error exporting {admin_code} in {year}: {e}")
                    continue
    else:
        # If no admin_codes provided, use temporal splitting only by year
        print("Using temporal splitting strategy by year")
        
        for year in years:
            try:
                year_filter = ee.Filter.eq('year', year)
                year_collection = collection.filter(year_filter)
                
                # Create year specific file prefix
                if path_prefix:
                    time_file_prefix = f"{path_prefix}/{file_prefix}_{year}"
                else:
                    time_file_prefix = f"{file_prefix}_{year}"
                
                # Create a sanitized description
                task_description = sanitize_description(f"{description}_{year}")
                
                # Create and start export task for this year
                time_task = ee.batch.Export.table.toCloudStorage(
                    collection=year_collection,
                    description=task_description,
                    bucket=bucket_name,
                    fileNamePrefix=time_file_prefix,
                    fileFormat="CSV")
                
                time_task.start()
                tasks.append(time_task)
                print(f"Started export task for {year}")
            except Exception as e:
                print(f"Error exporting {year}: {e}")
                continue
    
    if not tasks:
        print("All splitting strategies failed. Attempting to export the entire collection at once")
        try:
            # Last resort: try exporting the entire collection at once
            if path_prefix:
                file_prefix_full = f"{path_prefix}/{file_prefix}_full"
            else:
                file_prefix_full = f"{file_prefix}_full"
            
            # Create a sanitized description for the fallback
            fallback_description = sanitize_description(f"{description}_full")
            
            full_task = ee.batch.Export.table.toCloudStorage(
                collection=collection,
                description=fallback_description,
                bucket=bucket_name,
                fileNamePrefix=file_prefix_full,
                fileFormat="CSV")
            
            full_task.start()
            tasks.append(full_task)
            print("Started fallback export task for the entire collection")
        except Exception as e:
            print(f"Error in fallback export: {e}")
            print("All export strategies failed.")
            return []
    
    print(f"Started {len(tasks)} export tasks in total")
    return tasks

def export_no2_data_to_drive(collection, description, file_prefix, folder=None):
    """
    Exports processed NO2 data to Google Drive.
    
    Parameters:
    - collection (ee.FeatureCollection): NO2 data to export.
    - description (str): Description for the export task.
    - file_prefix (str): Prefix for the exported file name.
    - folder (str, optional): Google Drive folder. Defaults to "Earth Engine Exports".
    
    Returns:
    - export_task: The export task object.
    """
    import ee
    
    # Default to root folder if none specified
    folder = folder if folder else "Earth Engine Exports"
    
    export_task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=description,
        folder=folder,
        fileNamePrefix=file_prefix,
        fileFormat="CSV")

    export_task.start()
    
    # Monitor task
    print(f"Started export task: {description}")
    task_status = export_task.status()
    print(f"Initial task status: {task_status['state']}")
    
    return export_task

def export_no2_data(collection, description, output_file, export_type='Drive', destination=None, 
                    return_df=False, admin_code_field='admin_code', admin_codes=None):
    """
    Exports processed NO2 data to a DataFrame, Google Drive, or Google Cloud Storage.
    
    Parameters:
    - collection (ee.FeatureCollection): NO2 data to export.
    - description (str): Description for the export task.
    - output_file (str): Output filename.
    - export_type (str): 'DataFrame', 'Drive', or 'GCS'.
    - destination (str): Google Drive folder or GCS bucket.
    - return_df (bool): Whether to return the data as a DataFrame.
    - admin_code_field (str): Field name for admin region codes.
    - admin_codes (list): Optional list of admin region codes for spatial splitting.
    
    Returns:
    - Various: DataFrame, export task, or None depending on export_type and return_df.
    """
    valid_export_types = ['DataFrame', 'Drive', 'GCS']
    if export_type not in valid_export_types:
        raise ValueError(f"Export type must be one of {valid_export_types}")

    # Generate CSV file name without extension for Earth Engine export
    file_prefix = output_file.replace('.csv', '')
    
    # Handle DataFrame export
    if export_type == 'DataFrame' or return_df:
        df = export_no2_data_to_dataframe(collection, output_file if export_type == 'DataFrame' else None)
        if export_type == 'DataFrame':
            return df
    
    # Handle GCS export with admin regions if provided
    if export_type == 'GCS':
        return export_no2_data_to_gcs(
            collection, description, file_prefix, destination, 
            admin_code_field=admin_code_field, admin_codes=admin_codes
        )
    
    # Handle Drive export - you may want to add similar logic for Drive exports
    elif export_type == 'Drive':
        # If admin_codes provided, consider implementing a similar splitting strategy
        # for Drive exports as well
        if admin_codes and len(admin_codes) > 0:
            print("Warning: admin_codes splitting not implemented for Drive exports")
        return export_no2_data_to_drive(collection, description, file_prefix, destination)
    
    # Handle DataFrame with return_df=True flag
    if return_df:
        return df
    
    return None

def main(args):
    """
    Main function to run the NO2 data extraction and processing.
    
    Parameters:
    - args: Command line arguments.
    """
    # Initialize Earth Engine
    initialize_earth_engine()
    
    # Load area of interest
    if args.aoi_file:
        try:
            aoi = geemap.geopandas_to_ee(gpd.read_file(args.aoi_file))
            print(f"Loaded area of interest from {args.aoi_file}")
        except Exception as e:
            print(f"Error loading area of interest: {e}")
            traceback.print_exc()
            return
    else:
        # Default to global if no AOI specified
        aoi = ee.Geometry.Rectangle([-180, -90, 180, 90])
        print("Using global extent as area of interest")
    
    # Load admin regions if needed
    admin_regions = None
    if args.admin_file and args.spatial_resolution == 'admin':
        try:
            admin_regions = geemap.geopandas_to_ee(gpd.read_file(args.admin_file))
            print(f"Loaded admin regions from {args.admin_file}")
        except Exception as e:
            print(f"Error loading admin regions: {e}")
            traceback.print_exc()
            return
    
    print(f"Processing NO2 data from {args.start_date} to {args.end_date}")
    print(f"Temporal resolution: {args.temporal_resolution}")
    print(f"Spatial resolution: {args.spatial_resolution}")
    print(f"Collection type: {args.collection_type}")
    
    # Process the data
    try:
        collection = process_no2_data(
            start_date=args.start_date,
            end_date=args.end_date,
            aoi=aoi,
            admin_regions=admin_regions,
            temporal_resolution=args.temporal_resolution,
            spatial_resolution=args.spatial_resolution,
            collection_type=args.collection_type
        )
        
        # Export the data
        df = export_no2_data(
            collection=collection,
            description=args.description,
            output_file=args.output_file,
            export_type=args.export_type,
            destination=args.destination,
            return_df=args.return_df
        )
        
        # If DataFrame was requested, return it
        if df is not None and args.return_df:
            print(f"DataFrame with {len(df)} rows returned.")
            return df
    except Exception as e:
        print(f"Error in processing or export: {e}")
        traceback.print_exc()
        return None
    
    print("Processing complete!")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Extract and process NO2 data from Sentinel-5P.")
    
#     # Required arguments
#     parser.add_argument("--start_date", required=True, help="Start date in YYYY-MM-DD format")
#     parser.add_argument("--end_date", required=True, help="End date in YYYY-MM-DD format")
#     parser.add_argument("--output_file", required=True, help="Output file name (CSV)")
    
#     # Optional arguments
#     parser.add_argument("--aoi_file", help="Shapefile for area of interest")
#     parser.add_argument("--admin_file", help="Shapefile for administrative boundaries")
#     parser.add_argument("--description", default="NO2_extraction", help="Description for the export task")
#     parser.add_argument("--temporal_resolution", default="monthly", 
#                         choices=["daily", "monthly", "annual"], 
#                         help="Temporal resolution for processing")
#     parser.add_argument("--spatial_resolution", default="native", 
#                         choices=["native", "admin"], 
#                         help="Spatial resolution for processing")
#     parser.add_argument("--collection_type", default="OFFL", 
#                         choices=["NRTI", "OFFL"], 
#                         help="Collection type (NRTI or OFFL)")
#     parser.add_argument("--export_type", default="Drive", 
#                         choices=["DataFrame", "Drive", "GCS"], 
#                         help="Export type")
#     parser.add_argument("--destination", help="Google Drive folder or GCS bucket name")
#     parser.add_argument("--return_df", action="store_true", help="Return DataFrame in addition to export")
    
#     args = parser.parse_args()
    
#     # Run the main function
#     main(args)