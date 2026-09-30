import os
import sys
import shutil
import csv
import time
import json
import logging
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
import httpx
import asyncio

# --- ALASKA CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_DATA_DIR = os.path.join(ROOT_DIR, "full_data")
LOG_CSV_PATH = os.path.join(ROOT_DIR, "alaska_ingest_logs.csv")
MASTER_DB_PATH = os.path.join(FULL_DATA_DIR, "alaska_master_database.csv")
TENSOR_DIR = os.path.join(FULL_DATA_DIR, "tensor_injections")

# CHUNKING CONFIGURATION (Optimized for RAM conservation)
CHUNK_SIZE = 10000 

class AlaskaLogger:
    """CSV-Based Logger with Fail-Safe capabilities."""
    def __init__(self, log_path):
        self.log_path = log_path
        self._init_log_file()

    def _init_log_file(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Level', 'Module', 'Message', 'Traceback'])

    def log(self, level, module, message, tb=""):
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] {level} | {module} | {message}")
        try:
            with open(self.log_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, level, module, message, tb])
        except Exception as e:
            print(f"CRITICAL LOGGING ERROR: {e}")

    def info(self, module, msg): self.log("INFO", module, msg)
    def warning(self, module, msg): self.log("WARNING", module, msg)
    def error(self, module, msg, tb=""): self.log("ERROR", module, msg, tb)

logger = AlaskaLogger(LOG_CSV_PATH)


class AlaskaIngestionEngine:
    """JARVIS-Level Ingestion Agent for Sentinel V7.0"""
    
    def __init__(self):
        self.logger = logger
        self.session = httpx.AsyncClient(timeout=30.0)
        self.ingested_data_chunks = []

    def purge_system(self):
        """Purge conflicting files to ensure a clean slate."""
        self.logger.info("SYS_PURGE", "Initiating conflict resolution purge...")
        try:
            if os.path.exists(FULL_DATA_DIR):
                shutil.rmtree(FULL_DATA_DIR)
                self.logger.info("SYS_PURGE", f"Purged existing directory: {FULL_DATA_DIR}")
            
            os.makedirs(FULL_DATA_DIR, exist_ok=True)
            os.makedirs(TENSOR_DIR, exist_ok=True)
            self.logger.info("SYS_PURGE", "Clean directories established.")
            
            # Initialize the Master CSV with headers
            headers = [
                "node_id", "timestamp", "lat", "lon", 
                "glofas_discharge", "om_rainfall", "om_soil_moisture", "om_runoff",
                "nasa_gpm_precip", "cwc_barrage_level", "sar_vh_backscatter",
                "hydro_elevation", "osm_road_density", "gfd_historical_risk",
                "jrc_water_transition", "emdat_vulnerability"
            ]
            pd.DataFrame(columns=headers).to_csv(MASTER_DB_PATH, index=False)
            
        except Exception as e:
            self.logger.error("SYS_PURGE", f"Failed to purge system.", traceback.format_exc())
            sys.exit(1) # Critical failure

    async def fetch_open_meteo(self, lat, lon):
        """Source 2: Open-Meteo Flood & Weather API"""
        try:
            w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=precipitation&hourly=soil_moisture_0_to_7cm"
            f_url = f"https://flood-api.open-meteo.com/v1/flood?latitude={lat}&longitude={lon}&daily=river_discharge_max"
            
            w_res, f_res = await asyncio.gather(self.session.get(w_url), self.session.get(f_url))
            w_data, f_data = w_res.json(), f_res.json()
            
            return {
                "om_rainfall": w_data.get('current', {}).get('precipitation', 0.0),
                "om_soil_moisture": w_data.get('hourly', {}).get('soil_moisture_0_to_7cm', [0.0])[0],
                "om_runoff": w_data.get('current', {}).get('precipitation', 0.0) * 0.42, # Synthetic correlation
                "glofas_discharge": f_data.get('daily', {}).get('river_discharge_max', [0.0])[0] # GloFAS proxy
            }
        except Exception as e:
            self.logger.error("API_OPEN_METEO", f"Failed to fetch for {lat},{lon}", traceback.format_exc())
            return {"om_rainfall": 0, "om_soil_moisture": 0, "om_runoff": 0, "glofas_discharge": 0}

    async def fetch_overpass_roads(self, lat, lon):
        """Source 7: Overpass API (OSM) for infrastructure density"""
        try:
            # Bounding box of ~1km around point
            bbox = f"{lat-0.01},{lon-0.01},{lat+0.01},{lon+0.01}"
            query = f"[out:json][timeout:10];way['highway']({bbox});out count;"
            res = await self.session.get(f"https://overpass-api.de/api/interpreter?data={query}")
            data = res.json()
            count = int(data.get('elements', [{'tags': {}}])[0].get('tags', {}).get('nodes', 10))
            return {"osm_road_density": count}
        except Exception as e:
            self.logger.warning("API_OVERPASS", f"Failed to fetch OSM data: {str(e)}")
            return {"osm_road_density": 0}

    def simulate_authenticated_scientific_sources(self, lat, lon):
        """
        Sources 1, 3, 4, 5, 6, 8, 9, 10
        (NASA GPM, EE Sentinel-1, EM-DAT, HydroSHEDS, JRC)
        These require auth/heavy local rasters. Simulating extraction logic for pipeline stability.
        """
        # In a full deployment, this replaces with local raster reads via rasterio/GDAL
        return {
            "nasa_gpm_precip": np.random.uniform(0.0, 150.0), # mm/hr
            "cwc_barrage_level": np.random.uniform(105.0, 112.0), # meters
            "sar_vh_backscatter": np.random.uniform(10.0, 60.0), # dB
            "hydro_elevation": np.random.uniform(50.0, 130.0), # ASL meters
            "gfd_historical_risk": np.random.uniform(0.0, 1.0), # Probability
            "jrc_water_transition": np.random.choice([0, 1, 2]), # 0=Land, 1=Seasonal, 2=Permanent
            "emdat_vulnerability": np.random.uniform(0.1, 0.9) # Demographic risk multiplier
        }

    async def build_data_chunk(self, chunk_id, nodes):
        """Generates a chunk of data concurrently to prevent memory overflow."""
        chunk_data = []
        for node in nodes:
            try:
                # 1. Fetch Live APIs
                om_data = await self.fetch_open_meteo(node['lat'], node['lon'])
                osm_data = await self.fetch_overpass_roads(node['lat'], node['lon'])
                
                # 2. Extract Auth/Raster Data
                sci_data = self.simulate_authenticated_scientific_sources(node['lat'], node['lon'])
                
                # 3. Merge
                merged = {
                    "node_id": node['id'],
                    "timestamp": datetime.now().isoformat(),
                    "lat": node['lat'],
                    "lon": node['lon'],
                    **om_data,
                    **osm_data,
                    **sci_data
                }
                chunk_data.append(merged)
                
            except Exception as e:
                self.logger.error("CHUNK_BUILDER", f"Error on node {node['id']}", traceback.format_exc())

        # Optimize: Write chunk immediately to CSV and flush RAM
        df = pd.DataFrame(chunk_data)
        df.to_csv(MASTER_DB_PATH, mode='a', header=False, index=False)
        self.logger.info("DB_IO", f"Chunk {chunk_id} committed to Master DB. Rows: {len(df)}")
        
        return df # Return for immediate tensor injection

    def generate_tensor_injection(self, df_chunk, chunk_id):
        """
        Converts the raw Pandas chunk into ALASKA's Neural Tensor format.
        Shape: (Samples, TimeSteps=1, Features=10) for Bi-LSTM integration.
        """
        self.logger.info("TENSOR_GEN", f"Injecting tensors for Chunk {chunk_id}...")
        try:
            # Select the 10 core features for the new model
            feature_cols = [
                "hydro_elevation", "osm_road_density", "om_rainfall", "om_runoff",
                "om_soil_moisture", "glofas_discharge", "emdat_vulnerability", 
                "sar_vh_backscatter", "nasa_gpm_precip", "cwc_barrage_level"
            ]
            
            raw_matrix = df_chunk[feature_cols].to_numpy(dtype=np.float32)
            
            # Reshape for Sequential Models (Bi-LSTM / Transformers)
            # [Samples, Time_Steps, Features]
            tensor_matrix = raw_matrix.reshape((raw_matrix.shape[0], 1, raw_matrix.shape[1]))
            
            # Save Tensor to disk for rapid dataloader access during training
            tensor_path = os.path.join(TENSOR_DIR, f"alaska_tensors_chunk_{chunk_id}.npy")
            np.save(tensor_path, tensor_matrix)
            
            self.logger.info("TENSOR_GEN", f"Tensor {tensor_matrix.shape} saved successfully: {tensor_path}")
            
        except Exception as e:
            self.logger.error("TENSOR_GEN", "Tensor injection failed.", traceback.format_exc())

    async def execute_pipeline(self):
        """Main execution loop for ALASKA's data gathering."""
        self.logger.info("CORE", "ALASKA Ingestion Engine Online.")
        self.purge_system()

        # Simulate generating 2,025 Tactical Nodes for Lucknow Division
        total_nodes = 2025
        self.logger.info("CORE", f"Mapping {total_nodes} tactical nodes...")
        
        nodes = []
        for i in range(total_nodes):
            nodes.append({
                "id": f"LKO-{i+1:04d}",
                "lat": np.random.uniform(26.60, 27.10),
                "lon": np.random.uniform(80.70, 81.20)
            })

        # Process in chunks to prevent API rate limits and RAM saturation
        chunks = [nodes[i:i + CHUNK_SIZE] for i in range(0, len(nodes), CHUNK_SIZE)]
        
        for idx, chunk in enumerate(chunks):
            self.logger.info("PIPELINE", f"Processing Chunk {idx+1}/{len(chunks)}...")
            
            # Build and Save CSV Data
            df_chunk = await self.build_data_chunk(idx+1, chunk)
            
            # Inject into Tensors
            self.generate_tensor_injection(df_chunk, idx+1)
            
            # Rate limit protection for Live APIs
            time.sleep(2)

        self.logger.info("CORE", "ALASKA Pipeline Execution Complete. Awaiting Model Training.")
        await self.session.aclose()


if __name__ == "__main__":
    print("\n" + "="*50)
    print(" /// ALASKA DATA PIPELINE INITIALIZED ///")
    print("="*50 + "\n")
    
    agent = AlaskaIngestionEngine()
    asyncio.run(agent.execute_pipeline())