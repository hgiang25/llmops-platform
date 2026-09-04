import time
# pyrefly: ignore [missing-import]
import schedule
import logging
from datetime import datetime
import sys
import os

# Ensure the root project is in sys.path so we can import from mlops
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlops.pipeline import MLOpsPipeline

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("data/reports/auto_scheduler.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def job_run_mlops_pipeline():
    logging.info("="*50)
    logging.info("WAKING UP: Running scheduled MLOps Pipeline checks...")
    logging.info("="*50)
    
    pipeline = MLOpsPipeline()
    
    try:
        # Run the full pipeline. 
        # By default generate=False is what we want for real-world (read from raw),
        # but for demonstration we'll set generate=True so it creates synthetic data
        # to trigger the drift logic. In production, change to generate=False.
        result = pipeline.run_full_pipeline(
            force_retrain=False, 
            generate_data=True 
        )
        
        status = result.get("status", "unknown")
        
        if status == "completed":
            logging.info("Pipeline executed successfully.")
            # Let's check if retraining happened
            if result.get("retrained"):
                logging.info(f"Retraining occurred! New model recommendation: {result.get('comparison', {}).get('recommendation')}")
            else:
                logging.info("No drift detected. System is stable.")
        else:
            logging.error(f"Pipeline failed: {result.get('error')}")
            
    except Exception as e:
        logging.error(f"Critical error during scheduled pipeline execution: {e}")
        
    logging.info("Going back to sleep until next scheduled run...\n")

def start_scheduler():
    logging.info("Starting Automated MLOps Scheduler Daemon...")
    
    # Schedule the job. 
    # For demonstration purposes, we schedule it to run every 1 minute.
    # In production, you would do: schedule.every().day.at("02:00").do(job_run_mlops_pipeline)
    schedule.every().day.at("02:00").do(job_run_mlops_pipeline)

    
    logging.info("Scheduler is active. Waiting for the next tick (1 minute)...")
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    start_scheduler()
