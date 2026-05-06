import cv2
import os
import glob
import time
import threading
from queue import Queue
import numpy as np

# 1. The RAM Savior: A strictly limited queue holding a max of 30 frames
frame_queue = Queue(maxsize=30)
# A flag to let the GPU know when the SD card is out of images
loading_complete = threading.Event()

# ==========================================
# THREAD 1: THE PRODUCER (CPU / I/O)
# ==========================================
def fetch_and_preprocess(img_folder):
    print("[Thread 1] Booting Image Fetcher...")
    img_paths = glob.glob(os.path.join(img_folder, "*.jpg"))
    
    for path in img_paths:
        img = cv2.imread(path)
        if img is not None:
            # Standard YOLOv8 Preprocessing
            resized = cv2.resize(img, (640, 640))
            inp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = inp.transpose(2, 0, 1) # HWC to CHW
            inp = np.expand_dims(inp, axis=0) # Add batch dimension
            
            # Shove it into the tunnel. 
            # If the tunnel has 30 frames, this thread automatically pauses!
            frame_queue.put((os.path.basename(path), inp))
            
    print("[Thread 1] SD Card reading complete. Tunnel closing.")
    loading_complete.set()

# ==========================================
# THREAD 2: THE CONSUMER (GPU / TensorRT)
# ==========================================
def run_inference():
    print("[Thread 2] Booting TensorRT Engine...")
    
    # ---------------------------------------------------------
    # TODO: Drop your existing TensorRT engine initialization, 
    # Context creation, and PyCUDA memory allocation code here.
    # ---------------------------------------------------------
    
    processed_count = 0
    
    # Wait for the first few frames to enter the tunnel before starting the timer
    while frame_queue.qsize() < 5 and not loading_complete.is_set():
        time.sleep(0.1)
        
    print("\n--- STARTING ASYNC INFERENCE ---")
    start_time = time.perf_counter()
    
    # Keep running as long as the tunnel isn't empty, or Thread 1 is still working
    while not loading_complete.is_set() or not frame_queue.empty():
        if not frame_queue.empty():
            # Yank the next frame out of the tunnel
            filename, frame_data = frame_queue.get()
            
            # ---------------------------------------------------------
            # TODO: Drop your existing session.execute_async_v2() 
            # and bounding box parsing code here.
            # ---------------------------------------------------------
            
            processed_count += 1
            
    # Calculate True Pipeline FPS
    total_time = time.perf_counter() - start_time
    fps = processed_count / total_time
    
    print("\n=============================================")
    print("🏆 MULTI-THREADED PIPELINE RESULTS")
    print("=============================================")
    print(f"Frames Processed: {processed_count}")
    print(f"Total Time:       {total_time:.2f} seconds")
    print(f"Pipeline Speed:   {fps:.2f} FPS")
    print("=============================================")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    dataset_path = "replaced_missing_labels_thermal_dataset/test/images/"
    
    # Initialize both threads
    producer = threading.Thread(target=fetch_and_preprocess, args=(dataset_path,))
    consumer = threading.Thread(target=run_inference)
    
    # Start the engine
    producer.start()
    consumer.start()
    
    # Wait for both to finish cleanly
    producer.join()
    consumer.join()
    print("Pipeline Shutdown Complete.")
