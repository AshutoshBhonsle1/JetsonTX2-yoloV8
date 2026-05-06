import cv2
import os
import glob
import time
import threading
from queue import Queue
import numpy as np

# PyCUDA and TensorRT are required for bare-metal Pascal execution
import pycuda.driver as cuda
# REMOVED pycuda.autoinit to prevent multi-threading crashes
import tensorrt as trt

# ==========================================
# ARCHITECTURE CONFIGURATION
# ==========================================
ENGINE_PATH = "yolov8n.engine"
DATASET_PATH = "replaced_missing_labels_thermal_dataset/test/images/"
IMG_SIZE = 640

# The RAM Savior: Bounded queue prevents SD Card SWAP overflow
frame_queue = Queue(maxsize=30)
loading_complete = threading.Event()

# ==========================================
# THREAD 1: THE PRODUCER (CPU / I/O)
# ==========================================
def fetch_and_preprocess(img_folder):
    print("[Thread 1] Booting Thermal Image Fetcher...")
    img_paths = glob.glob(os.path.join(img_folder, "*.jpg"))
    
    if not img_paths:
        print(f"[ERROR] No images found in {img_folder}")
        loading_complete.set()
        return

    for path in img_paths:
        img = cv2.imread(path)
        if img is not None:
            # Native YOLOv8 Preprocessing
            resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            inp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = inp.transpose(2, 0, 1) # HWC to CHW
            inp = np.expand_dims(inp, axis=0) # Add batch dimension
            inp = np.ascontiguousarray(inp) # Crucial for PyCUDA memory pointers
            
            # ⏱️ TIMESTAMP 1: The exact moment the frame enters the pipeline
            frame_start_time = time.perf_counter()
            
            # Push into the tunnel (Pauses automatically if tunnel hits 30 frames)
            frame_queue.put((os.path.basename(path), inp, frame_start_time))
            
    print("[Thread 1] SD Card reading complete. Tunnel closing.")
    loading_complete.set()


# ==========================================
# TENSORRT MEMORY ALLOCATOR HELPER
# ==========================================
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # Append the device buffer to device bindings
        bindings.append(int(device_mem))
        
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
            
    return inputs, outputs, bindings, stream


# ==========================================
# THREAD 2: THE CONSUMER (GPU)
# ==========================================
def run_inference():
    print("[Thread 2] Booting TensorRT Engine & GPU Context...")
    
    # ---------------------------------------------------------
    # MANUALLY BUILD THE GPU CONTEXT FOR THIS SPECIFIC THREAD
    # ---------------------------------------------------------
    cuda.init()
    cuda_device = cuda.Device(0)
    cuda_context = cuda_device.make_context()
    
    try:
        # 1. Initialize TensorRT
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace="")
        
        with open(ENGINE_PATH, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            
        context = engine.create_execution_context()
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        
        processed_count = 0
        latency_records_ms = []
        
        # Wait for the producer to buffer the first few frames
        while frame_queue.qsize() < 5 and not loading_complete.is_set():
            time.sleep(0.1)
            
        print("\n--- STARTING ASYNC INFERENCE ---")
        pipeline_start = time.perf_counter()
        
        while not loading_complete.is_set() or not frame_queue.empty():
            if not frame_queue.empty():
                # Unpack the frame AND its original timestamp
                filename, frame_data, frame_start_time = frame_queue.get()
                
                # Copy image data to PyCUDA host buffer
                np.copyto(inputs[0]['host'], frame_data.ravel())
                
                # Transfer to GPU (Host to Device)
                cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
                
                # Execute Neural Network Math
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                
                # Transfer results back to CPU (Device to Host)
                cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
                
                # Synchronize the stream to ensure completion
                stream.synchronize()
                
                # ⏱️ TIMESTAMP 2: The exact moment the GPU finishes this frame
                frame_end_time = time.perf_counter()
                
                # Calculate Latency
                true_latency_ms = (frame_end_time - frame_start_time) * 1000
                latency_records_ms.append(true_latency_ms)
                
                processed_count += 1
                
                # Optional: Print every 100 frames to monitor progress
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} frames... Current Latency: {true_latency_ms:.1f}ms")

        # ==========================================
        # FINAL ACADEMIC METRICS CALCULATION
        # ==========================================
        total_time = time.perf_counter() - pipeline_start
        fps = processed_count / total_time
        avg_latency_ms = sum(latency_records_ms) / len(latency_records_ms)
        
        print("\n=====================================================")
        print("🏆 FINAL MULTI-THREADED PERFORMANCE METRICS")
        print("=====================================================")
        print(f"Total Thermal Images: {processed_count}")
        print(f"Total Pipeline Time:  {total_time:.2f} seconds")
        print("-----------------------------------------------------")
        print(f"Throughput (FPS):     {fps:.2f} Frames Per Second")
        print(f"Average Latency:      {avg_latency_ms:.2f} Milliseconds")
        print("=====================================================")

    finally:
        # ---------------------------------------------------------
        # ALWAYS CLEAN UP THE GPU MEMORY WHEN THE THREAD DIES
        # ---------------------------------------------------------
        cuda_context.pop()


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Ensure hardware limits are removed before running
    # sudo nvpmodel -m 0 && sudo jetson_clocks
    
    producer = threading.Thread(target=fetch_and_preprocess, args=(DATASET_PATH,))
    consumer = threading.Thread(target=run_inference)
    
    producer.start()
    consumer.start()
    
    producer.join()
    consumer.join()
    print("Pipeline Shutdown Complete.")
