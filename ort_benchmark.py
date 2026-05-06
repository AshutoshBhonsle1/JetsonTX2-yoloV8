import cv2
import numpy as np
import onnxruntime as ort
import sys
import os
import time
import glob

def benchmark_ort(onnx_path, img_folder):
    out_dir = "eval_labels_rtdetr"
    os.makedirs(out_dir, exist_ok=True)
    
    img_paths = glob.glob(os.path.join(img_folder, "*.jpg"))
    total_images = len(img_paths)
    if total_images == 0:
        print("No .jpg images found in folder!")
        return

    print(f"Found {total_images} images. Booting ONNX Runtime on GPU...")

    # 1. Initialize ONNX Runtime Session (Force CUDA)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Get model input/output names and shapes
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    
    # RT-DETR is usually exported at 640x640, dynamically fetch it just in case
    input_h, input_w = input_shape[2], input_shape[3]
    if isinstance(input_h, str): # Handle dynamic axes if present
        input_h, input_w = 640, 640

    # 2. Pre-load images into RAM (Bypass SD Card bottleneck)
    print("Pre-loading images into RAM for accurate GPU benchmarking...")
    processed_images = []
    filenames = []
    
    for path in img_paths:
        img = cv2.imread(path)
        if img is not None:
            # Standard YOLO/RT-DETR preprocessing
            resized = cv2.resize(img, (input_w, input_h))
            inp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            inp = inp.astype(np.float32) / 255.0
            inp = inp.transpose(2, 0, 1)  # HWC to CHW
            inp = np.expand_dims(inp, axis=0) # Add batch dimension
            
            processed_images.append(inp)
            filenames.append(os.path.basename(path))

    total_valid = len(processed_images)

    # --- WARMUP PHASE ---
    print("Warming up GPU Clocks (20 frames)...")
    for _ in range(20):
        session.run([output_name], {input_name: processed_images[0]})

    # --- BENCHMARK PHASE ---
    print("\n--- STARTING RT-DETR EVALUATION ---")
    start_time = time.perf_counter()

    for i in range(total_valid):
        # Run Inference
        outputs = session.run([output_name], {input_name: processed_images[i]})
        
        # RT-DETR Output Parsing
        # Ultralytics exports RT-DETR similarly to YOLO: [1, classes + 4, num_queries]
        # Usually shape is (1, 5, 300) for a 1-class model
        preds = outputs[0][0] 
        
        # Transpose it so rows are the 300 queries, and columns are [x, y, w, h, conf]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T
            
        txt_name = os.path.splitext(filenames[i])[0] + ".txt"
        with open(os.path.join(out_dir, txt_name), "w") as f:
            for query in preds:
                # Extract coordinates and class scores
                boxes = query[:4]
                scores = query[4:]
                
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    # Write in YOLO format (Class, X_center, Y_center, W, H, Conf)
                    f.write(f"{class_id} {boxes[0]:.6f} {boxes[1]:.6f} {boxes[2]:.6f} {boxes[3]:.6f} {confidence:.6f}\n")

    end_time = time.perf_counter()

    # --- CALCULATE METRICS ---
    total_time = end_time - start_time
    fps = total_valid / total_time

    print("\n=============================================")
    print(f"🏆 RT-DETR (ONNX RUNTIME) RESULTS")
    print("=============================================")
    print(f"Total Inference Time: {total_time:.2f} seconds")
    print(f"Throughput (FPS):     {fps:.2f} Frames Per Second")
    print("=============================================")
    print(f"Evaluation labels saved to: ./{out_dir}/")
    print("=============================================")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 ort_benchmark.py best_rtdetr_patched.onnx path/to/image/folder/")
    else:
        benchmark_ort(sys.argv[1], sys.argv[2])
