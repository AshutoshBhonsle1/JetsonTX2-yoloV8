import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys
import os

# Pure NumPy NMS (Bypasses the missing cv2.dnn module)
def nms(boxes, scores, iou_threshold=0.4):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def infer(engine_path, image_path):
    # 1. Setup TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # 2. Allocate GPU Memory
    input_shape = context.get_binding_shape(0)
    output_shape = context.get_binding_shape(1)
    
    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    # 3. Load & Process Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
        
    orig_h, orig_w = img.shape[:2]
    input_w, input_h = input_shape[3], input_shape[2]
    
    resized = cv2.resize(img, (input_w, input_h))
    input_data = resized.astype(np.float32) / 255.0
    input_data = input_data.transpose(2, 0, 1) 
    input_data = np.expand_dims(input_data, axis=0) 
    np.copyto(h_input, input_data.ravel())

    # 4. Execute the TensorRT Engine
    print("Processing on GPU...")
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    # 5. Extract Bounding Boxes & Apply Custom NMS
    output = h_output.reshape(output_shape)[0].T 
    
    boxes = []
    scores = []
    
    for row in output:
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        score = classes_scores[class_id]
        
        if score > 0.5: 
            xc, yc, w, h = row[0], row[1], row[2], row[3]
            x1 = int((xc - w/2) * (orig_w / input_w))
            y1 = int((yc - h/2) * (orig_h / input_h))
            w_scaled = int(w * (orig_w / input_w))
            h_scaled = int(h * (orig_h / input_h))
            
            boxes.append([x1, y1, w_scaled, h_scaled])
            scores.append(float(score))

    # Use our custom Math-based NMS instead of OpenCV
    indices = nms(boxes, scores, iou_threshold=0.4)
    
    if len(indices) > 0:
        for i in indices:
            x, y, w, h = boxes[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"Conf: {scores[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Detected object at: X:{x} Y:{y}")
    else:
        print("No objects detected above threshold.")

    out_name = "result_" + os.path.basename(image_path)
    cv2.imwrite(out_name, img)
    print(f"\nSuccess! Image saved as: {out_name}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 infer.py yolov8n.engine path/to/your/test_image.jpg")
    else:
        infer(sys.argv[1], sys.argv[2])
