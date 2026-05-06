import os
import glob
import numpy as np
import sys

def bbox_iou(box1, box2):
    # YOLO format (xc, yc, w, h) -> (x1, y1, x2, y2)
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    ix1, iy1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    ix2, iy2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (b1_x2 - b1_x1)*(b1_y2 - b1_y1) + (b2_x2 - b2_x1)*(b2_y2 - b2_y1) - inter
    return inter / union if union > 0 else 0

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

def evaluate(pred_dir, gt_dir, iou_thresh=0.5):
    pred_files = glob.glob(os.path.join(pred_dir, "*.txt"))
    gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))
    
    if not pred_files or not gt_files:
        print("Error: Could not find .txt files in one or both directories.")
        return

    all_preds = [] 
    gt_boxes = {} 
    num_gt = 0
    ignored_lines = 0
    
    # 1. Load Ground Truths (Safely!)
    for gt_f in gt_files:
        img_id = os.path.basename(gt_f)
        gt_boxes[img_id] = []
        with open(gt_f, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                # Skip any line with commas or letters that would break the float conversion
                try:
                    parts = list(map(float, line.split()))
                    if len(parts) >= 5: 
                        gt_boxes[img_id].append(parts[1:5])
                        num_gt += 1
                except ValueError:
                    ignored_lines += 1
                    continue
                    
    # 2. Load Predictions (Safely!)
    for pred_f in pred_files:
        img_id = os.path.basename(pred_f)
        with open(pred_f, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    parts = list(map(float, line.split()))
                    if len(parts) >= 6: 
                        all_preds.append([parts[5], img_id, parts[1:5]])
                except ValueError:
                    continue
                    
    all_preds.sort(key=lambda x: x[0], reverse=True)
    
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))
    matched_gts = {img_id: [] for img_id in gt_boxes.keys()}
    
    print(f"Skipped {ignored_lines} corrupted/invalid lines in ground truth files.")
    print("Calculating Matches & PR Curve...")
    
    # 3. Calculate Matches
    for d, pred in enumerate(all_preds):
        conf, img_id, pred_box = pred
        if img_id not in gt_boxes:
            fp[d] = 1
            continue
            
        gts = gt_boxes[img_id]
        best_iou, best_gt_idx = 0, -1
        
        for i, gt_box in enumerate(gts):
            iou = bbox_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_gt_idx = iou, i
                
        if best_iou >= iou_thresh:
            if best_gt_idx not in matched_gts[img_id]:
                tp[d] = 1
                matched_gts[img_id].append(best_gt_idx)
            else:
                fp[d] = 1 
        else:
            fp[d] = 1
            
    # 4. Compute Metrics
    fpc = np.cumsum(fp)
    tpc = np.cumsum(tp)
    
    rec = tpc / num_gt if num_gt > 0 else np.zeros_like(tpc)
    prec = tpc / np.maximum(tpc + fpc, np.finfo(np.float64).eps)
    ap = compute_ap(rec, prec)
    
    if len(prec) > 0:
        f1 = 2 * (prec * rec) / np.maximum(prec + rec, np.finfo(np.float64).eps)
        max_f1_idx = np.argmax(f1)
        best_prec, best_rec = prec[max_f1_idx], rec[max_f1_idx]
    else:
        best_prec, best_rec = 0, 0
        
    print("\n=============================================")
    print("📈 EVALUATION METRICS (IoU=0.50)")
    print("=============================================")
    print(f"Total Ground Truths: {num_gt}")
    print(f"Total Predictions:   {len(all_preds)}")
    print("---------------------------------------------")
    print(f"Precision: {best_prec:.4f}")
    print(f"Recall:    {best_rec:.4f}")
    print(f"mAP@50:    {ap:.4f}")
    print("=============================================")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 evaluator.py <predictions_folder> <ground_truth_folder>")
    else:
        evaluate(sys.argv[1], sys.argv[2])
