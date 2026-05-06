Before running benchmarks, maximise the hardware clocks of the board to ensure peak performance.\
Run:\
sudo nvpmodel -m 0 && sudo jetson_clocks\
\
Files:\
\
infer.py allows you to test your model by giving it a single image.\
Terminal command to run infer.py:\
python3 infer.py <path/to/model.engine> <path/to/test_image.jpg>\
\
infer_threaded.py allows you to run multithreaded inference to maximise fps.\
Terminal command to run infer_threaded.py:\
python3 infer_threaded.py\
Add paths to the code itself.\
\

evaluator.py measures precision, recall and mAP of your model.\
Terminal command to run evaluator.py:\
python3 evaluator.py <path/to/predictions_directory/> <path/to/ground_truth_directory/>\
\

ort_benchmark.py allows you to benchmark performance using ONNX runtime.\
Terminal command to run ort_benchmark.py:\
python3 ort_benchmark.py <path/to/model.onnx> <path/to/image_directory/>\
\
threaded_latency.py measures the latency in inferenceing on the board with multithreading.\
Terminal command to run threaded_latency.py:\
python3 threaded_latency.py\
Add paths to the code itself.\
