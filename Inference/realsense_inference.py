from openvino.inference_engine import IENetwork, IECore
# if error, run source /opt/intel/openvino/bin/setupvars.sh
import sys
import cv2
import numpy as np
from time import time
from tqdm import trange
import pyrealsense2 as rs
import time
from Inference.utils import *


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    prof = pipeline.start(config)
    sensor = prof.get_device().query_sensors()[1]
    sensor.set_option(rs.option.exposure, 2000.0)

    args = build_argparser().parse_args()
    model_xml = args.model + '.xml'
    model_bin = args.model + '.bin'

    # Plugin initialization for specified device and load extensions library if specified
    print("Creating Inference Engine")
    ie = IECore()
    print("Loading network files:\n\t {} \n\t {}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            print("Following layers are not supported by the plugin for specified device {}:\n {}".format(args.device,
                                                                                                          ', '.join(
                                                                                                              not_supported_layers)))
            print("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                  "or --cpu_extension command line argument")
            sys.exit(1)

    print("Preparing input blobs")
    input_blob = u"input"
    output_fmap_blob = "Mul_88"
    output_reg_blob = "Conv_84"
    output_wh_blob = "Conv_81"
    net.batch_size = 1


    class_names = ["red1", "red2", "blue1", "blue2", "red armor", "blue armor"]
    class_colors = [(0, 125, 209), (148, 76, 243), (243, 215, 76), (250, 245, 92), (0, 0, 255), (0, 255, 0)]
    n, c, h, w = net.inputs[input_blob].shape

    print("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)


    for _ in trange(200):
        results = exec_net.infer(inputs={input_blob: np.zeros((1, c, h, w), np.uint8)})
        fmap = torch.from_numpy(results[output_fmap_blob])
        reg = torch.from_numpy(results[output_reg_blob])
        wh = torch.from_numpy(results[output_wh_blob])
        bboxes, scores, classes = decode(fmap, reg, wh)
        bboxes, scores, classes = bboxes[0].numpy(), scores[0].numpy(), classes[0].numpy()

    videowriter = cv2.VideoWriter('detection.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))

    for i in trange(10000):
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        frame = np.asanyarray(color_frame.get_data())
        frame = cv2.resize(frame, (w, h))
        # frame = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2RGB)
        begin_millis = int(round(time.time() * 1000))
        results = exec_net.infer(inputs={input_blob: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[None].transpose(0, 3, 1, 2)})
        fmap = torch.from_numpy(results[output_fmap_blob])
        reg = torch.from_numpy(results[output_reg_blob])
        wh = torch.from_numpy(results[output_wh_blob])
        bboxes, scores, classes = decode(fmap, reg, wh)
        end_millis = int(round(time.time() * 1000)) - 1
        bboxes, scores, classes = bboxes[0].numpy(), scores[0].numpy(), classes[0].numpy().astype(int)
        bboxes = bboxes * 4
        for bbox, score, clas in zip(bboxes, scores, classes):
            if score > args.threshold:
                cls_id = clas[0]
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), class_colors[cls_id], 2)
                frame = cv2.putText(frame, class_names[cls_id], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors[cls_id], 1)
        cv2.putText(frame, "inference time: {}ms".format(end_millis - begin_millis), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "resolution: {}x{}".format(w, h), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame_display = cv2.resize(frame, (w*4, h*4))
        cv2.imshow("frame", frame_display)
        videowriter.write(frame)
        k = cv2.waitKey(1)
        if k == 27:  # Esc key to stop
            break
    
    videowriter.release()


if __name__ == '__main__':
    sys.exit(main() or 0)
