from openvino.inference_engine import IENetwork, IECore
# if error, run source /opt/intel/openvino/bin/setupvars.sh
import sys
import cv2
import numpy as np
from tqdm import trange
import pyrealsense2 as rs
from Inference.utils import *
import zmq
import json

def main():
    pipeline = rs.pipeline()
    align_op = rs.align(rs.stream.color)
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
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
            print("Following layers are not supported by the plugin for specified device {}:\n {}"
                  .format(args.device, ', '.join(not_supported_layers)))
            print("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                  "or --cpu_extension command line argument")
            sys.exit(1)

    print("Preparing input blobs")
    input_blob = u"input"
    output_fmap_blob = "Mul_88"
    output_reg_blob = "Conv_84"
    output_wh_blob = "Conv_81"
    net.batch_size = 1

    n, c, h, w = net.inputs[input_blob].shape

    print("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    print("Starting ZeroMQ")
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:{}".format(args.port))

    if args.color == "red":
        target_cls_id = 4
    elif args.color == "blue":
        target_cls_id = 5
    else:
        assert False

    print("Starting Inference Node")
    for i in trange(1000000):
        frames = pipeline.wait_for_frames()
        frames = align_op.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        img_h, img_w, _ = frame.shape
        frame = cv2.resize(frame, (w, h))
        results = exec_net.infer(inputs={input_blob: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[None].transpose(0, 3, 1, 2)})
        fmap = torch.from_numpy(results[output_fmap_blob])
        reg = torch.from_numpy(results[output_reg_blob])
        wh = torch.from_numpy(results[output_wh_blob])
        bboxes, scores, classes = decode(fmap, reg, wh)
        bboxes, scores, classes = bboxes[0].numpy(), scores[0].numpy(), classes[0].numpy().astype(int)
        bboxes = bboxes * 4
        bboxes[0] = bboxes[0] / w * img_w
        bboxes[2] = bboxes[2] / w * img_w
        bboxes[1] = bboxes[1] / h * img_h
        bboxes[3] = bboxes[3] / h * img_h

        armors = []
        for bbox, score, clas in zip(bboxes, scores, classes):
            if score > args.threshold and clas[0] == target_cls_id:
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                priority = img_h / 2 - bbox[3] - 0.1 * abs(img_w / 2 - center[0])
                armors.append((center, priority))
        
        armors = sorted(armors, key=lambda x: -x[1])
        if armors:
            armor = armors[0][0]
            pixel_x = int(armor[0])
            pixel_y = int(armor[1])
            depth = depth_frame.get_distance(pixel_x, pixel_y)
            cord = rs.rs2_deproject_pixel_to_point(depth_frame.profile.as_video_stream_profile().intrinsics, [pixel_x, pixel_y], depth)
            print(cord)
            message = (1, cord[0], cord[1], cord[2])
        else:
            message = (0, 0, 0, 0)

        socket.send_string(json.dumps(message))
        


if __name__ == '__main__':
    sys.exit(main() or 0)
