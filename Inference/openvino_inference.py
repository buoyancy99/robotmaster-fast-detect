import sys
import cv2
import numpy as np
from tqdm import trange
from Inference.utils import *


def main():
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

    n, c, h, w = net.inputs[input_blob].shape

    print("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)
    cap = cv2.VideoCapture(0)

    for _ in trange(200):
        results = exec_net.infer(inputs={input_blob: np.zeros((1, c, h, w), np.uint8)})
        fmap = torch.from_numpy(results[output_fmap_blob])
        reg = torch.from_numpy(results[output_reg_blob])
        wh = torch.from_numpy(results[output_wh_blob])
        bboxes, scores, classes = decode(fmap, reg, wh)
        bboxes, scores, classes = bboxes[0].numpy(), scores[0].numpy(), classes[0].numpy()

    for i in trange(1000):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (w, h))
        results = exec_net.infer(inputs={input_blob: frame[None].transpose(0, 3, 1, 2)})
        fmap = torch.from_numpy(results[output_fmap_blob])
        reg = torch.from_numpy(results[output_reg_blob])
        wh = torch.from_numpy(results[output_wh_blob])
        bboxes, scores, classes = decode(fmap, reg, wh)
        bboxes, scores, classes = bboxes[0].numpy(), scores[0].numpy(), classes[0].numpy()
        bboxes = bboxes * 4
        for bbox, score, clas in zip(bboxes, scores, classes):
            if score > args.threshold:
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
        cv2.imshow("frame", frame)
        k = cv2.waitKey(10)
        if k == 27:  # Esc key to stop
            break


if __name__ == '__main__':
    sys.exit(main() or 0)
