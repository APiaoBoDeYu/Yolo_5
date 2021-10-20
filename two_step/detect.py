import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import os
import numpy as np


@torch.no_grad()
def detect(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA


    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    ##修改
    device2=select_device('2,3')
    #Load model2
    model2=attempt_load('../runs/train/exp42/weights/best.pt',map_location=device2)
    names2 = model2.module.names if hasattr(model2, 'module') else model2.names  # get class names

    model3=attempt_load('/home/igi/media/ymt/cv_competition/LJJJ_Xd/yolov5-develop/runs/train/exp56/weights/best.pt')
    names3=model3.module.names if hasattr(model3,'module') else model3.names


    for i in range(len(names2)):
        names2[i]=names2[i]+'_v9'
    if half:
        model2.half()  # to FP16
        # model3.half()
    if device.type != 'cpu':
        model2(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model2.parameters())))  # run once
        # model3(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model3.parameters())))

    if half:
        #训练时什么数据类型，使用时什么数据类型
        model3.float()
    if device.type != 'cpu':
        model3(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model3.parameters())))
        model3=model3.cuda()

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                #在这里把检测的坐标转化为原始图像的坐标
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    offset_x, offset_y = xyxy[0],xyxy[1]
                    #保存第一步检测结果
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # line = (names[int(cls)], conf, *xyxy)  # label format
                    # with open(txt_path + '.txt', 'a') as f:
                    #     f.write(('%s ' + '%g ' * (len(line) - 1)) % line + '\n')


                    save_one_box(xyxy, imc, file=save_dir / 'crops' / f'{p.stem}.jpg', BGR=True)

                    #读取图片
                    img2_0 = cv2.imread(os.path.join('./'+str(save_dir)+'/crops',f'{p.stem}.jpg'))

                    #图片预处理
                    img2 = letterbox(img2_0, 640, stride=32)[0]
                    img2 = img2[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    img2 = np.ascontiguousarray(img2)
                    img2 = torch.from_numpy(img2).to(device)
                    img2 = img2.half() if half else img2.float()  # uint8 to fp16/32
                    img2 /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img2.ndimension() == 3:
                        img2 = img2.unsqueeze(0)

                    #预测
                    pred2 = model2(img2, augment=opt.augment)[0]
                    pred2 = non_max_suppression(pred2, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                                max_det=opt.max_det)
                    #结果处理
                    for j, det2 in enumerate(pred2):
                        det2[:, :4] = scale_coords(img2.shape[2:], det2[:, :4], img2_0.shape).round()
                        #记录结果,保存类别、置信度、坐标
                        for *xyxy, conf, cls in reversed(det2):
                            if names2[int(cls)] in ['41_NZXJ_v9','44_NZXJ_YYX_v9','NZXJ_yjh_v9','NZXJ_yjb_v9']:
                                continue
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # normalized xywh
                            xyxy[0], xyxy[1], xyxy[2], xyxy[3] = xyxy[0] + offset_x, xyxy[1] + offset_y, xyxy[2] + offset_x, xyxy[3] + offset_y
                            #金具坐标对应xmin，ymin
                            offset_x2=xyxy[0]
                            offset_y2=xyxy[1]
                            line = (names2[int(cls)], conf, *xyxy)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%s ' + '%g ' * (len(line) - 1)) % line + '\n')
                            #每一个金具使用第三模型计算销钉
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / f'temp3.jpg', BGR=True)
                            # 读取图片
                            img3_0 = cv2.imread(os.path.join('./' + str(save_dir) + '/crops', f'temp3.jpg'))
                            # 图片预处理
                            img3 = letterbox(img3_0, 640, stride=32)[0]
                            img3 = img3[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                            img3 = np.ascontiguousarray(img3)
                            img3 = torch.from_numpy(img3).to(device)
                            img3 = img3.float()  # uint8 to fp16/32
                            img3 /= 255.0  # 0 - 255 to 0.0 - 1.0
                            if img3.ndimension() == 3:
                                img3 = img3.unsqueeze(0)

                            pred3 = model3(img3, augment=opt.augment)[0]
                            pred3 = non_max_suppression(pred3, opt.conf_thres, opt.iou_thres, opt.classes,
                                                        opt.agnostic_nms,
                                                        max_det=opt.max_det)
                            os.remove(os.path.join('./' + str(save_dir) + '/crops', f'temp3.jpg'))
                            for j, det3 in enumerate(pred3):
                                det3[:, :4] = scale_coords(img3.shape[2:], det3[:, :4], img3_0.shape).round()
                                # 记录结果,保存类别、置信度、坐标
                                for *xyxy, conf, cls in reversed(det3):
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                        -1).tolist()  # normalized xywh
                                    xyxy[0], xyxy[1], xyxy[2], xyxy[3] = xyxy[0] + offset_x2, xyxy[1] + offset_y2, xyxy[
                                        2] + offset_x2, xyxy[3] + offset_y2
                                    line = (names3[int(cls)], conf, *xyxy)  # label format
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write(('%s ' + '%g ' * (len(line) - 1)) % line + '\n')

                    os.remove(os.path.join('./'+str(save_dir)+'/crops',f'{p.stem}.jpg'))


            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')


        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            detect(opt=opt)
            strip_optimizer(opt.weights)
    else:
        detect(opt=opt)
