from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector
from tqdm import tqdm

from mmpose.apis import (inference_pose_model, init_pose_model,
                         show_result_pyplot)


def _xyxy2xywh(bbox_xyxy):
    """transform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        bbox_xywh (ndarray): Bounding boxes (with scores),
            shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0] + 1
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1] + 1
    return bbox_xywh


def _xywh2xyxy(bbox_xywh):
    """transform the bbox format from xywh to x1y1x2y2.

    Args:
        bbox_xywh (ndarray): Bounding boxes (with scores),
            shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    Returns:
        bbox (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])
    """
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[:, 2] = bbox_xyxy[:, 2] + bbox_xyxy[:, 0] - 1
    bbox_xyxy[:, 3] = bbox_xyxy[:, 3] + bbox_xyxy[:, 1] - 1
    return bbox_xyxy


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--det_config', default=None, help='Config file for detection')
    parser.add_argument(
        '--det_checkpoint', default=None, help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--box_thr', type=float, default=0.3, help='box score threshold')
    parser.add_argument(
        '--pose_thr', type=float, default=0.3, help='box score threshold')
    args = parser.parse_args()

    if args.load_json_file == '':
        assert args.det_config is not None
        assert args.det_checkpoint is not None

        det_model = init_detector(
            args.det_config, args.det_checkpoint, device='cuda:0')
        # build the pose model from a config file and a checkpoint file
        pose_model = init_pose_model(
            args.pose_config, args.pose_checkpoint, device=args.device)

        # test a single image, the resulting box is (x1, y1, x2, y2)
        box_results = inference_detector(det_model, args.img)
        person_bboxes = box_results[0][0]
        person_bboxes = _xyxy2xywh(person_bboxes)
        pose_results = []

        if len(person_bboxes) > 0:
            bboxes = person_bboxes[person_bboxes[:, 4] > args.box_thr]
            for bbox in bboxes:
                pose = inference_pose_model(pose_model, args.img, bbox)
                pose_results.append({
                    'bbox': _xywh2xyxy(bbox),
                    'keypoints': pose,
                })

        # show the results
        show_result_pyplot(
            pose_model, args.img, pose_results, score_thr=args.score_thr)

    else:
        from pycocotools.coco import COCO
        coco = COCO(args.load_json_file)

        # build the pose model from a config file and a checkpoint file
        pose_model = init_pose_model(
            args.pose_config, args.pose_checkpoint, device=args.device)

        img_keys = list(coco.imgs.keys())

        for i in tqdm(range(len(img_keys))):
            image_id = img_keys[i]
            image = coco.loadImgs(image_id)[0]
            img_name = image['file_name']
            anns = coco.getAnnIds(image_id)
            pose_results = []
            for ann in anns:
                bbox = ann['bbox']
                pose = inference_pose_model(pose_model, img_name, bbox)
                pose_results.append({
                    'bbox': _xywh2xyxy(bbox),
                    'keypoints': pose,
                })
            # show the results
            show_result_pyplot(
                pose_model, args.img, pose_results, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
