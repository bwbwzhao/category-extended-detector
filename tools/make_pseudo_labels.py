import argparse
import os

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from tools.fuse_conv_bn import fuse_module

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument(
        '--config',
        default='./configs/category_extended/coco75_coco5/retinanet_r50_fpn_1x_coco75coco5_conflictfree.py',
        help='test config file path')
    parser.add_argument(
        '--checkpoint', 
        default='./work_dirs/mm2021/retinanet_r50_fpn_1x_coco75coco5_conflictfree/epoch_12.pth',
        help='checkpoint file')
    parser.add_argument(
        '--prefix', 
        default='CC')
    parser.add_argument(
        '--out', 
        help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        default='bbox',
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', 
        default=None,
        help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.5,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if isinstance(cfg.data.test, dict):
        cfg.data.test = [cfg.data.test]
    for i in range(len(cfg.data.test)):
        cfg.data.test[i].test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    for i in range(len(cfg.data.test)):
        # build the dataloader
        # TODO: support multiple images per gpu (only minor changes are needed)
        dataset = build_dataset(cfg.data.test[i])
        # dataset.data_infos = dataset.data_infos[:5] # just for debug
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=1,
            dist=distributed,
            shuffle=False)

        # build the model and load checkpoint
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        if args.fuse_conv_bn:
            model = fuse_module(model)
        model.CLASSES = dataset.CLASSES


        test_type = 'drop' if args.prefix=='CLC' else 'none'
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                    args.show_score_thr, test_type)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                    args.gpu_collect, test_type)

        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)

            if cfg.model.bbox_head.set_splits==['A', 'B']:
                original_classes, new_classes_B = cfg.model.bbox_head.class_splits[0], cfg.model.bbox_head.class_splits[1]
                json_results = dataset._det2json_pseudo_labels(outputs, 
                                                            A=list(range(original_classes[0], original_classes[1])), 
                                                            B=list(range(new_classes_B[0], new_classes_B[1])), 
                                                            score_thre=0.1)
            f_name = cfg.data.test[i]['ann_file'][:-5].split('/')
            anns_path = '/'.join(f_name[:-1]) + '/pseudo/'
            if not os.path.exists(anns_path):
                os.makedirs(anns_path)
            combined_ann_file = anns_path + '%s_%s_pseudo_%d.json'%(f_name[-1], args.prefix, len(json_results))
            # combine original ann and pseudo ann
            with open(cfg.data.test[i]['ann_file']) as ori_f:
                ori_data = json.loads(ori_f.read())
            with open(combined_ann_file, 'w') as combined_f:
                ori_data['annotations'] = ori_data['annotations'] + json_results
                json.dump(ori_data, combined_f)

if __name__ == '__main__':
    main()
