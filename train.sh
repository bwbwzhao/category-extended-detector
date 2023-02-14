# =============================================================================================================================================
# ==================================================================== 2023  ===============================================================
# =============================================================================================================================================


# =========================================================== D_o: coco75 & D_n: coco5 ========================================================
# step one


# plain joint training
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco75_coco5/retinanet_r50_fpn_1x_plain.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco75coco5_plain/

# dataset-aware
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco75_coco5/retinanet_r50_fpn_1x_datasetaware.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco75coco5_datasetaware/

# conflict-free
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco75_coco5/retinanet_r50_fpn_1x_conflictfree.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco75coco5_conflictfree/


# step two


# cc-based unlabeled ground-truth mining
sh ./tools/make_pseudo_labels.sh \
        ./configs/category_extended/coco75_coco5/retinanet_r50_fpn_1x_coco75coco5_conflictfree.py \
        ./work_dirs/2022/retinanet_r50_fpn_1x_coco75coco5_conflictfree/epoch_12.pth 8 \
        --prefix CC 

# clc-based unlabeled ground-truth mining
sh ./tools/make_pseudo_labels.sh \
        ./configs/category_extended/coco75_coco5/retinanet_r50_fpn_1x_coco75coco5_conflictfree.py \
        ./work_dirs/2022/retinanet_r50_fpn_1x_coco75coco5_conflictfree/epoch_12.pth 8 \
        --prefix CLC 


# step three


# cc + as fully labeled
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco75_coco5/retinanet_r50_fpn_1x_coco75coco5_ccasfull.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco75coco5_ccasfull/

# cc + conflict-free
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco75_coco5/retinanet_r50_fpn_1x_coco75coco5_ccposonly.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco75coco5_ccposonly/

# cc + safe negatives
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco75_coco5/retinanet_r50_fpn_1x_coco75coco5_ccpossafeneg.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco75coco5_ccpossafeneg/

# cc + overlap-weighted
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco75_coco5/retinanet_r50_fpn_1x_coco75coco5_ccposweightedneg.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco75coco5_ccposweightedneg/

# clc + overlap-weighted
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco75_coco5/retinanet_r50_fpn_1x_coco75coco5_clcposweightedneg.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco75coco5_clcposweightedneg/


# =========================================================== D_o: coco79 & D_n: coco1 ========================================================
# step one


# plain joint training
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco79_coco1/retinanet_r50_fpn_1x_coco79coco1_plain.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco79coco1_plain/

# conflict-free
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco79_coco1/retinanet_r50_fpn_1x_coco79coco1_conflictfree.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco79coco1_conflictfree/


# step two


# cc-based unlabeled ground-truth mining
sh ./tools/make_pseudo_labels.sh \
        ./configs/category_extended/coco79_coco1/retinanet_r50_fpn_1x_coco79coco1_conflictfree.py \
        ./work_dirs/2022/retinanet_r50_fpn_1x_coco79coco1_conflictfree/epoch_12.pth 8 \
        --prefix CC 

# clc-based unlabeled ground-truth mining
sh ./tools/make_pseudo_labels.sh \
        ./configs/category_extended/coco79_coco1/retinanet_r50_fpn_1x_coco79coco1_conflictfree.py \
        ./work_dirs/2022/retinanet_r50_fpn_1x_coco79coco1_conflictfree/epoch_12.pth 8 \
        --prefix CLC 


# step three


# cc + as fully labeled
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco79_coco1/retinanet_r50_fpn_1x_coco79coco1_ccasfull.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco79coco1_ccasfull/

# cc + conflict-free
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco79_coco1/retinanet_r50_fpn_1x_coco79coco1_ccposonly.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco79coco1_ccposonly/

# cc + safe negatives
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco79_coco1/retinanet_r50_fpn_1x_coco79coco1_ccpossafeneg.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco79coco1_ccpossafeneg/

# cc + overlap-weighted
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco79_coco1/retinanet_r50_fpn_1x_coco79coco1_ccposweightedneg.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco79coco1_ccposweightedneg/

# clc + overlap-weighted
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco79_coco1/retinanet_r50_fpn_1x_coco79coco1_clcposweightedneg.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco79coco1_clcposweightedneg/


# =========================================================== D_o: coco60 & D_n: voc ========================================================
# step one


# plain joint training
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco60_voc20/retinanet_r50_fpn_1x_coco60voc_plain.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco60voc_plain/

# conflict-free
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco60_voc20/retinanet_r50_fpn_1x_coco60voc_conflictfree.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco60voc_conflictfree/


# step two


# cc-based unlabeled ground-truth mining
sh ./tools/make_pseudo_labels.sh \
        ./configs/category_extended/coco60_voc20/retinanet_r50_fpn_1x_coco60voc_conflictfree.py \
        ./work_dirs/2022/retinanet_r50_fpn_1x_coco60voc_conflictfree/epoch_12.pth 8 \
        --prefix CC 

# clc-based unlabeled ground-truth mining
sh ./tools/make_pseudo_labels.sh \
        ./configs/category_extended/coco60_voc20/retinanet_r50_fpn_1x_coco60voc_conflictfree.py \
        ./work_dirs/2022/retinanet_r50_fpn_1x_coco60voc_conflictfree/epoch_12.pth 8 \
        --prefix CLC


# step three


# cc + as fully labeled
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco60_voc20/retinanet_r50_fpn_1x_coco60voc_ccasfull.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco60voc_ccasfull/

# cc + conflict-free
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco60_voc20/retinanet_r50_fpn_1x_coco60voc_ccposonly.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco60voc_ccposonly/

# cc + safe negatives
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco60_voc20/retinanet_r50_fpn_1x_coco60voc_ccpossafeneg.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco60voc_ccpossafeneg/

# cc + overlap-weighted
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco60_voc20/retinanet_r50_fpn_1x_coco60voc_ccposweightedneg.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco60voc_ccposweightedneg/

# clc + overlap-weighted
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco60_voc20/retinanet_r50_fpn_1x_coco60voc_clcposweightedneg.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_coco60voc_clcposweightedneg/


# =========================================================== D_o: coco & D_n: widerface ========================================================
# step one


# plain joint training
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco_widerface/retinanet_r50_fpn_1x_cocoface_plain.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_cocoface_plain/

# conflict-free
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco_widerface/retinanet_r50_fpn_1x_cocoface_conflictfree.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_cocoface_conflictfree/


# step two


# cc-based unlabeled ground-truth mining
sh ./tools/make_pseudo_labels.sh \
        ./configs/category_extended/coco_widerface/retinanet_r50_fpn_1x_cocoface_conflictfree.py \
        ./work_dirs/2022/retinanet_r50_fpn_1x_cocoface_conflictfree/epoch_12.pth 8 \
        --prefix CC 

# clc-based unlabeled ground-truth mining
sh ./tools/make_pseudo_labels.sh \
        ./configs/category_extended/coco_widerface/retinanet_r50_fpn_1x_cocoface_conflictfree.py \
        ./work_dirs/2022/retinanet_r50_fpn_1x_cocoface_conflictfree/epoch_12.pth 8 \
        --prefix CLC


# step three


# cc + as fully labeled
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco_widerface/retinanet_r50_fpn_1x_cocoface_ccasfull.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_cocoface_ccasfull/

# cc + conflict-free
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco_widerface/retinanet_r50_fpn_1x_cocoface_ccposonly.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_cocoface_ccposonly/

# cc + safe negatives
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco_widerface/retinanet_r50_fpn_1x_cocoface_ccpossafeneg.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_cocoface_ccpossafeneg/

# cc + overlap-weighted
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco_widerface/retinanet_r50_fpn_1x_cocoface_ccposweightedneg.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_cocoface_ccposweightedneg/

# clc + overlap-weighted
sh ./tools/dist_train.sh \
        ./configs/category_extended/coco_widerface/retinanet_r50_fpn_1x_cocoface_clcposweightedneg.py 8 \
        --work-dir ./work_dirs/2022/retinanet_r50_fpn_1x_cocoface_clcposweightedneg/

