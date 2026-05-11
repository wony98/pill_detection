# training/notebooks/

모델 학습/평가 노트북 모음. 메인 라인은 RTMDet-tiny v1~v4, 비교군으로 Faster R-CNN / RF-DETR / YOLOv8n.

| 파일 | 내용 |
|---|---|
| `rtmdet_tiny_synth_train_v1.ipynb` | RTMDet-tiny v1 (베이스라인 합성 적용) |
| `rtmdet_tiny_synth_train_v2.ipynb` | v2 — 배경 다양화 + 격자 단축경로 차단 |
| `rtmdet_tiny_synth_train_v3.ipynb` | v3 — palm 마스크 + dark surface |
| `rtmdet_tiny_synth_train_v4.ipynb` | v4 — imgsz 1536 + Albu 4종 + EMA 튜닝 (**최종 채택**) |
| `rtmdet_train_seed42.ipynb` / `rtmdet_train_seed100.ipynb` | RTMDet 시드별 재학습 |
| `ablation_11_experiments.ipynb` | 11개 ablation 실험 묶음 |
| `faster_rcnn_train_eval.ipynb` | Faster R-CNN 비교 |
| `rfdetr_train_eval.ipynb` | RF-DETR 비교 |
| `yolov8n_no_aug.ipynb` | YOLOv8n no-aug 비교 |

config는 [`training/configs/`](../configs/), 모델 가중치는 Hugging Face Hub 참조 (README 최상단 안내).
