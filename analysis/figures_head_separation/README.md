# analysis/figures_head_separation/

`head_separation.ipynb` 산출물: RTMDet 분류 head(cls)와 회귀 head(reg)의 출력을 분리해 어느 쪽이 정답률을 결정하는지 본 분석.

| 파일 | 내용 |
|---|---|
| `11_cls_head_logit.png` | 분류 head logit 분포 (정답 vs 오답 박스) |
| `11_reg_head_iou.png` | 회귀 head IoU 분포 (박스 위치 정확도) |
| `11_summary.png` | cls vs reg 정답률 기여 요약 |

결론: 박스 위치(reg)는 거의 다 맞히는데 분류(cls)에서 Top-1을 놓친다. 후처리/클래스 임계값 조정 여지의 정량 근거다.
