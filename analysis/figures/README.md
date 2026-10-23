# analysis/figures/

`analysis/` 노트북들이 생성한 진단 그림 모음. 4-way (baseline/v2/v3/v4) 비교 단계에서 만든 메인 진단 자료가 여기 모입니다.

## 그룹별 정리

| 프리픽스 | 내용 |
|---|---|
| `01_*` ~ `13_*` | 메인 진단: 33장 그리드, 중심 분포, bbox 크기 vs verdict, pill count vs verdict, 환경별 정답률, 컨퓨전 페어, FP 분석, 클래스 정답률, 학습 커브, IoU 분포, Top-3 회복, pin 클래스 |
| `fp_01` ~ `fp_08` | FP 심층 분석: confidence 분포, aspect ratio 격자, edge strip 분류, scatter 위치, heatmap, edge 거리 히스토, top-50 conf, 의심 GT 확인 |
| `imprint_01` ~ `imprint_08` | 각인 가시성 연구: 정의, downsample 단일/dual, resize 방식, 가설 검증, m2 일반성, 100 ROI 분포, area 보존 |
| `kcode_01` ~ `kcode_07` | K-code ROI 품질: bbox crop, 100 ROI 분포, area vs preservation, 각인 강도, rembg pre/post |
| `quality_01` ~ `quality_08` | 합성 데이터 품질: pill count, size 분포, box center, bg 밝기, sample, cache ROI, palm/hand, color diff |

> 그룹별 평가 그림은 `figures_v2_compare/`, `figures_v3_eval/`, `figures_v4_eval/` 별도 폴더 참조.
