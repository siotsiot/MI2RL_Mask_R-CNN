# Chest X-ray Lung Segmentation using Mask R-CNN

> MI2RL 겨울방학 인턴 기간 동안 진행한 Mask R-CNN 기반 흉부 X-ray 폐 영역 분할 프로젝트입니다.  
> This repository contains a Mask R-CNN-based chest X-ray lung segmentation project conducted during my MI2RL winter internship.

---

## 프로젝트 개요

본 프로젝트는 서울아산병원 의료영상지능실현연구실(MI2RL) 겨울방학 인턴 기간 동안 진행한 흉부 X-ray 폐 영역 분할 프로젝트입니다.

주된 목적은 Mask R-CNN 모델 구조를 이해하고, 공개 Chest X-ray Lung Segmentation 데이터셋에 적용하여 의료영상 segmentation의 전체 파이프라인을 경험하는 것이었습니다.

본 프로젝트는 새로운 알고리즘을 제안하기 위한 연구라기보다는, Mask R-CNN의 구조와 학습 과정을 이해하고 의료영상 데이터에 적용해보는 인턴 학습 프로젝트 성격이 강합니다.

---

## 주요 내용

- R-CNN 계열 모델 학습
  - R-CNN
  - Fast R-CNN
  - Faster R-CNN
  - Mask R-CNN
- 공개 Mask R-CNN PyTorch 코드 분석
- COCO 데이터셋 기반 모델 정상 동작 검증
- Chest X-ray Lung Segmentation 데이터셋 적용
- mask 이미지 기반 annotation에서 bounding box 자동 생성
- PyTorch Dataset/DataLoader 구조로 학습 파이프라인 정리
- IoU, Dice 기반 성능 평가
- Automatic Mixed Precision(AMP) 적용 전후 학습 속도 비교

---

## 사용 데이터셋

본 프로젝트에서는 Kaggle에 공개된 Chest X-ray Lung Segmentation 데이터셋을 사용했습니다.

- Dataset: Lung segmentation from Chest X-Ray dataset
- Source: Kaggle public dataset
- Data type:
  - Chest X-ray image
  - Lung mask image

데이터셋은 흉부 X-ray 원본 영상과 폐 영역 mask로 구성되어 있습니다.  
본 프로젝트에서는 좌폐와 우폐를 각각 instance로 분리하여 Mask R-CNN 학습에 활용했습니다.

당시 실험에서는 다음과 같이 데이터를 사용했습니다.

| 구분 | 이미지 수 |
|---|---:|
| Training | 400 |
| Validation/Evaluation | 100 |

다만 별도의 독립적인 test set을 명확히 분리하지 못한 점은 본 프로젝트의 한계입니다.

---

## 모델 및 접근 방식

본 프로젝트에서는 Mask R-CNN을 사용했습니다.

Mask R-CNN은 객체의 위치를 나타내는 bounding box와 함께, 픽셀 단위의 mask를 예측하는 instance segmentation 모델입니다.

흉부 X-ray 폐 영역 분할 task에서는 일반적으로 semantic segmentation 모델인 U-Net 계열도 많이 사용될 수 있습니다.  
하지만 본 프로젝트에서는 인턴 기간 동안 R-CNN 계열 모델을 공부하는 것이 주요 목표였기 때문에 Mask R-CNN을 중심으로 실험을 진행했습니다.

---

## 데이터 처리 방식

원본 Chest X-ray Lung Segmentation 데이터셋은 JSON annotation이 아니라, 이미지와 mask 파일로 구성되어 있습니다.

하지만 Mask R-CNN 학습을 위해서는 각 instance에 대한 다음 정보가 필요합니다.

- bounding box
- class label
- segmentation mask

따라서 본 프로젝트에서는 mask 이미지를 기반으로 좌폐와 우폐 영역을 분리하고, 각 폐 영역에 대해 bounding box를 자동 생성했습니다.

처리 흐름은 다음과 같습니다.

```text
Chest X-ray image
    ↓
Lung mask image
    ↓
Left lung / Right lung instance 분리
    ↓
각 instance의 bounding box 자동 생성
    ↓
Mask R-CNN target 구조로 변환
    ↓
PyTorch Dataset / DataLoader에 적용
```

## 프로젝트 진행 과정

### 1. COCO 데이터셋 기반 모델 검증

먼저 COCO 데이터셋을 사용하여 기존 Mask R-CNN 코드가 정상적으로 동작하는지 확인했습니다.

이를 통해 다음 내용을 확인했습니다.

- 모델 학습 파이프라인 동작 여부
- loss 감소 여부
- Dataset / DataLoader 구조
- inference 및 evaluation 과정

---

### 2. Chest X-ray 데이터셋 적용

이후 Chest X-ray Lung Segmentation 데이터셋에 맞게 코드를 수정했습니다.

주요 수정 사항은 다음과 같습니다.

- mask 이미지 기반 annotation 처리
- 좌폐 / 우폐 instance 분리
- bounding box 자동 생성
- PyTorch Dataset 구조 정리
- 의료영상 데이터에 맞게 COCO 기반 로직 단순화
- 학습 진행 상황을 `tqdm`으로 시각화

---

### 3. 학습 및 평가

본 학습은 10 epoch 동안 진행했습니다.

학습 과정에서는 epoch별 loss와 iteration time, forward time, backward time 등을 확인했습니다.  
이후 검증/평가 데이터 100장을 대상으로 IoU와 Dice score를 계산했습니다.

---

## 실험 결과

100개 검증/평가 영상 기준 평균 성능은 다음과 같습니다.

| Metric | Average |
|---|---:|
| IoU | 0.825 |
| Dice | 0.904 |

IoU와 Dice는 segmentation 결과가 ground truth mask와 얼마나 잘 겹치는지를 평가하는 지표입니다.

- IoU는 예측 영역과 정답 영역의 교집합을 합집합으로 나눈 값입니다.
- Dice는 예측 영역과 정답 영역의 겹치는 정도를 더 민감하게 반영하는 지표입니다.

---

## AMP 적용 실험

Automatic Mixed Precision(AMP)을 적용하여 학습 속도 변화를 비교했습니다.

AMP는 PyTorch가 연산 특성에 따라 FP16과 FP32를 자동으로 선택하여 학습 속도와 메모리 사용량을 개선하는 방법입니다.

| 구분 | Iteration Time | IoU |
|---|---:|---:|
| AMP 미적용 | 800–900 ms | 0.825 |
| AMP 적용 | 430–700 ms | 0.818 |

AMP 적용 시 전체 학습 속도는 약 30–40% 향상되었고, IoU 기준 성능 저하는 크지 않았습니다.

---

## 배운 점

이 프로젝트를 통해 다음 내용을 경험했습니다.

- R-CNN 계열 모델의 발전 흐름
- Object Detection과 Instance Segmentation의 차이
- Mask R-CNN의 전체 구조
- 의료영상 segmentation 데이터셋 처리 방식
- mask 기반 annotation에서 bounding box를 생성하는 방법
- PyTorch 기반 Dataset / DataLoader 구성
- 학습, inference, evaluation 파이프라인 구성
- IoU, Dice 기반 segmentation 평가
- AMP를 활용한 학습 속도 개선

---

## 발표 및 피드백

본 프로젝트는 2026년 2월 9일 의료 AI Lab 세미나에서 인턴 종료 발표 형태로 정리했습니다.

발표 이후 교수님들께 받은 주요 피드백은 다음과 같습니다.

### 1. Train / Validation / Test 구분 명확화

데이터셋을 train, validation, test로 어떻게 나누었는지 더 명확하게 작성할 필요가 있다는 피드백을 받았습니다.

당시 프로젝트에서는 학습 데이터 400장과 검증/평가 데이터 100장을 사용했지만, 별도의 독립적인 test set을 명확히 분리하지 못했습니다.  
이 부분은 본 프로젝트의 한계로 정리했습니다.

---

### 2. 실패 사례 분석 필요

잘 분할된 사례뿐만 아니라, 잘못 분할된 사례도 함께 분석하면 좋겠다는 피드백을 받았습니다.

예를 들어 다음과 같은 경우를 분석할 수 있습니다.

- 폐 영역 일부가 누락된 경우
- 폐가 아닌 영역까지 과분할된 경우
- 좌폐와 우폐 instance 분리가 부정확한 경우
- 경계가 부정확하게 예측된 경우

본 프로젝트에서는 성공 사례 중심으로 결과를 확인했기 때문에, 체계적인 실패 사례 분석은 충분히 수행하지 못했습니다.

---

### 3. U-Net 기반 접근과의 비교 가능성

폐 영역 분할 task에서는 Mask R-CNN뿐만 아니라 U-Net 계열 모델이 더 직접적인 baseline이 될 수 있다는 피드백을 받았습니다.

Mask R-CNN은 instance segmentation 모델이기 때문에 좌폐와 우폐를 각각 instance로 구분하는 데 활용할 수 있습니다.  
반면 폐 영역 분할 자체는 semantic segmentation 문제로도 볼 수 있으므로, U-Net과 같은 encoder-decoder 기반 segmentation 모델과 비교했다면 더 적절한 분석이 가능했을 것입니다.

---

### 4. Instance 분리 로직 및 예외 처리

본 프로젝트에서는 좌폐와 우폐를 각각 instance로 분리하여 처리했습니다.

일반적인 흉부 X-ray에서는 좌폐와 우폐가 각각 존재하지만, 의료영상에서는 예외적인 케이스가 있을 수 있습니다.  
따라서 폐 영역이 하나만 검출되는 경우나 mask가 비정상적으로 구성된 경우에 대한 예외 처리 로직을 더 명확히 작성하면 좋겠다는 피드백을 받았습니다.

---

## 한계

본 프로젝트는 새로운 알고리즘을 제안하기 위한 연구라기보다는, 인턴 기간 동안 Mask R-CNN과 의료영상 segmentation 파이프라인을 학습하기 위한 프로젝트였습니다.

따라서 다음과 같은 한계가 있습니다.

- 공개 데이터셋 기반 실습 프로젝트입니다.
- 별도의 독립적인 test set을 명확히 분리하지 못했습니다.
- 다양한 segmentation 모델과의 비교 실험은 수행하지 않았습니다.
- 성공 사례 중심으로 결과를 확인했으며, 체계적인 실패 사례 분석은 충분히 수행하지 못했습니다.
- 외부 데이터셋에 대한 일반화 성능 검증은 제한적입니다.
- 임상 적용을 목적으로 한 연구는 아닙니다.
- 좌폐 / 우폐 instance 분리 과정에서 예외 케이스에 대한 처리가 충분하지 않았습니다.

---

## 정리

본 프로젝트는 Mask R-CNN을 활용하여 흉부 X-ray 영상에서 폐 영역을 분할하는 인턴 학습 프로젝트입니다.

프로젝트를 통해 R-CNN 계열 모델의 흐름과 Mask R-CNN의 instance segmentation 구조를 공부했고, 공개 의료영상 데이터셋에 적용하여 학습, 추론, 평가, 시각화까지의 전체 파이프라인을 경험했습니다.

또한 교수님들께 받은 피드백을 통해 데이터셋 분할의 명확성, 실패 사례 분석, U-Net 계열 baseline의 필요성, instance 분리 로직의 예외 처리 등 의료영상 segmentation 연구에서 고려해야 할 요소들을 이해할 수 있었습니다.
