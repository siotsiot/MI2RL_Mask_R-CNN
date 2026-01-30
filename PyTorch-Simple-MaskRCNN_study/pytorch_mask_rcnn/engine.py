import sys
import time

import torch

from .utils import Meter, TextArea
try:
    from .datasets import CocoEvaluator, prepare_for_coco
except:
    pass

##### 데이터를 batch 단위로 돌면서 forward → loss → backward → optimizer.step을 반복해 모델을 실제로 학습하는 함수 #####
# 모델 학습은 다음과 같은 순서로 진행
# 1. 현재 epoch에서 사용할 learning rate를 optimizer에 적용한다.
# 2. 모델을 학습 모드로 전환한다.
# 3. 훈련 데이터 batch를 하나씩 꺼낸다.
# 4. 순전파 → 손실 계산 → 역전파 → 가중치 갱신 과정을 수행한다.
# 5. 시간과 속도 측정해서 평균 시간을 반환한다.
#
# @return:
#   epoch당 평균 iteration 시간 (초)
def train_one_epoch(model, optimizer, data_loader, device, epoch, args):
    ##### epoch lr을 optimizer에 반영
    # optimizer는 내부에 param_groups라는 파라미터 묶음 리스트가 있음
    # 각 묶음마다 lr이 존재
    # 현재 epoch에서 쓸 lr로 실제 optimizer의 lr을 설정
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    ##### iteration 수 결정
    # 현재 epoch에서 몇 번 반복할지 iters에 정함
    # len(data_loader): 한 epoch에 배치가 총 몇 개인지 반환
    # args.iters: -1이면 data_loader의 길이만큼 반복, 양수면 해당 횟수만큼 반복
    iters = len(data_loader) if args.iters < 0 else args.iters

    ##### 학습 중 걸리는 시간을 종류별로 측정 #####
    # t_m: 전체 시간 측정용 Meter
    # m_m: 모델 순전파 시간 측정용 Meter
    # b_m: 역전파 시간 측정용 Meter
    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    # 모델을 학습 모드로 전환 (Dropout, BatchNorm 등 활성화)
    model.train()
    # 전체 학습 시작 시간 기록
    A = time.time()
    # 훈련 데이터에서 배치 하나씩 꺼내며 반복 시작
    for i, (image, target) in enumerate(data_loader):
        # 현재 반복 시작 시간 기록
        T = time.time()
        # 현재 몇 번째 iteration인지 계산
        num_iters = epoch * len(data_loader) + i
        # 현재 warmup 구간인지 확인
        # warmup 구간이면 learning rate를 선형 증가시켜 적용
        if num_iters <= args.warmup_iters:
            r = num_iters / args.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * args.lr_epoch

        # 입력 영상을 device(GPU)로 이동        
        image = image.to(device)
        # 타겟(정답) 데이터를 device(GPU)로 이동
        target = {k: v.to(device) for k, v in target.items()}
        # 모델 순전파 시작 시간 기록
        S = time.time()
        
        # 모델에 입력과 정답을 넣어서 여러 개의 손실값을 계산
        losses = model(image, target)
        # 여러 개의 손실값을 하나로 합침
        total_loss = sum(losses.values())
        # 모델 순전파 종료 후 걸린 시간 기록
        m_m.update(time.time() - S)
        
        # 역전파 시작 시간 기록
        S = time.time()
        # 역전파 실행
        total_loss.backward()
        # 역전파 종료 후 걸린 시간 기록
        b_m.update(time.time() - S)
        
        # 계산된 gradient를 이용해 모델 파라미터를 실제로 갱신
        optimizer.step()
        # 다음 iteration을 위해 gradient를 초기화
        optimizer.zero_grad()

        # 일정 iteration마다 로그 출력 조건 확인
        if num_iters % args.print_freq == 0:
            # 현재 iteration 번호, 각 손실값 출력
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))

        # 현재 batch에 걸린 시간 기록
        t_m.update(time.time() - T)
        # 반복 횟수 도달 시 종료
        if i >= iters - 1:
            break

    # epoch 전체 학습에 걸린 시간 계산   
    A = time.time() - A
    # iteration당 평균 시간(ms) 출력 (전체, 순전파, 역전파)
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg,1000*b_m.avg))
    # iteration당 평균 시간 반환
    return A / iters
            
##### 모델 평가 함수 #####
# 모델 평가는 다음과 같은 순서로 진행
# 1. (선택) 모델로 예측 결과를 생성한다.
# 2. COCO evaluator를 만든다.
# 3. 저장된 예측 결과로 AP 같은 지표를 생성한다.
# 4. 출력 결과를 문자열로 모아 반환한다.
#
# @param generate: True이면 1번 과정 수행, False이면 수행하지 않음
def evaluate(model, data_loader, device, args, generate=True):
    # 평가 iteration 관련 정보를 담을 변수 선언
    iter_eval = None
    # 예측 결과 생성
    if generate:
        iter_eval = generate_results(model, data_loader, device, args)

    dataset = data_loader
    # 평가 대상 지정
    # bbox: bounding box AP
    # segm: segmentation mask AP
    iou_types = ["bbox", "segm"]
    # COCO evaluator 생성
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

    results = torch.load(args.results, map_location="cpu")

    S = time.time()
    coco_evaluator.accumulate(results)
    print("accumulate: {:.1f}s".format(time.time() - S))

    # collect outputs of buildin function print
    temp = sys.stdout
    sys.stdout = TextArea()

    coco_evaluator.summarize()

    output = sys.stdout
    sys.stdout = temp
        
    return output, iter_eval
    
    
# generate results file   
@torch.no_grad()   
def generate_results(model, data_loader, device, args):
    iters = len(data_loader) if args.iters < 0 else args.iters
        
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        S = time.time()
        #torch.cuda.synchronize()
        output = model(image)
        m_m.update(time.time() - S)
        
        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
        coco_results.extend(prepare_for_coco(prediction))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
     
    A = time.time() - A 
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    torch.save(coco_results, args.results)
        
    return A / iters
    

