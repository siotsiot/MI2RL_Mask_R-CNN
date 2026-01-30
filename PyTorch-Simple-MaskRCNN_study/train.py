import bisect
import glob
import os
import re
import time

import torch

import pytorch_mask_rcnn as pmr

# ------------------------------------------------------------- #
#                            main 함수                           #
# ------------------------------------------------------------- #
def main(args):

    '''
    학습 전 어떤 장치(CPU 또는 CUDA(GPU))를 사용하는지 정하는 코드
    '''
    # - torch.device
    # PyTorch에서 텐서와 모델이 올라갈 장치(device)를 표현하는 객체
    # 모델과 텐서가 모두 같은 device에 있어야 연산 가능
    
    # - if torch.cuda.is_available() and args.use_cuda
    # torch.cuda.is_available(): 현재 실행 환경에서 CUDA(GPU)를 쓸 수 있는지
    # args.use_cuda: 사용자가 명시적으로 CUDA를 사용하겠다고 말했는지, 즉 명령줄에 --use-cuda가 있어야 True

    # - else "cpu"
    # CUDA를 사용할 수 없거나 사용자가 원치 않으면 CPU를 사용

    # -> device 객체에는 "cuda" 또는 "cpu" 문자열이 들어감
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    # - if device.type == "cuda":
    # device는 객체이므로 device.type으로 장치 종류를 문자열로 가져옴
    # 즉 현재 프로그램이 CUDA(GPU)를 사용 중이라면

    # - pmr.get_gpu_prop(show=True)
    # pmr 모듈의 정의된 get_gpu_prop() 유틸 함수로 GPU 속성 정보를 출력
    # show=Ture로 설정하여 현재 사용 중인 GPU의 세부 정보를 화면에 출력
    if device.type == "cuda":
        pmr.get_gpu_prop(show=True)

    # 현재 사용 중인 장치 정보 출력
    print("\ndevice: {}".format(device))
    
    # ------------------------------- prepare data loader ------------------------------- #

    ##### 학습용 데이터셋 생성 #####
    # @function pmr.datasets:
    #   pytorch_mask_rcnn/datasets/utils.py의 datasets 함수
    #   PyTorch 기본 Dataset이 아님
    #
    # @param args.dataset:
    #   사용할 데이터셋 종류("coco" 또는 "voc")
    #
    # @param args.data_dir:
    #   데이터셋이 저장된 디렉터리 경로
    #
    # @param "train2017":
    #   coco 데이터셋 중 학습용 데이터셋 디렉터리 이름
    #
    # @param train=True
    #   데이터셋이 image(영상)와 target(정답)을 모두 로드할지 여부
    #   "train2017" 디렉터리에는 영상과 정답이 함께 있음
    #   train=True  -> image + target 반환 (학습 / 검증 시 사용)
    #   train=False -> image만 반환 (inference에 사용)
    #
    # @variable dataset_train:
    #   학습용 데이터셋 객체
    dataset_train = pmr.datasets(args.dataset, args.data_dir, "train2017", train=True)

    ##### 무작위로 섞인 학습 데이터셋 생성 #####
    # @function torch.randperm:
    #   학습 데이터셋의 인덱스를 무작위로 섞음 (shuffle)
    #   0부터 dataset_train의 전체 데이터 개수 - 1까지의 정수를 무작위로 섞은 텐서를 반환
    #   데이터를 섞는 이유: 데이터 순서가 고정되면 특정 패턴에 과적합될 수 있음
    #
    # @param len(dataset_train):
    #   dataset_train의 전체 데이터 개수
    #
    # @method .tolist():
    #   텐서 객체를 Python의 리스트로 변환
    #   이후 Subset에서 리스트를 사용하기 때문
    #
    # @variable indices:
    #   무작위로 섞인 학습 데이터셋의 인덱스(리스트 형태)
    indices = torch.randperm(len(dataset_train)).tolist()

    ##### 학습용 / 평가용 데이터셋 분리 #####
    # @class torch.utils.data.Subset:
    #   기존 데이터셋을 감싸서(wrap) 새로운 데이터셋처럼 동작하게 만드는 PyTorch 유틸리티 클래스
    #
    # @param dataset_train:
    #   원본 학습용 데이터셋 객체(coco의 train2017)
    #
    # @param indices:
    #   사용할 데이터 인덱스 리스트(현재 무작위의 숫자가 들어 있음)
    #
    # @variable d_train:
    #   무작위로 섞인 학습용 데이터셋 객체
    d_train = torch.utils.data.Subset(dataset_train, indices)

    ##### 평가용 데이터셋 생성 #####
    # @function pmr.datasets:
    #   pytorch_mask_rcnn/datasets/utils.py의 datasets 함수
    #   PyTorch 기본 Dataset이 아님
    #
    # @param args.dataset:
    #   사용할 데이터셋 종류("coco" 또는 "voc")
    #
    # @param args.data_dir:
    #   데이터셋이 저장된 디렉터리 경로
    #
    # @param "val2017":
    #   coco 데이터셋 중 평가용 데이터셋 디렉터리 이름
    #
    # @param train=True
    #   데이터셋이 image(영상)와 target(정답)을 모두 로드할지 여부
    #   "val2017" 디렉터리에는 영상과 정답이 함께 있음
    #   검증 역시 학습과 마찬가지로 정답이 필요하므로 train=True로 설정
    #   train=True  -> image + target 반환 (학습 / 검증 시 사용)
    #   train=False -> image만 반환 (inference에 사용)
    #
    # @variable d_test:
    #   평가용 데이터셋 객체
    d_test = pmr.datasets(args.dataset, args.data_dir, "val2017", train=True) # set train=True for eval
    
    ##### warmup iters 설정 #####
    # warmup iters: 학습 초기에 학습률을 점진적으로 증가시키는 단계에서 사용할 iteration 수
    # 너무 큰 학습률로 학습을 시작하면 모델이 불안정하므로 최초 1 epoch 동안 학습률을 서서히 증가
    # case 1: warmup iters보다 학습 데이터셋의 개수가 더 큰 경우
    #   1 epoch 동안 warmup 진행
    # case 2: warmup iters보다 학습 데이터셋의 개수가 더 작은 경우
    #   최소 1000 iteration 동안 warmup 진행
    #
    # @param 1000:
    #   최소 웜업 iteration 수
    # @param len(d_train):
    #   학습용 데이터셋의 전체 개수
    # @variable args.warmup_iters:
    #   warmup에 사용할 iteration 수
    args.warmup_iters = max(1000, len(d_train))

    ##### 프로그램 실행 시 출력되는 args 객체 내용 #####
    # @variable args:
    #   커맨드라인에서 설정한 각종 파라미터들이 속성 출력
    #   예: args.lr, args.momentum, args.epochs, args.print_freq 등
    print(args)

    # -------------------------------------------------------------------------- #
    ##### 클래스 개수 계산 #####
    # @param d_train.dataset.classes:
    #   d_train.dataset은 원본 데이터셋(dataset_train)을 의미
    #   .classes는 원본 데이터셋의 클래스 레이블 리스트
    #   예: {1: "person", 2: "bicycle", ...}
    #   딕셔너리를 max() 함수에 넣으면 가장 큰 키 값을 반환
    #   여기에 1을 더하면 배경 클래스를 포함한 전체 클래스 개수
    # @variable num_classes:
    #   모델의 출력 클래스 개수(배경 + 실제 클래스))
    num_classes = max(d_train.dataset.classes) + 1 # including background class

    ##### 모델 생성 #####
    # ResNet-50 + FPN을 Backbone으로 사용하는 Mask R-CNN 모델 생성
    # 논문에서 말하는 다음 구조를 코드로 구현
    #         Input Image
    #              ↓
    #      Backbone (ResNet-50)
    #              ↓
    # Feature Pyramid Network (FPN)
    #              ↓
    # RPN (Region Proposal Network)
    #              ↓
    #           RoIAlign
    #              ↓
    # ┌────────────────┬────────────────┬───────────────┐
    # │ classification │ box regression │   mask head   │
    # └────────────────┴────────────────┴───────────────┘
    # @function pmr.maskrcnn_resnet50:
    #   pytorch_mask_rcnn/models/maskrcnn.py의 maskrcnn_resnet50 함수
    #   ResNet-50 + FPN을 백본으로 하는 Mask R-CNN 모델을 반환
    #
    # @param True:
    #   COCO 데이터셋으로 사전 학습된(pretrained) ResNet-50 백본을 사용
    #
    # @param num_classes:
    #   모델의 출력 클래스 개수(배경 + 실제 클래스)
    #
    # @method .to(device):
    #   모델을 지정한 장치(CPU 또는 CUDA(GPU))로 이동
    model = pmr.maskrcnn_resnet50(True, num_classes).to(device)

    ##### 파라미터 중 학습 가능한(gradient 계산이 필요한) 파라미터만 추출 #####
    # @method model.parameters():
    #   모델의 모든 파라미터(가중치와 편향)를 반환
    #
    # @condition p.requires_grad:
    #   파라미터 p가 학습 가능한지 여부
    #   .requires_grad는 이 값이 학습 중에 바뀌어야 하는 대상
    #   True -> 학습 대상
    #   False -> 고정(사전학습된 가중치 등)
    params = [p for p in model.parameters() if p.requires_grad]

    ##### 옵티마이저 생성 #####
    # gradient를 통해 파라미터를 실제로 업데이트 하는 역할
    # @class torch.optim.SGD:
    #   PyTorch에서 제공하는 확률적 경사 하강법(SGD)
    #   torch: PyTorch 라이브러리
    #   optim: PyTorch의 최적화(optimization) 관련 모듈
    #   SGD: 확률적 경사 하강법(Stochastic Gradient Descent) 클래스
    #   -> SGD 방식으로 파라미터를 업데이트하는 도구 생성
    #
    # @arg params:
    #   학습 가능한 파라미터 리스트
    #
    # @arg lr=args.lr:
    #   학습률(learning rate)
    #
    # @arg momentum=args.momentum:
    #   이전 움직임을 얼마나 반영할지 결정하는 모멘텀 계수
    #   아까 가던 방향이면 더 밀기
    #
    # @arg weight_decay=args.weight_decay:
    #   파라미터가 너무 커지지 않게 제동 거는 장치
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    ##### 0.1 배수로 학습률 감소하는 스케줄러 생성 #####
    # 람다식: x를 받아서 오른쪽 계산  결과를 돌려주는 함수
    # def lr_lambda(x):
    #     return 0.1 ** bisect.bisect(args.lr_steps, x)
    # 위 코드와 동일
    # @param x:
    #  현재 epoch 번호
    # 
    # @function bisect.bisect:
    #   x가 lr_steps에서 몇 번째 구간에 속하는지 알려줌
    #   @args.lr_steps 리스트에서 x가 들어갈 위치의 인덱스를 반환
    lr_lambda = lambda x: 0.1 ** bisect.bisect(args.lr_steps, x)

    # 시작 epoch 번호 초기화
    start_epoch = 0
    
    # find all checkpoints, and load the latest checkpoint
    # 이전에 저장된 체크포인트가 있으면 모델, 옵티마이저, 에포크 상태를 전부 복원 후 이어서 학습할 준비를 함
    prefix, ext = os.path.splitext(args.ckpt_path)
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device) # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]
        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))
    
    # ------------------------------- train ------------------------------------ #
    ##### 학습 루프 #####
    # 한 epoch마다 다음 과정을 수행
    # 1. 현재 epoch에 사용할 learning rate를 계산한다.
    # 2. train_one_epoch로 학습을 수행한다. 이때 가중치 업데이트가 발생한다.
    # 3. evaluate로 검증 데이터셋으로 성능을 측정한다.
    # 4. 시간 / 속도 / AP를 출력한다.
    # 5. 체크포인트를 저장한다.
    # 6. 체크포인트가 너무 많으면 오래된 것을 삭제한다.
    #
    # @loop epoch in range(start_epoch, args.epochs):
    #   start_epoch부터 args.epochs - 1까지 반복
    #   -> 총 몇 epoch 학습할지 결정
    for epoch in range(start_epoch, args.epochs):
        # 현재 epoch 출력 (0부터 시작하므로 +1)
        print("\nepoch: {}".format(epoch + 1))
        
        # 학습 시작 시각 기록
        A = time.time()
        # 현재 epoch에 맞는 학습률 계산
        args.lr_epoch = lr_lambda(epoch) * args.lr
        # 현재 epoch에서 실제로 쓰는 learning rate와 그때 적용된 배율 출력
        print("lr_epoch: {:.5f}, factor: {:.5f}".format(args.lr_epoch, lr_lambda(epoch)))
        # 훈련데이터를 한 epoch 전부 사용해서 모델 학습 후, 그 과정의 반복 횟수를 받아옴
        # @function pmr.train_one_epoch:
        #   pytorch_mask_rcnn/engine.py의 train_one_epoch 함수
        #
        # @param model:
        #   학습할 Mask R-CNN 모델
        #
        # @param optimizer:
        #   모델 파라미터를 업데이트할 옵티마이저
        #
        # @param d_train:
        #   학습용 데이터셋
        #
        # @param device:
        #   모델과 데이터를 올릴 장치(CPU 또는 CUDA(GPU))
        #
        # @param epoch:
        #   현재 epoch 번호
        #
        # @param args:
        #   각종 하이퍼파라미터가 들어 있는 객체
        iter_train = pmr.train_one_epoch(model, optimizer, d_train, device, epoch, args)
        # 학습 종료 시각 기록
        A = time.time() - A
        
        # 평가 시작 시각 기록
        B = time.time()
        # 학습이 끝난 모델을 가지고 테스트 및 검증 데이터로 성능 측정
        #
        # @function pmr.evaluate:
        #   pytorch_mask_rcnn/engine.py의 evaluate 함수
        #
        # @param model:
        #   평가할 Mask R-CNN 모델
        #
        # @param d_test:
        #   평가용 데이터셋
        #
        # @param device:
        #   모델과 데이터를 올릴 장치(CPU 또는 CUDA(GPU))
        #
        # @param args:
        #   각종 하이퍼파라미터가 들어 있는 객체
        eval_output, iter_eval = pmr.evaluate(model, d_test, device, args)
        # 평가 종료 시각 기록
        B = time.time() - B

        # 코드에서 epoch은 0부터 시작하므로 +1
        trained_epoch = epoch + 1
        print("training: {:.1f} s, evaluation: {:.1f} s".format(A, B))
        pmr.collect_gpu_info("maskrcnn", [1 / iter_train, 1 / iter_eval])
        print(eval_output.get_AP())

        pmr.save_ckpt(model, optimizer, trained_epoch, args.ckpt_path, eval_info=str(eval_output))

        # it will create many checkpoint files during training, so delete some.
        # 체크포인트 파일을 만들어서 현재까지 학습한 가중치 저장
        prefix, ext = os.path.splitext(args.ckpt_path)
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        n = 10
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.system("rm {}".format(ckpts[i]))
        
    # -------------------------------------------------------------------------- #

    print("\ntotal time of this training: {:.1f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))
    
# ------------------------------------------------------------- #
#                            코드 진입점                           #
# ------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    ## 실험 파라미터 선언부
    parser = argparse.ArgumentParser()

    ## GPU 사용 여부를 사용자가 선택
    # @param use_cuda: CUDA(GPU) 사용 여부
    # @param action: 'store_true' 옵션은 해당 플래그가 명령줄에 존재하면 True로 설정됨
    parser.add_argument("--use-cuda", action="store_true")
    
    ## 데이터셋
    # @param --dataset: 사용할 데이터셋 종류 (기본값: coco)
    parser.add_argument("--dataset", default="coco", help="coco or voc")

    # @param --data-dir: 데이터셋이 저장된 디렉토리 경로
    parser.add_argument("--data-dir", default="E:/PyTorch/data/coco2017")

    # @param --ckpt-path: 모델 가중치 저장 경로
    parser.add_argument("--ckpt-path")

    # @param --results: evaluation 결과 저장 경로
    parser.add_argument("--results")
    
    ## 하이퍼파라미터
    # 난수 고정용 시드 설정하는 코드로 재현성(Reproducibility)을 위해 사용
    # @param --seed: 랜덤 시드 값
    parser.add_argument("--seed", type=int, default=3)
    
    # @param --lr-steps: 학습률 감소 시점(epoch) 리스트
    # @param nargs +: 하나 이상의 값을 받을 수 있도록 설정
    # 없으면 기본값인 6-7로 설정
    # 학습률이 6에서 한 번, 7에서 한 번 감소
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[6, 7])

    # @param --lr: 초기 학습률
    # 학습률: 가중치를 업데이트할 때 얼마나 크게 조정할지를 결정하는 하이퍼파라미터
    # 학습률은 실수여야 하므로 float로 설정
    parser.add_argument("--lr", type=float)

    # @param --momentum: SGD 옵티마이저 모멘텀 계수, 기본값은 0.9
    # 모멘텀: 이전에 내려가던 방향을 기억해서 관성을 부여하는 역할
    # SGD: 확률적 경사 하강법(Stochastic Gradient Descent)
    parser.add_argument("--momentum", type=float, default=0.9)

    # @param --weight-decay: 가중치 감쇠(Weight Decay) 계수, 기본값은 0.0001
    # 가중치 감쇠: 가중치가 너무 커지는 것을 억제하는 정규화 기법
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    
    # @param --epochs: 전체 학습 epoch 수, 기본값은 3
    # epoch: 전체 데이터셋을 한 번 학습하는 주기
    parser.add_argument("--epochs", type=int, default=3)

    # @param --iters: epoch 당 최대 반복 횟수, 기본값은 10
    # -1로 설정하면 자동으로 결정
    # iteration: 미니배치 1개를 처리하는 단위
    # 미니배치(mini-batch): 전체 데이터 중 일부를 묶어서 한 번에 학습에 사용하는 단위
    parser.add_argument("--iters", type=int, default=10, help="max iters per epoch, -1 denotes auto")
    
    # 학습 중 손실(loss)를 몇 iteration마다 한 번 출력할지 정하는 코드
    # @param --print-freq: 손실 출력 빈도(커맨드라인 옵션 이름), args.print_freq로 접근
    # @param type=int: 출력 주기는 정수, 예: 1, 10, 50, 100, 1000, ...
    # @param default=100: 옵션을 주지 않으면 args.print_freq는 100, 즉 iteration 100번당 한 번 로그 출력
    # @param help: loss를 출력하는 빈도 설명
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")

    # 위에서 정의한 모든 parser.add_argument() 옵션들을 파싱하여 args 객체에 저장
    # .vscode/launch.json에서 커맨드라인 인자를 설정 가능
    # 이후 args.lr, args.momentum 등으로 접근 가능
    args = parser.parse_args()
    
    # 사용자가 --lr 옵션을 안 줬으면 다음과 같이 설정
    # 논문에 나온 값: 초기 학습률: 0.02, 배치 크기: 16
    if args.lr is None:
        args.lr = 0.02 * 1 / 16
    
    # 모델과 옵티마이저 상태를 저장하는 체크포인트 파일 경로, 예: maskrcnn_coco.pth
    # 사용자가 --ckpt-path 옵션을 안 줬으면 다음과 같이 설정
    # 예:
    #   args.dataset = "coco" -> ./maskrcnn_coco.pth
    #   args.dataset = "voc"  -> ./maskrcnn_voc.pth
    if args.ckpt_path is None:
        args.ckpt_path = "./maskrcnn_{}.pth".format(args.dataset)

    # 평가 결과(AP, 정확도 등)를 저장하는 파일 경로
    if args.results is None:
        # 체크포인트 파일과 동일한 디렉터리에 저장
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.pth")
    
    # main 함수 호출
    # 객체 agrs에는 커맨드라인에서 설정한 값이 들은 상태로 main 함수로 전달
    main(args)

