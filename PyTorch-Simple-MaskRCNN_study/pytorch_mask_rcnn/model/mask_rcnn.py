from collections import OrderedDict

import torch.nn.functional as F
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.ops import misc

from .utils import AnchorGenerator
from .rpn import RPNHead, RegionProposalNetwork
from .pooler import RoIAlign
from .roi_heads import RoIHeads
from .transform import Transformer

##### Mask R-CNN 모델 클래스 정의 #####
# - Mask R-CNN이란?
#   R-CNN 계열의 객체 검출 모델
#   Fast R-CNN + RPN + 마스크 분류기
#   자세한 부분은 Notion 참고
#
# @class MaskRCNN:
#   Mask R-CNN 모델을 정의한 클래스
#   nn.Module을 상속
class MaskRCNN(nn.Module):
    """
    Implements Mask R-CNN.

    The input image to the model is expected to be a tensor, shape [C, H, W], and should be in 0-1 range.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensor, as well as a target (dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [xmin, ymin, xmax, ymax] format, with values
          between 0-H and 0-W
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

    The model returns a Dict[Tensor], containing the classification and regression losses 
    for both the RPN and the R-CNN, and the mask loss.

    During inference, the model requires only the input tensor, and returns the post-processed
    predictions as a Dict[Tensor]. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [xmin, ymin, xmax, ymax] format, 
          with values between 0-H and 0-W
        - labels (Int64Tensor[N]): the predicted labels
        - scores (FloatTensor[N]): the scores for each prediction
        - masks (FloatTensor[N, H, W]): the predicted masks for each instance, in 0-1 range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (mask >= 0.5)
        
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
        num_classes (int): number of output classes of the model (including the background).
        
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_num_samples (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors during training of the RPN
        rpn_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_num_samples (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals during training of the 
            classification head
        box_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_num_detections (int): maximum number of detections, for all classes.
        
    """
    ##### Mask R-CNN 생성자 #####
    # @param self:
    #   MaskRCNN 클래스의 인스턴스 (현재 객체)
    #
    # @param backbone:
    #   특징 추출기 역할
    #
    # @param num_classes:
    #   모델이 예측해야 하는 클래스 수 (클래스 수 + 배경 1개)
    #
    # ====================================== 용어 ======================================
    # GT(Ground Truth): 실제 정답 데이터
    #
    # GT box: 사람이 준 정답 경계 박스
    #
    # Box(Bounding Box라고도 함): 사각형 영역으로 물체 위치를 대략적으로 표현
    #   ㄴ 물체 위치를 대략적으로 표현
    #   ㄴ segmentation 전에 위치를 좁히는 역할
    #
    # Mask: 픽셀 단위 정답
    #   ㄴ 물체의 정확한 윤곽을 표현
    #
    # Foreground / Background: 물체의 분류 기준
    #   ㄴ Foreground: 관심 있는 물체 (예: 폐)
    #   ㄴ Background: 관심 없는 물체 (그 외 전부)
    #
    # Positive / Negative:
    #   ㄴ Positive: 여기에 foreground가 있음
    #   ㄴ Negative: 여기에 foreground가 없음
    #
    # Anchor: 모델이 미리 깔아두는 박스 후보 템플릿
    #   ㄴ 이미지 전체에 규칙적으로 배치
    #   ㄴ 크기와 비율만 다르고, 위치는 고정
    #
    # IoU(Intersection over Union): 두 박스(Anchor와 GT box)가 얼마나 겹치는지 수치로 나타내는 점수
    #   ㄴ 0-1 사이의 값
    #   ㄴ 1에 가까울수록 두 박스가 많이 겹침
    #
    # RPN(Region Proposal Network): 여기에 물체가 있을 것 같다고 proposal(물체 후보 박스)를 뽑아주는 신경망
    #   ㄴ Faster R-CNN에서 도입
    #
    # Proposal: RPN이 뽑아준 물체 후보 박스
    #   ㄴ Anchor를 바탕으로 물체가 있을 법한 위치를 제공
    #   ㄴ Anchor → Refinement → Proposal
    #   ㄴ 아직 최종 예측이 아님
    # 
    # NMS(Non-Maximum Suppression): 겹치는 proposal(물체 후보 박스) 제거 알고리즘
    #   ㄴ 점수가 높은 박스만 남기고, 다른 겹치는 박스는 제거
    #
    # RoIHeads(Region of Interest Head): RPN이 뽑아준 proposal(물체 후보 박스)을 다듬고, 분류 및 마스크 생성을 담당하는 신경망
    #   ㄴ proposal → RoI Pooling → 분류기 + 회귀기 + 마스크 분류기
    #
    # Objectness Score: 여기에 물체가 있을 확률 점수
    #   ㄴ RPN이 각 Anchor에 대해 예측
    #   ㄴ Foreground 또는 Background 판단용
    #
    # ================================================================================
    #
    ### RPN 매개변수 ###
    # @param rpn_fg_iou_thresh:
    #   Anchor와 GT box 간의 IoU값이 이 값 이상이면 Positive로 간주
    #   즉 Anchor와 GT box ioU >= 0.7 → Positive
    #
    # @param rpn_bg_iou_thresh:
    #   Anchor와 GT box 간의 IoU값이 이 값 이하이면 Negative로 간주
    #   즉 Anchor와 GT box ioU <= 0.3 → Negative
    #   이때 0.3-0.7 구간은 무시
    #   이유는 모호한 Anchor값으로 학습시 RPN이 헷갈림
    #
    # @param rpn_num_samples:
    #   RPN이 한 번 학습할 때 샘플링하는 Anchor 개수
    #   영상 한 장이 들어오면 RPN은 약 20,000개 정도의 앵커(Anchors)를 생성
    #   만약 모든 앵커(20,000개)를 전부 학습하면 너무 오래 걸리고, 학습이 편향됨 (배경이 물체보다 많을 경우 모델이 배경이라고 예측)
    #   그래서 일부 앵커만 사용 (주로 256개 사용)
    #
    # @param rpn_positive_fraction:
    #   RPN이 샘플링한 앵커 중에서 Positive 앵커(물체) 비율
    #   예: 0.5 → 256개 앵커 중에서 128개는 Positive 앵커(물체), 128개는 Negative 앵커(배경)
    #   만약 Positive 앵커가 부족하면 Negative 앵커로 채움
    #   예: Positive 앵커가 100개밖에 없으면, 나머지 156개는 Negative 앵커로 채움
    #   주로 0.5로 설정
    # 
    # @param rpn_reg_weights:
    #   좌표 보정 중요도 비율
    #
    # @param rpn_pre_nms_top_n_train:
    def __init__(self, backbone, num_classes,
                 # RPN parameters
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_num_samples=256, rpn_positive_fraction=0.5,
                 rpn_reg_weights=(1., 1., 1., 1.),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 # RoIHeads parameters
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_num_samples=512, box_positive_fraction=0.25,
                 box_reg_weights=(10., 10., 5., 5.),
                 box_score_thresh=0.1, box_nms_thresh=0.6, box_num_detections=100):
        # nn.Module(부모 클래스) 생성자 호출
        super().__init__()
        # Backbone 설정, 여기선 ResNet-50
        self.backbone = backbone
        # backbone이 출력하는 특징 맵의 채널 수, 여기선 256
        out_channels = backbone.out_channels
        
        #------------- RPN --------------------------
        # anchor의 기본 크기 설정
        # 128px, 256px, 512px 크기의 anchor 사용
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
        # anchor의 가로:세로 비율
        # 0.5(1:2) 세로로 긴 박스, 1(1:1) 정사각형, 2(2:1) 가로로 긴 박스 비율의 anchor 사용
        anchor_ratios = (0.5, 1, 2)
        # 특징 맵의 한 위치당 9개의 anchor 생성
        # @param len(anchor_sizes):
        #
        #   anchor_sizes 튜플의 길이 (3)
        # @param len(anchor_ratios):
        #
        #   anchor_ratios 튜플의 길이 (3)
        # @return:
        #   3 * 3 = 9개의 anchor
        num_anchors = len(anchor_sizes) * len(anchor_ratios)
        # RPN용 AnchorGenerator 생성
        # anchor를 실제 좌표로 생성하는 객체 생성
        # 실제 anchor 생성은 forward()에서 수행
        # 아직 anchor가 생성되지는 않음
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
        # RPN의 실제 신경망 부분
        rpn_head = RPNHead(out_channels, num_anchors)
        
        # NMS 전에 유지할 proposal 개수 설정
        # 훈련 시와 테스트 시 각각 다르게 설정 -> 딕셔너리 사용
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        # NMS 후에 유지할 proposal 개수 설정
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        # RPN 생성
        self.rpn = RegionProposalNetwork(
             rpn_anchor_generator, rpn_head, 
             rpn_fg_iou_thresh, rpn_bg_iou_thresh,
             rpn_num_samples, rpn_positive_fraction,
             rpn_reg_weights,
             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        
        #------------ RoIHeads --------------------------
        ##### RoI Align #####
        # @param output_size:
        #   RoIAlign이 출력하는 특징 맵의 크기
        #
        # @param sampling_ratio:
        #   샘플링 비율
        box_roi_pool = RoIAlign(output_size=(7, 7), sampling_ratio=2) # RoIAlign 클래스의 인스턴스
        # RoIAlign 결과의 공간 해상도 한 변의 길이
        resolution = box_roi_pool.output_size[0] # 7
        # FC층에 넣기 위해 평탄화(Flatten)
        # @variable in_channels:
        #   박스 분류기(Fast R-CNN)의 입력 채널 수
        in_channels = out_channels * resolution ** 2 # 256 * 7 * 7 = 12544
        # 박스 분류기 들어가기 전에 한 번 거치는 은닉 표현의 차원 수
        # 바로 분류하지 않고, 한 번 더 변환(압축 및 정제) 후, 분류
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)
        
        self.head = RoIHeads(
             box_roi_pool, box_predictor,
             box_fg_iou_thresh, box_bg_iou_thresh,
             box_num_samples, box_positive_fraction,
             box_reg_weights,
             box_score_thresh, box_nms_thresh, box_num_detections)
        
        self.head.mask_roi_pool = RoIAlign(output_size=(14, 14), sampling_ratio=2)
        
        layers = (256, 256, 256, 256)
        dim_reduced = 256
        self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)
        
        #------------ Transformer -------------
        self.transformer = Transformer(
            min_size=800, max_size=1333, 
            image_mean=[0.485, 0.456, 0.406], 
            image_std=[0.229, 0.224, 0.225])
        
    def forward(self, image, target=None):
        ori_image_shape = image.shape[-2:]
        
        image, target = self.transformer(image, target)
        image_shape = image.shape[-2:]
        feature = self.backbone(image)
        
        proposal, rpn_losses = self.rpn(feature, image_shape, target)
        result, roi_losses = self.head(feature, proposal, image_shape, target)
        
        if self.training:
            return dict(**rpn_losses, **roi_losses)
        else:
            result = self.transformer.postprocess(result, image_shape, ori_image_shape)
            return result
        
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)
        self.bbox_pred = nn.Linear(mid_channels, num_classes * 4)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)

        return score, bbox_delta        
    
    
class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        """
        Arguments:
            in_channels (int)
            layers (Tuple[int])
            dim_reduced (int)
            num_classes (int)
        """
        
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features
        
        d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['relu5'] = nn.ReLU(inplace=True)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        super().__init__(d)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                
##### ResNet-50 backbone 정의 클래스 #####
# 특징 추출기 역할
# 모든 backbone 클래스는 nn.Module을 상속
class ResBackbone(nn.Module):
    ##### ResNet-50 backbone 생성자 #####
    # @paam self:
    #   ResBackbone 클래스의 인스턴스
    #   Python에서는 메소드에 넣을 때 항상 첫 번째 인수로 self를 넣어야 함 (Python 문법)
    #
    # @param backbone_name:
    #   어떤 ResNet 모델을 사용할지 지정
    #   예: 'resnet50': 깊이 50층짜리 ResNet 모델 사용
    #
    # @param pretrained:
    #   ResNet-50 backbone만 ImageNet 사전학습 가중치 사용
    #   True: ImageNet 사전학습 가중치 사용
    #   False: ImageNet 사전학습 가중치 사용 안 함
    def __init__(self, backbone_name, pretrained):
        ##### nn.Module의 생성자 호출 #####
        # ResBackbone 클래스는 nn.Module을 상속받은 자식클래스이므로 부모클래스의 생성자를 호출해야 함
        super().__init__()
        
        # ResNet-50 모델 생성 #####
        # @class models.resnet.__dict__:
        #   model.sresnet: torchvision 안에 ResNet 모음으로 내부적으로는 객체 형태의 딕셔너리(__dict__)로 저장되어 있음
        #   backbone_name 문자열에 해당하는 함수 동적으로 선택 후 호출
        #
        # 인수들은 키워드 인수로 함수의 매개변수를 명시적으로 적음
        # @param pretrained:
        #   ImageNet 사전학습 가중치 사용 여부
        # 
        # @param misc.FrozenBatchNorm2d:
        #   BatchNorm: 신경망에서 한 층을 지날 때마다 데이터의 분포가 바뀌는 현상을 막기 위해 사용하는 방법
        #   신경망이 깊어질수록 데이터 분포 변화가 심해져서 학습이 어려워지는 현상(내부 공변량 변화)을 줄여줌
        #   FrozenBatchNorm2d: BatchNorm을 고정 (정규화는 하지만 기준은 고정)
        #   사용 이유:
        #     Detection / Segmetation은 Batch Size가 작음
        #     Batch Size가 작은 상태로 BatchNorm을 사용하면 평균/분산이 매번 바뀌면서 불안정한 상태가 됨
        #     따라서 BatchNorm을 고정시킨 FrozenBatchNorm2d 사용
        body = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
        
        for name, parameter in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
                
        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = 2048
        self.out_channels = 256
        
        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, 1)
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        for module in self.body.values():
            x = module(x)
        x = self.inner_block_module(x)
        x = self.layer_block_module(x)
        return x

##### 모델을 생성하는 팩토리 함수 #####
# 팩토리 함수: 객체를 만들어서 반환하는 함수
# 
# - ResNet-50 backbone을 사용
#   backbone: 특징을 뽑는 뼈대 CNN
#   깊이 50층짜리
#
# - Mask R-CNN 모델 생성
#   RPN + 박스 다듬기 + 마스크 만들기
#   이걸 ResNet-50 backbone 위에 얹음
#
# - 필요 시 COCO 사전학습 가중치 로드
#   COCO 데이터셋으로 학습한 모델은 이미 윤곽, 경계, 객체 등의 패턴을 배운 상태
#   처음부터 자연 영상을 보여줬을 때 배경인지 물체인지 빠르게 구분 가능
#   즉 학습이 더 빠르고 안정적
#
# - num_classes에 맞게 일부 가중치만 골라서 로드
#   COCO 데이터셋의 클래스 수와 내가 해결해야 할 문제의 클래스 수가 다르기 때문
#   COCO는 80개 클래스 + 배경(1개) = 81개, 내가 해결해야 하는 건 폐 Segmentation = 2개
#   즉 COCO 사전학습 가중치를 전부 다 가져오는 것이 아닌 num_classes와 무관한 가중치만 로드
#   - 가져오는 것: ResNet-50 backboe, FPN, RPN
#   - 가져오지 않는 것: 박스 분류기, 박스 회귀기, 마스크 분류기
#
# @param pretrained:
#   COCO 사전학습 가중치 사용 여부
#   True: COCO 사전학습 가중치 사용
#   False: COCO 사전학습 가중치 사용 안 함 -> 전부 무작위로 초기화
#
# @param num_classes:
#   모델이 예측해야 하는 클래스 수 (배경 포함)
#   예) 폐 Segmentation 문제: 2 (폐, 배경)
# 
# @param pretrained_backbone:
#   ResNet-50 backbone만 ImageNet 사전학습 가중치 사용 여부
#   첫 번째 매개변수 pretraine에는 COCO 사전학습 가중치를 사용하는데, 이미 ResNet-50 backbone은 ImageNet 사전학습 가중치가 포함되어 있음
def maskrcnn_resnet50(pretrained, num_classes, pretrained_backbone=True):
    """
    Constructs a Mask R-CNN model with a ResNet-50 backbone.
    
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017.
        num_classes (int): number of classes (including the background).
    """
    
    ##### COCO 사전학습 가중치 사용시 backbone은 ImageNet 사전학습 가중치 사용 안 함 #####
    # COCO 사전학습 가중치에는 이미 ResNet-50 backbone의 ImageNet 사전학습 가중치가 포함되어 있기 때문
    if pretrained:
        backbone_pretrained = False

    ##### ResNet-50 backbone 생성 #####
    # @class ResBackbone:
    #   ResNet-50 backbone을 정의한 클래스
    #
    # @param 'resnet50':
    #   어떤 ResNet 모델을 사용할지 지정
    #
    # @param pretrained_backbone:
    #   ResNet-50 backbone만 ImageNet 사전학습 가중치 사용
    #   True: ImageNet 사전학습 가중치 사용
    #   False: ImageNet 사전학습 가중치 사용 안 함 -> 무작위로 초기화
    #
    # @return:
    #   ResBackbone 클래스의 인스턴스(ResNet-50 구조의 모델)
    backbone = ResBackbone('resnet50', pretrained_backbone)

    ##### Mask R-CNN 모델 생성 #####
    # @class MaskRCNN:
    #   Mask R-CNN 모델을 정의한 클래스
    # 
    # @param backbone:
    #   바로 위 코드에서 생성한 ResNet-50 구조 backbone
    # 
    # @param num_classes:
    #   모델이 예측해야 하는 클래스 수 (배경 포함)
    model = MaskRCNN(backbone, num_classes)
    
    ##### COCO 사전학습 가중치 로드 #####
    # COCO 사전학습 가중치는 torchvision에서 제공하는 URL에서 다운로드
    if pretrained:
        model_urls = {
            'maskrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        }
        model_state_dict = load_url(model_urls['maskrcnn_resnet50_fpn_coco'])
        
        # 
        pretrained_msd = list(model_state_dict.values())
        del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)

        msd = model.state_dict()
        skip_list = [271, 272, 273, 274, 279, 280, 281, 282, 293, 294]
        if num_classes == 91:
            skip_list = [271, 272, 273, 274]
        for i, name in enumerate(msd):
            if i in skip_list:
                continue
            msd[name].copy_(pretrained_msd[i])
            
        model.load_state_dict(msd)
    
    return 