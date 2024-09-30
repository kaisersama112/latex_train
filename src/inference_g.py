import os
import argparse
import cv2 as cv

from pathlib import Path
from onnxruntime import InferenceSession
from models.thrid_party.paddleocr.infer import predict_det, predict_rec
from models.thrid_party.paddleocr.infer import utility

from models.utils import mix_inference
from models.ocr_model.utils.to_katex import to_katex
from models.ocr_model.utils.inference import inference as latex_inference

from models.ocr_model.model.TexTeller import TexTeller
from models.det_model.inference import PredictConfig


class ArgsConfig:
    # 是否输出md格式
    mix = False
    num_beam = 1
    inference_mode = "cuda"
    infer_config = PredictConfig("./models/det_model/model/infer_cfg.yml")
    latex_det_model = InferenceSession("./models/det_model/model/rtdetr_r50vd_6x_coco.onnx")
    det_use_gpu = True
    use_gpu = inference_mode
    SIZE_LIMIT = 20 * 1024 * 1024
    # 文本检测模型
    det_model_dir = "./models/thrid_party/paddleocr/checkpoints/det/default_model.onnx"
    # 文本识别模型
    rec_model_dir = "./models/thrid_party/paddleocr/checkpoints/rec/default_model.onnx"


def latex_rec_model_init():
    return TexTeller.from_pretrained()


def tokenizer_init():
    return TexTeller.get_tokenizer()


argsConfig = ArgsConfig()
image_path = r"Snipaste_2024-09-25_11-17-21.png"
if __name__ == '__main__':
    # args = parser_init()
    print('Loading model and tokenizer...')
    # 数学公式检测模型初始化
    latex_rec_model = latex_rec_model_init()
    # tokenizer 初始化
    tokenizer = tokenizer_init()
    print('Model and tokenizer loaded.')
    img = cv.imread(image_path)
    print('Inference...')

    if not argsConfig.mix:
        res = latex_inference(latex_rec_model, tokenizer, [img], argsConfig.inference_mode, argsConfig.num_beam)
        res = to_katex(res[0])
        print(res)
    else:
        rec_use_gpu = argsConfig.use_gpu and not (os.path.getsize(argsConfig.rec_model_dir) < argsConfig.SIZE_LIMIT)
        paddleocr_args = utility.parse_args()
        paddleocr_args.use_onnx = True
        paddleocr_args.det_model_dir = argsConfig.det_model_dir
        paddleocr_args.rec_model_dir = argsConfig.rec_model_dir

        paddleocr_args.use_gpu = argsConfig.det_use_gpu
        detector = predict_det.TextDetector(paddleocr_args)
        paddleocr_args.use_gpu = rec_use_gpu
        recognizer = predict_rec.TextRecognizer(paddleocr_args)
        lang_ocr_models = [detector, recognizer]
        latex_rec_models = [latex_rec_model, tokenizer]
        res = mix_inference(image_path, argsConfig.infer_config, argsConfig.latex_det_model, lang_ocr_models,
                            latex_rec_models,
                            argsConfig.inference_mode, argsConfig.num_beam)
        print("-------------------------------------")
        print(res)
        print("-------------------------------------")
