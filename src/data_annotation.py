import json
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
    inference_mode = "cpu"
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
    return TexTeller.from_pretrained("./models/ocr_model/train/train_result/checkpoint-230")


def tokenizer_init():
    return TexTeller.get_tokenizer("./models/ocr_model/train/train_result/checkpoint-230")


def infer_image(image_path):
    img = cv.imread(image_path)
    if not argsConfig.mix:
        res = latex_inference(latex_rec_model, tokenizer, [img], argsConfig.inference_mode, argsConfig.num_beam)
        res = to_katex(res[0])
        return res


argsConfig = ArgsConfig()
base_image_path = "../new_data/latex_zh_anno/matrix_data/images"
output_file = "../new_data/latex_zh_anno/matrix_data.jsonl"


def write_latex(img_name, formula):
    # 续写(需要注意)
    with open(output_file, 'a+', encoding='utf-8') as f:
        entry = {"img_name": img_name, "formula": formula}
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')
    f.close()


if __name__ == '__main__':
    print('Loading model and tokenizer...')
    latex_rec_model = latex_rec_model_init()
    tokenizer = tokenizer_init()
    print('Model and tokenizer loaded.')
    data_name_list = os.listdir(base_image_path)
    print(data_name_list)
    for data_path in data_name_list:
        res = infer_image(os.path.join(base_image_path, data_path))
        write_latex(data_path, res)
