from pathlib import Path

from ...globals import (
    VOCAB_SIZE,
    FIXED_IMG_SIZE,
    IMG_CHANNELS,
    MAX_TOKEN_SIZE
)

from transformers import (
    RobertaTokenizerFast,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig
)


class TexTeller(VisionEncoderDecoderModel):
    def __init__(self):
        config = VisionEncoderDecoderConfig.from_pretrained(Path(__file__).resolve().parent / "config.json")
        config.encoder.image_size = FIXED_IMG_SIZE
        config.encoder.num_channels = IMG_CHANNELS
        config.decoder.vocab_size = VOCAB_SIZE
        config.decoder.max_position_embeddings = MAX_TOKEN_SIZE

        super().__init__(config=config)

    @classmethod
    def from_pretrained(cls, model_path: str = "./models/ocr_model/train/train_result/checkpoint-230", use_onnx=False,
                        onnx_provider=None):
        model_path = Path(model_path).resolve()
        return VisionEncoderDecoderModel.from_pretrained(str(model_path))

    @classmethod
    def get_tokenizer(cls,
                      tokenizer_path: str = "./models/ocr_model/train/train_result/checkpoint-230") -> RobertaTokenizerFast:
        tokenizer_path = Path(tokenizer_path).resolve()
        return RobertaTokenizerFast.from_pretrained(str(tokenizer_path))
