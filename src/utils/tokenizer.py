from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter

def get_tokenizer_converter(config):
    if config.token_type is None:
        tokenizer = None
    elif (
        config.token_type == "bpe"
        or config.token_type == "hugging_face"
        or "whisper" in config.token_type
    ):
        if config.bpemodel is not None:
            tokenizer = build_tokenizer(token_type=config.token_type, bpemodel=config.bpemodel)
        else:
            tokenizer = None
    else:
        tokenizer = build_tokenizer(token_type=config.token_type)

    if config.bpemodel not in ["whisper_en", "whisper_multilingual"]:
        converter = TokenIDConverter(token_list=config.token_list)
    else:
        converter = OpenAIWhisperTokenIDConverter(model_type=config.bpemodel)

    return tokenizer, converter
