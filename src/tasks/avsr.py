import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug

from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.whisper import WhisperFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from src.frontend.conv3d_resnet18.conv3d_resnet18  import Conv3dResNet18

from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs

from src.embedding_for_avsr.default import DefaultEmbeddingLayerForAVSR
from src.embedding_for_avsr.embedding_abs_layer import EmbeddingForAVSRAbsLayer

from src.encoder.audiovisual.audiovisual_abs_encoder import AudioVisualAbsEncoder
from src.encoder.audiovisual.tailored.encoder import TailoredEncoder
from src.encoder.audiovisual.conventional.encoder import ConventionalEncoder

from src.audiovisual_fusion.audiovisual_fusion_abs_module import AudioVisualFusionAbsModule
from src.audiovisual_fusion.adaptive_audiovisual_fusion import AdaptiveAudioVisualFusion

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,
)

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.hugging_face_transformers_decoder import (  # noqa: H301
    HuggingFaceTransformersDecoder,
)
from espnet2.asr.decoder.mlm_decoder import MLMDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.s4_decoder import S4Decoder
from espnet2.asr.decoder.transducer_decoder import TransducerDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,
    DynamicConvolutionTransformerDecoder,
    LightweightConvolution2DTransformerDecoder,
    LightweightConvolutionTransformerDecoder,
    TransformerDecoder,
)
from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder

from src.models.avsr_espnet_model import ESPnetAVSRModel
from src.models.avsr_maskctc_model import AVSRMaskCTCModel

from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import (
    AbsPreprocessor,
    CommonPreprocessor,
    CommonPreprocessor_multi,
)
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none, int_or_none, str2bool, str_or_none

acoustic_frontend_choices = ClassChoices(
    name="acoustic_frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
        fused=FusedFrontends,
        whisper=WhisperFrontend,
    ),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(
        specaug=SpecAug,
    ),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    name="normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)
visual_frontend_choices = ClassChoices(
    name="visual_frontend",
    classes=dict(
        conv3dresnet18=Conv3dResNet18,
    ),
    type_check=AbsFrontend,
    default="conv3dresnet18",
)
acoustic_preencoder_choices = ClassChoices(
    name="acoustic_preencoder",
    classes=dict(
        sinc=LightweightSincConvs,
        linear=LinearProjection,
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)
visual_preencoder_choices = ClassChoices(
    name="visual_preencoder",
    classes=dict(
        sinc=LightweightSincConvs,
        linear=LinearProjection,
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)
acoustic_embed_choices = ClassChoices(
    name="acoustic_embed",
    classes=dict(
        default=DefaultEmbeddingLayerForAVSR,
    ),
    type_check=EmbeddingForAVSRAbsLayer,
    default="default",
)
visual_embed_choices = ClassChoices(
    name="visual_embed",
    classes=dict(
        default=DefaultEmbeddingLayerForAVSR,
    ),
    type_check=EmbeddingForAVSRAbsLayer,
    default="default",
)
encoder_choices = ClassChoices(
    name="encoder",
    classes=dict(
        tailored=TailoredEncoder,
        conventional=ConventionalEncoder,
    ),
    type_check=AudioVisualAbsEncoder,
    default="tailored",
)
audiovisual_fusion_choices = ClassChoices(
    name="audiovisual_fusion",
    classes=dict(
        adaptive=AdaptiveAudioVisualFusion,
    ),
    type_check=AudioVisualFusionAbsModule,
    default="adaptive",
)
postencoder_choices = ClassChoices(
    name="postencoder",
    classes=dict(
        hugging_face_transformers=HuggingFaceTransformersPostEncoder,
    ),
    type_check=AbsPostEncoder,
    default=None,
    optional=True,
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
        transducer=TransducerDecoder,
        mlm=MLMDecoder,
        whisper=OpenAIWhisperDecoder,
        hugging_face_transformers=HuggingFaceTransformersDecoder,
        s4=S4Decoder,
    ),
    type_check=AbsDecoder,
    default=None,
    optional=True,
)
preprocessor_choices = ClassChoices(
    "preprocessor",
    classes=dict(
        default=CommonPreprocessor,
        multi=CommonPreprocessor_multi,
    ),
    type_check=AbsPreprocessor,
    default="default",
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        espnet=ESPnetAVSRModel,
        maskctc=AVSRMaskCTCModel,
    ),
    type_check=ESPnetAVSRModel,
    default="espnet",
)

class AVSRTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --acoustic_frontend and --acoustic_frontend_conf
        acoustic_frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --visual_frontend and --visual_frontend_conf
        visual_frontend_choices,
        # --acoustic_preencoder and --acoustic_preencoder_conf
        acoustic_preencoder_choices,
        # --visual_preencoder and --visual_preencoder_conf
        visual_preencoder_choices,
        # --acoustic_embed and --acoustic_embed_conf
        acoustic_embed_choices,
        # --visual_embed and --visual_embed_conf
        visual_embed_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --audiovisual_fusion and --audiovisual_fusion_conf
        audiovisual_fusion_choices,
        # --postencoder and --postencoder_conf
        postencoder_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --preprocessor and --preprocessor_conf
        preprocessor_choices,
        # --model and --model_conf
        model_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for CTC class.",
        )
        group.add_argument(
            "--joint_net_conf",
            action=NestedDictAction,
            default=None,
            help="The keyword arguments for joint network class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=[
                "bpe",
                "char",
                "word",
                "phn",
                "hugging_face",
                "whisper_en",
                "whisper_multilingual",
            ],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        group.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[
                None,
                "tacotron",
                "jaconv",
                "vietnamese",
                "whisper_en",
                "whisper_basic",
            ],
            default=None,
            help="Apply text cleaning",
        )
        group.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        group.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        group.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        group.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        group.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        group.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        group.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )
        group.add_argument(
            "--short_noise_thres",
            type=float,
            default=0.5,
            help="If len(noise) / len(speech) is smaller than this threshold during "
            "dynamic mixing, a warning will be displayed.",
        )
        group.add_argument(
            "--aux_ctc_tasks",
            type=str,
            nargs="+",
            default=[],
            help="Auxillary tasks to train on using CTC loss. ",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            try:
                _ = getattr(args, "preprocessor")
            except AttributeError:
                setattr(args, "preprocessor", "default")
                setattr(args, "preprocessor_conf", dict())
            except Exception as e:
                raise e

            preprocessor_class = preprocessor_choices.get_class(args.preprocessor)
            retval = preprocessor_class(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                # NOTE(kamo): Check attribute existence for backward compatibility
                rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                rir_apply_prob=args.rir_apply_prob
                if hasattr(args, "rir_apply_prob")
                else 1.0,
                noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                noise_apply_prob=args.noise_apply_prob
                if hasattr(args, "noise_apply_prob")
                else 1.0,
                noise_db_range=args.noise_db_range
                if hasattr(args, "noise_db_range")
                else "13_15",
                short_noise_thres=args.short_noise_thres
                if hasattr(args, "short_noise_thres")
                else 0.5,
                speech_volume_normalize=args.speech_volume_normalize
                if hasattr(args, "rir_scp")
                else None,
                aux_task_names=args.aux_ctc_tasks
                if hasattr(args, "aux_ctc_tasks")
                else None,
                **args.preprocessor_conf,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", "text")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        MAX_REFERENCE_NUM = 4

        retval = ["text_spk{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval = tuple(retval)

        logging.info(f"Optional Data Names: {retval }")
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetAVSRModel:
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")

        # If use multi-blank transducer criterion,
        # big blank symbols are added just before the standard blank
        if args.model_conf.get("transducer_multi_blank_durations", None) is not None:
            sym_blank = args.model_conf.get("sym_blank", "<blank>")
            blank_idx = token_list.index(sym_blank)
            for dur in args.model_conf.get("transducer_multi_blank_durations"):
                if f"<blank{dur}>" not in token_list:  # avoid this during inference
                    token_list.insert(blank_idx, f"<blank{dur}>")
            args.token_list = token_list

        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 1. Acoustic frontend
        if args.acoustic_input_size is None:
            # Extract acoustic features in the model
            acoustic_frontend_class = acoustic_frontend_choices.get_class(args.acoustic_frontend)
            acoustic_frontend = acoustic_frontend_class(**args.acoustic_frontend_conf)
            acoustic_input_size = acoustic_frontend.output_size()
        else:
            # Give acoustic features from data-loader
            args.acoustic_frontend = None
            args.acoustic_frontend_conf = {}
            acoustic_frontend = None
            acoustic_input_size = args.acoustic_input_size

        # 2. Visual frontend
        if args.visual_input_size is None:
            # Extract visual features in the model
            visual_frontend_class = visual_frontend_choices.get_class(args.visual_frontend)
            visual_frontend = visual_frontend_class(**args.visual_frontend_conf)
            visual_input_size = visual_frontend.output_size()
        else:
            # Give visual features from data-loader
            args.visual_frontend = None
            args.visual_frontend_conf = {}
            visual_frontend = None
            visual_input_size = args.visual_input_size

        # 2. Acoustic data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Acoustic normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 4. Acoustic pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "acoustic_preencoder", None) is not None:
            acoustic_preencoder_class = acoustic_preencoder_choices.get_class(args.acoustic_preencoder)
            acoustic_preencoder = acoustic_preencoder_class(**args.acoustic_preencoder_conf)
            acoustic_input_size = acoustic_preencoder.output_size()
        else:
            acoustic_preencoder = None

        # 4. Visual pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "visual_preencoder", None) is not None:
            visual_preencoder_class = visual_preencoder_choices.get_class(args.visual_preencoder)
            visual_preencoder = visual_preencoder_class(**args.visual_preencoder_conf)
            visual_input_size = visual_preencoder.output_size()
        else:
            visual_preencoder = None

        # 5. Embedding layers
        acoustic_embed_output_size = args.encoder_conf["output_size"]
        visual_embed_output_size = args.encoder_conf["output_size"]

        # 5.1. Acoustic embedding layer
        acoustic_embed_class = acoustic_embed_choices.get_class(args.acoustic_embed)
        acoustic_embed = acoustic_embed_class(
             input_size=acoustic_input_size,
             output_size=acoustic_embed_output_size,
             **args.acoustic_embed_conf,
        )
        acoustic_input_size = acoustic_embed.output_size()

        # 5.2. Visual embedding layer
        visual_embed_class = visual_embed_choices.get_class(args.visual_embed)
        visual_embed = visual_embed_class(
             input_size=visual_input_size,
             output_size=visual_embed_output_size,
             **args.visual_embed_conf,
        )
        visual_input_size = visual_embed.output_size()

        # 5.3. Embedding sanity checks
        assert acoustic_embed._rel_pos_type == visual_embed._rel_pos_type
        assert acoustic_embed._pos_enc_layer_type == visual_embed._pos_enc_layer_type
        assert acoustic_input_size == visual_input_size

        # 6. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        if encoder_class == TailoredEncoder:
            encoder = encoder_class(
                embed_pos_enc_layer_type=acoustic_embed._pos_enc_layer_type,
                embed_rel_pos_type=acoustic_embed._rel_pos_type,
                **args.encoder_conf,
            )
        elif encoder_class == ConventionalEncoder:
            encoder = encoder_class(
                input_size=acoustic_input_size,
                embed_pos_enc_layer_type=acoustic_embed._pos_enc_layer_type,
                embed_rel_pos_type=acoustic_embed._rel_pos_type,
                **args.encoder_conf,
            )

        encoder_output_size = encoder.output_size()

        # 7. Audio-Visual Fusion
        audiovisual_fusion_class = audiovisual_fusion_choices.get_class(args.audiovisual_fusion)
        audiovisual_fusion = audiovisual_fusion_class(
            input_size=encoder_output_size,
            **args.audiovisual_fusion_conf,
        )
        encoder_output_size = audiovisual_fusion.output_size()

        # 8. Post-encoder block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "postencoder", None) is not None:
            postencoder_class = postencoder_choices.get_class(args.postencoder)
            postencoder = postencoder_class(
                input_size=encoder_output_size, **args.postencoder_conf
            )
            encoder_output_size = postencoder.output_size()
        else:
            postencoder = None

        # 9. Decoder
        if getattr(args, "decoder", None) is not None:
            decoder_class = decoder_choices.get_class(args.decoder)

            if args.decoder == "transducer":
                decoder = decoder_class(
                    vocab_size,
                    embed_pad=0,
                    **args.decoder_conf,
                )

                joint_network = JointNetwork(
                    vocab_size,
                    encoder.output_size(),
                    decoder.dunits,
                    **args.joint_net_conf,
                )
            else:
                decoder = decoder_class(
                    vocab_size=vocab_size,
                    encoder_output_size=encoder_output_size,
                    **args.decoder_conf,
                )
                joint_network = None
        else:
            decoder = None
            joint_network = None

        # 10. CTC
        ctc = CTC(
            odim=vocab_size, encoder_output_size=encoder_output_size, **args.ctc_conf
        )

        # 11. Build model
        try:
            model_class = model_choices.get_class(args.model)
        except AttributeError:
            model_class = model_choices.get_class("espnet")
        model = model_class(
            vocab_size=vocab_size,
            token_list=token_list,
            specaug=specaug,
            normalize=normalize,
            acoustic_frontend=acoustic_frontend,
            visual_frontend=visual_frontend,
            acoustic_preencoder=acoustic_preencoder,
            visual_preencoder=visual_preencoder,
            acoustic_embed=acoustic_embed,
            visual_embed=visual_embed,
            encoder=encoder,
            audiovisual_fusion=audiovisual_fusion,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 10. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
