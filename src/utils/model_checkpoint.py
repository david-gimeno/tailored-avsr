import os
import torch
import pandas as pd
from collections import OrderedDict

def load_frontend_lrw(e2e, checkpoint, module_name):
    frontend_lrw = OrderedDict()
    for key in checkpoint.keys():
        if "tcn_trunk" not in key:
            if ("trunk" in key) or ("frontend3D" in key):
                frontend_lrw[key] = checkpoint[key]

    if module_name == "frontend":
        e2e.frontend.load_state_dict(frontend_lrw)
    elif module_name == "visual_frontend":
        e2e.visual_frontend.load_state_dict(frontend_lrw)

def load_module(e2e, module, checkpoint, ctc_weight):
    # -- creating the module's checkpoint
    module_checkpoint = OrderedDict()
    for key in checkpoint.keys():
        if module+"." in key:
            new_key = key.replace(module+".", "")
            module_checkpoint[new_key] = checkpoint[key]

    # -- loading the chosen module
    if module == "frontend":
        e2e.frontend.load_state_dict(module_checkpoint)

    if module == "encoder":
        e2e.encoder.load_state_dict(module_checkpoint)

    if module == "decoder":
        if ctc_weight < 1.0:
            e2e.decoder.load_state_dict(module_checkpoint)
        else:
            raise RuntimeError("The end-to-end model does not have an Attention-based decoding branch!")

    if module == "ctc":
        if ctc_weight > 0.0:
            e2e.ctc.load_state_dict(module_checkpoint)
        else:
            raise RuntimeError("The end-to-end model does not have a CTC-based decoding branch!")

def load_e2e(e2e, modules, checkpoint_path, ctc_weight):
    if checkpoint_path != "":
        checkpoint = torch.load(checkpoint_path)
        if "entire-e2e" not in modules:
            for module in modules:
                if ("LRW" in checkpoint_path):
                    assert module in ["frontend", "visual_frontend"], "When loading from the LRW model, it is only possible loading the frontend."
                    print(f"Loading pre-trained visual frontend from {checkpoint_path}")
                    load_frontend_lrw(e2e, checkpoint, module)
                else:
                    print(f"Loading pre-trained {module} from {checkpoint_path}.")
                    load_module(e2e, module, checkpoint, ctc_weight)
        else:
            print(f"Loading the entire E2E system from {checkpoint_path}")
            e2e.load_state_dict(checkpoint, strict=False)
            # if (ctc_weight > 0.0) and (ctc_weight < 1.0):
            #      e2e.load_state_dict(checkpoint)
            # else: # If there is no Attention- or CTC-based branch
            #      e2e.load_state_dict(checkpoint, strict=False)
            #      print(f"The end-to-end model was pre-train but there is a mismatch. It is probably missing the Attention- or the CTC-based decoding branch.")
    else:
        print(f"Training the end-to-end model from scracth!")

def average_model(e2e, checkpoint_paths):
    """
      Code based on the implentation publicly released by FairSeq.
        https://github.com/facebookresearch/fairseq/blob/main/scripts/average_checkpoints.py
    """
    average_model = {}
    for checkpoint_path in checkpoint_paths:
        model = torch.load(checkpoint_path)

        model_keys = list(model.keys())
        for k in model_keys:
            p = model[k]
            if average_model.get(k) is None:
                average_model[k] = p.clone()
            else:
                average_model[k] += p.clone()

    nmodels = len(checkpoint_paths)
    for k, v in average_model.items():
        average_model[k] = torch.div(v, nmodels)

    e2e.load_state_dict(average_model)

def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()

def freeze_e2e(e2e, modules, mtlalpha):
    if "no-frozen" not in modules:
        for module in modules:
            if module == "frontend":
                for param in e2e.frontend.parameters():
                    param.requires_grad = False
                print("The Frontend is frozen!!")
            elif module == "encoder":
                for param in e2e.encoder.parameters():
                    param.requires_grad = False
                print("The Encoder is frozen!!")
            elif module == "decoder":
                if mtlalpha < 1.0:
                    for param in e2e.decoder.parameters():
                        param.requires_grad = False
                    print("The Attention-based Decoder is frozen!!")
                else:
                    raise RuntimeError("The end-to-end model does not have a Attention-based decoding branch!")
            elif module == "ctc":
                if mtlalpha > 0.0:
                    for param in e2e.ctc.parameters():
                        param.requieres_grad = False
                    print("The CTC-based Decoder is frozen!!")
                else:
                    raise RuntimeError("The end-to-end model does not have a CTC-based decoding branch!")
    else:
        print("The entire E2E system will be trained")

def save_model(output_dir, model, suffix):
    dst_root = output_dir + "/models/"

    os.makedirs(dst_root, exist_ok=True)
    dst_path = os.path.join(dst_root, "model_" + suffix + ".pth")
    print(f"Saving model in {dst_path} ...")
    torch.save(model.state_dict(), dst_path)

    return dst_path

def save_val_stats(output_dir, val_stats):
    dst_path = os.path.join(output_dir, "val_stats.csv")
    df = pd.DataFrame(val_stats, columns=["model_check_path", "cer"])
    df.to_csv(dst_path)
