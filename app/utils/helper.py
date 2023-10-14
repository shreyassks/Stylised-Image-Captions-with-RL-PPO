import io
import os
import sys

import cv2
import numpy as np
import requests
import skimage.transform
import torch
from PIL import Image
from torchvision import transforms
from transformers import T5ForConditionalGeneration, T5Tokenizer

from app import models
from app.core.config import cfg, cfg_from_file
from app.core.log import logger

folder = "app/artifacts"
cfg_from_file(os.path.join(folder, 'config.yml'))
cfg.ROOT_DIR = folder
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
print(f"Device is set to : {device}")


def load_model(path):
    """ Creates Swin Transformer Architecture from definition and loads trained weights from the file"""
    try:
        model = models.create("PureT")
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(path, map_location=device))

        style_model = T5ForConditionalGeneration.from_pretrained("app/artifacts/stylized_model").to(device)
        return model.eval(), style_model
    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        filename = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, filename, exc_tb.tb_lineno)


def load_vocab(path):
    vocab = ['.']
    with open(path, 'r') as fid:
        for line in fid:
            vocab.append(line.strip())
    return vocab


model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512)
caption_model, stylized_model = load_model(cfg.INFERENCE.MODEL_PATH)
caption_vocab = load_vocab(cfg.INFERENCE.VOCAB)
print("Loaded Models and Vocabulary")


def process_image(image):
    """
    Read image and apply tensor transformations
    """
    try:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        logger.info(f"Image Shape : {img.shape}")

        img = skimage.transform.resize(img, (384, 384))
        logger.info(f"Image Shape after Resize : {img.shape}")
        img = img.transpose(2, 0, 1)  # (3, 384, 384)

        img = torch.FloatTensor(img)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([normalize])
        image = transform(img)
        logger.info(f"Image Shape after Tensor Transform : {image.shape}")

        image = image.unsqueeze(0).to(device)  # (1, 3, 256, 256)
        return image

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        filename = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, filename, exc_tb.tb_lineno)


def decode_sequence(vocab, seq):
    N, T = seq.size()
    sents = []
    for n in range(N):
        words = []
        for t in range(T):
            ix = seq[n, t]
            if ix == 0:
                break
            words.append(vocab[ix])
        sent = ' '.join(words)
        sents.append(sent)
    return sents


def make_kwargs(att_feats, att_mask):
    kwargs = {cfg.PARAM.ATT_FEATS: att_feats, cfg.PARAM.ATT_FEATS_MASK: att_mask, 'BEAM_SIZE': cfg.INFERENCE.BEAM_SIZE}
    return kwargs


def stylize_text(input_text, tokenizer, model, num_return_sequences):
    batch = tokenizer(input_text, truncation=True, padding='max_length', max_length=40, return_tensors="pt").to(device)
    translated = model.generate(**batch,
                                max_length=25,
                                num_beams=10,
                                num_return_sequences=num_return_sequences,
                                temperature=1.8,
                                top_k=50,
                                top_p=0.95,
                                use_cache=True,
                                do_sample=True,
                                early_stopping=True)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


def inference(image_path):
    """
    Predicts the caption given an Image
    image_path: [channels, dim_x, dim_y] --> numpy array
    vocab: vocab_size -> dictionary
    """
    try:
        attention_feature = process_image(image_path)
        attention_mask = torch.ones(1, 12 * 12)

        kwargs = make_kwargs(att_feats=attention_feature.to(device),
                             att_mask=attention_mask.to(device))
        outs, _ = caption_model.module.decode_beam(**kwargs)
        generated_caption = decode_sequence(caption_vocab, outs.data)[0]
        stylized_captions = stylize_text([generated_caption], tokenizer, stylized_model, 5)

        return stylized_captions

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        filename = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, filename, exc_tb.tb_lineno)


def create_response(image: str, captions: str):
    """Creates a complete Response object from given parameters
        Args:
            image: Image URL
            captions: Generated Captions from the Model

        Returns:
            Response: a class object of Response
        """
    response = {"image_url": image, "generated_captions": captions}
    return response


def map_to_caption(image_dict):
    try:
        img_url = image_dict.get("image_url")
        logger.debug("Reading Image Bytes")
        r = requests.get(img_url, stream=True)
        image = Image.open(io.BytesIO(r.content)).convert("RGB")
        generated_captions = inference(image)
        logger.info(f"Generated Captions : {generated_captions}")
        output = create_response(img_url, generated_captions)
        return output
    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        filename = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, filename, exc_tb.tb_lineno)
