import cv2
import warnings
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, logging
import torch
from transformers.modeling_outputs import Seq2SeqLMOutput
import traceback
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
model_path = "D:/Neb_Ocr_Final/trocr-base-printed"
# model_path = 'microsoft/trocr-base-printed'
preprocessor = TrOCRProcessor.from_pretrained(model_path, use_fast=False, local_files_only=True)
warnings.filterwarnings("ignore")
# logging.set_verbosity_warning()
# logging.set_verbosity_info()
logging.set_verbosity_error()
logging.disable_progress_bar() 

class BetterHFTrOCR(VisionEncoderDecoderModel):
    """creates a TrOCR model"""

    def __init__(self, model_path):
        model_ = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)

        super().__init__(model_.config)

        self.encoder = model_.encoder
        self.decoder = model_.decoder

    def forward(
        self,
        pixel_values=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if hasattr(encoder_outputs, "last_hidden_state"):
            encoder_hidden_states = encoder_outputs.last_hidden_state
        elif isinstance(encoder_outputs, dict):
            encoder_hidden_states = encoder_outputs["last_hidden_state"]
        elif isinstance(encoder_outputs, (list, tuple)):
            encoder_hidden_states = encoder_outputs[0]
        else:
            raise TypeError(f"Unexpected encoder_outputs type: {type(encoder_outputs)}")

        encoder_attention_mask = None
        
        eos_mask = decoder_input_ids[:, -1] <= self.config.eos_token_id

        # Decode        
        if any(eos_mask) and (decoder_input_ids.shape[1] > 1):
            reduced_logits = self.decoder(
                input_ids=decoder_input_ids[torch.logical_not(eos_mask), :],
                attention_mask=(
                    decoder_attention_mask[torch.logical_not(eos_mask), :]
                    if decoder_attention_mask is not None else None
                ),
                encoder_hidden_states=encoder_hidden_states[torch.logical_not(eos_mask), :, :],
                encoder_attention_mask=encoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                past_key_values=past_key_values,
            ).logits
            
            logits = torch.full((decoder_input_ids.shape[0], decoder_input_ids.shape[1], self.config.decoder.vocab_size), fill_value=self.config.pad_token_id, dtype=reduced_logits.dtype, device=reduced_logits.device)
            logits[torch.logical_not(eos_mask), :, :] = reduced_logits
            logits[eos_mask, :, :] = self.ids_to_logits(decoder_input_ids[eos_mask, 1:], reduced_logits)
        else:
            logits = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                past_key_values=past_key_values,
            ).logits

        return Seq2SeqLMOutput(
          logits=logits,
        )

    def ids_to_logits(self, ids, reduced_logits):
        logits = torch.zeros((ids.shape[0], ids.shape[1]+1, self.config.decoder.vocab_size), dtype=reduced_logits.dtype, device=reduced_logits.device)
        logits[:, -1, 2] = 1 # max_pad_token
        for i in range(ids.shape[1]):
            logits[:, i, ids[:, i]] = 1
        
        return logits

def configure_generation(model, beams=1):
    model.config.pad_token_id = model.config.decoder.pad_token_id = preprocessor.tokenizer.pad_token_id
    model.config.eos_token_id = model.config.decoder.eos_token_id = preprocessor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder.early_stopping = True
    model.config.decoder.no_repeat_ngram_size = 3
    model.config.decoder.length_penalty = 2.0
    model.config.decoder.num_beams = beams


better_cuda_hf_trocr = BetterHFTrOCR(
    model_path=model_path, 
    ).to('cuda')

configure_generation(better_cuda_hf_trocr)


def extract_data(image, boxes_selected):
    if image is None:
        print("Image not found:")
        return

    try:
        textlines = []
        coordinates_lines = [] 
        # output_folder = "output_steps/v1"
        # os.makedirs(output_folder, exist_ok=True)

        # Process only selected boxes
        for idx, (x1, y1, x2, y2) in enumerate(boxes_selected):
            roi = image[y1:y2, x1:x2]
            
            pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).convert("RGB")
            # pil_img.save(os.path.join(output_folder, f"roi_{idx+1}.jpg"))
            textlines.append(pil_img)
            coordinates_lines.append((x1, y1, x2, y2))

        # Convert to pixel_values for OCR
        pixel_values = preprocessor(textlines, return_tensors="pt").pixel_values

        # Chunking if too many
        if pixel_values.shape[0] > 100:
            max_batch_size = 100
            ocr_results = []
            with torch.no_grad():
                for i in range(0, len(pixel_values), max_batch_size):
                    batch = pixel_values[i : i + max_batch_size].to('cuda')
                    generated_ids_chunk = better_cuda_hf_trocr.generate(batch)
                    decoded = preprocessor.tokenizer.batch_decode(generated_ids_chunk, skip_special_tokens=True)
                    ocr_results.extend(decoded)
                    torch.cuda.empty_cache() 
        else:
            with torch.no_grad():
                generated_ids = better_cuda_hf_trocr.generate(pixel_values.to('cuda'))
                ocr_results = preprocessor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # result_new = list(zip(coordinates_lines, ocr_results))
        # print(ocr_results)
        return ocr_results
 
    except Exception as exception:
            print(str(exception))
