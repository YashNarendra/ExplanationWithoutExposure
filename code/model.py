import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    SiglipVisionModel,
    BitsAndBytesConfig
)

class ExplanationVLM(nn.Module):
    def __init__(self, slm_name="microsoft/phi-2", vision_name="google/siglip-so400m-patch14-384"):
        super().__init__()
        
        # 1. Vision Encoder (Frozen, High-Res 384px)
        # We use the SigLIP model as described in Section 5.1
        print(f"Loading Vision Encoder: {vision_name}...")
        self.vision_encoder = SiglipVisionModel.from_pretrained(vision_name)
        self.vision_processor = AutoProcessor.from_pretrained(vision_name)
        
        # Freeze vision encoder to save memory (Section 5.3)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # 2. Small Language Model (4-bit Quantized)
        print(f"Loading SLM: {slm_name}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.slm = AutoModelForCausalLM.from_pretrained(
            slm_name,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        self.tokenizer = AutoProcessor.from_pretrained(slm_name, trust_remote_code=True).tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 3. Projection Network (Eq. 3 in paper)
        # Maps vision embedding dim (1152) to SLM embedding dim (2560 for Phi-2)
        vision_dim = self.vision_encoder.config.hidden_size
        slm_dim = self.slm.config.hidden_size
        
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, slm_dim),
            nn.GELU(),
            nn.Linear(slm_dim, slm_dim)
        )
        
        # Move projection to the same device/dtype as the SLM's expected input
        self.projection.to(torch.float16)

    def get_visual_embs(self, pixel_values):
        """Pass image through SigLIP and Projector."""
        with torch.no_grad():
            # Get patch embeddings (last_hidden_state)
            # Shape: [Batch, Num_Patches, Vision_Dim]
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            image_embeds = vision_outputs.last_hidden_state
        
        # Project to SLM space
        # Shape: [Batch, Num_Patches, SLM_Dim]
        projected_embeds = self.projection(image_embeds.to(torch.float16))
        return projected_embeds

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        # 1. Get Visual Soft Prompts
        visual_embeds = self.get_visual_embs(pixel_values)
        
        # 2. Get Text Embeddings
        text_embeds = self.slm.model.embed_tokens(input_ids)
        
        # 3. Multimodal Fusion (Concat [Image; Text])
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        # 4. Create Attention Mask for the concatenated sequence
        # We assume full attention for the image part (1s)
        batch_size = inputs_embeds.shape[0]
        vis_seq_len = visual_embeds.shape[1]
        
        if attention_mask is not None:
            vis_mask = torch.ones((batch_size, vis_seq_len), device=inputs_embeds.device)
            attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
            
        # 5. Handle Labels (Masking Visual Tokens)
        # As per Section 5.3, we exclude visual tokens from the loss
        if labels is not None:
            vis_labels = torch.full((batch_size, vis_seq_len), -100, device=inputs_embeds.device)
            labels = torch.cat([vis_labels, labels], dim=1)

        # 6. Pass to SLM
        outputs = self.slm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def generate(self, image, prompt, max_new_tokens=100):
        """Inference function."""
        # Process inputs
        pixel_values = self.vision_processor(images=image, return_tensors="pt").pixel_values.to("cuda")
        text_inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Get embeddings
        visual_embeds = self.get_visual_embs(pixel_values)
        text_embeds = self.slm.model.embed_tokens(text_inputs.input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        # Generate
        outputs = self.slm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode only the new tokens (stripping the prompt is implicit in decoder-only if handled carefully, 
        # but here we just decode the output ids. Note: generate() with inputs_embeds returns tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)