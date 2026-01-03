import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AdamW, get_linear_schedule_with_warmup
from model import ExplanationVLM
from dataset import ExplanationDataset

# --- Configuration ---
BATCH_SIZE = 2      # Adjusted for T4 GPU (Section 8 implementation details)
GRAD_ACCUM = 8
EPOCHS = 3
LEARNING_RATE = 2e-4

def main():
    # 1. Initialize Model
    vlm = ExplanationVLM()
    vlm.train() # Set to train mode
    
    # 2. Apply QLoRA (Section 5.3)
    # We only train the SLM via adapters and the Projection layer
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "fc1", "fc2"], # Targets for Phi-2
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Wrap SLM with LoRA
    vlm.slm = get_peft_model(vlm.slm, lora_config)
    vlm.slm.print_trainable_parameters()
    
    # Ensure Projector is trainable (it's not part of LoRA, so we set requires_grad manually)
    for param in vlm.projection.parameters():
        param.requires_grad = True
        
    # Move model components to GPU
    # Note: SLM is already on GPU via bitsandbytes load_in_4bit
    vlm.vision_encoder.to("cuda")
    vlm.projection.to("cuda")

    # 3. Setup Mock Data (Replace with your real data loading logic)
    # In a real run, load your JSON of image paths and teacher explanations here
    dummy_data = [
        {"image_path": "example_shap.png", "text": "This SHAP plot shows that Age contributes positively..."}
    ] * 50 # Duplicate for testing
    
    dataset = ExplanationDataset(
        dummy_data, 
        vlm.tokenizer, 
        vlm.vision_processor
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Optimizer and Scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, vlm.parameters()), lr=LEARNING_RATE)
    
    print("Starting training...")
    
    # 5. Training Loop
    step = 0
    for epoch in range(EPOCHS):
        for batch in dataloader:
            # Move batch to GPU
            pixel_values = batch["pixel_values"].to("cuda")
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")
            
            # Forward Pass
            outputs = vlm(pixel_values, input_ids, attention_mask, labels)
            loss = outputs.loss / GRAD_ACCUM
            
            # Backward
            loss.backward()
            
            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item() * GRAD_ACCUM:.4f}")
            
            step += 1

    # 6. Save Model
    print("Saving adapters and projector...")
    vlm.slm.save_pretrained("saved_models/slm_lora")
    torch.save(vlm.projection.state_dict(), "saved_models/projector.pt")

if __name__ == "__main__":
    main()