# Load weights
vlm.slm.load_adapter("saved_models/slm_lora", adapter_name="default")
vlm.projection.load_state_dict(torch.load("saved_models/projector.pt"))

# Run
img = Image.open("example_shap.png")
print(vlm.generate(img, "Explain this plot: "))