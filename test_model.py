from models.load_model import load_model


tokenizer, model = load_model()

print("\nMODEL LOADED SUCCESSFULLY\n")

print(model.device)