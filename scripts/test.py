from . import skipthoughts
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
vectors = encoder.encode(["I am a banana","I dare you"])


