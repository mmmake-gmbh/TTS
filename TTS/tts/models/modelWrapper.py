import mlflow


class MyModel(mlflow.pyfunc.PythonModel):

    def __init__(self, synthesizer):
        self.synthesizer = synthesizer

    def predict(self, context, model_input):
        synthesizer = self.synthesizer
        wav = synthesizer.tts(model_input)
        synthesizer.save_wav(wav, 'output.wav')

        return wav
