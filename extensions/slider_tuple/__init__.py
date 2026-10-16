# Image-pair / image-tuple slider training (see NOTES.md and the Krea 2 slider handoff).
from toolkit.extension import Extension


class SliderTupleTrainerExtension(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "slider_tuple_trainer"

    # name is the name of the extension for printing
    name = "Slider Tuple Trainer"

    # keep the import in here so it is only loaded when needed
    @classmethod
    def get_process(cls):
        from .SliderTupleTrainer import SliderTupleTrainer

        return SliderTupleTrainer


AI_TOOLKIT_EXTENSIONS = [
    SliderTupleTrainerExtension,
]
