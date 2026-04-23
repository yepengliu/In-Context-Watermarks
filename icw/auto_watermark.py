import importlib

WATERMARK_MAPPING_NAMES={
    'UNICODE': 'icw.unicode.UnicodeICW',
    'INITIALS': 'icw.initials.InitialsICW',
    'LEXICAL': 'icw.lexical.LexicalICW',
    'ACROSTICS': 'icw.acrostics.AcrosticsICW',
}


def watermark_name_from_alg_name(name):
    """Get the watermark class name from the algorithm name."""
    for algorithm_name, watermark_name in WATERMARK_MAPPING_NAMES.items():
        if name == algorithm_name:
            return watermark_name
    return None


class AutoWatermark:

    def __init__(self):
        raise EnvironmentError(
            "AutoWatermark is designed to be instantiated "
            "using the `AutoWatermark.load(algorithm_name, algorithm_config, transformers_config)` method."
        )

    @staticmethod
    def load(config):
        """Load the watermark algorithm instance based on the algorithm name."""
        watermark_name = watermark_name_from_alg_name(config['icw'])
        module_name, class_name = watermark_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        watermark_class = getattr(module, class_name)
        watermark_instance = watermark_class(config)
        return watermark_instance
