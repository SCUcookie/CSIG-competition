class DeepProUnavailable(RuntimeError): pass
class DeepProDetector:
    def __init__(self, source_root, weights, **kwargs):
        raise DeepProUnavailable("DeepPro is an explicit optional adapter; provide the pinned source and weights before use")
