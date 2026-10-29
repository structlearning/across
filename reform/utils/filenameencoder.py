class EncDecBase:
    """
    Base class for encoding logfile names and validation model names to shorter strings,
    or for decoding them back to the original names.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_map = {
            "function=cos,stemp=0.1,margin=1.0": "fc,s0.1m1",
            "100000": "e5",
            "everydim": "ed",
            "lossce-avg": "Lav",
            "aug_emb": "ae",
            "mlp_sym": "ms",
            "linear": "lin",
            "outdim64": "o64",
            "collMargin1.0": "cl1",
            "tTemp1.0": "",
            "active": "ac",
            "patience": "p",
            "freeze": "fz",
            "True": "tt",
        }

        self.reverse_map = {v: k for k, v in self.forward_map.items()}


class ShortEncoder(EncDecBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, name):
        for k, v in self.forward_map.items():
            name = name.replace(k, v)
        return name


class ShortDecoder(EncDecBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def decode(self, name):
        for k, v in self.reverse_map.items():
            name = name.replace(k, v)
        return name
