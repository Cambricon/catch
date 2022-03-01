def is_available():
    """
    Keep for backward compatibility

    """
    return True


if is_available():
    from .mlu_ddp import *
