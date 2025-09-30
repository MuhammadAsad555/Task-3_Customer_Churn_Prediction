
def ensure_dirs():
    import os
    for d in ('models', 'outputs'):
        os.makedirs(d, exist_ok=True)
