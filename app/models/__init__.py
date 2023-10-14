from app.models.pure_transformer import PureT

__factory = {
    'PureT': PureT,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption stylize_model:", name)
    return __factory[name](*args, **kwargs)
