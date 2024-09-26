from .referformer import build
from .onlinerefer import build as build_online

def build_model(args):
    if args.online:
        return build_online(args)
    else:
        return build(args)
