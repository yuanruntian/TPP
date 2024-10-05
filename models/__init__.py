from .tpp import build

def build_model(args):
    if args.online:
        return build(args)
