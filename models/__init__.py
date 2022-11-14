def get_model(args):
    """
    Obtain model according to name
    """
    module = __import__(f'models.{args.model_name}_model', fromlist=['build_model'])
    func = getattr(module, 'build_model')
    return func(args)