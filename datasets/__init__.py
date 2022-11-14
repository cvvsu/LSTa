def get_dataset(args):
    """
    Get the data loaders for training, validation, and test sets.
    """
    # task = args.name.split('_')[0]
    module = __import__(f'datasets.{args.model_name}_loader', fromlist=['build_dataset'])
    func = getattr(module, 'build_dataset')
    return func(args)