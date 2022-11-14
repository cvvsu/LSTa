def get_network(args):
    """
    Obtain network according to name
    """
    module = __import__(f'models.networks.{args.network_name}', fromlist=['build_network'])
    func = getattr(module, 'build_network')
    return func(args)