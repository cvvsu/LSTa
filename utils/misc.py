import os
import torch
import platform
from loguru import logger


def mkdirs(fp):
    os.makedirs(f'{fp}', exist_ok=True)


def get_sys_info():
    """
    os.uname only works for unix|linux systems.
    platform is available for both windows and linux.
    """

    # system
    names = ['System', 'Node', 'Release', 'Version', 'Machine', 'Processor']
    for name, value in zip(names, platform.uname()):
        logger.info(f'{name}: {value}')
    
    # architecture
    logger.info(f'Architecture: {platform.architecture()}')

    # python version and complier
    logger.info(f'Python version: {platform.python_version()}')
    logger.info(f'Python complier: {platform.python_compiler()}')

    # Pytorch version
    if torch.cuda.is_available():
        logger.info(f'PyTorch version: {torch.__version__}')

    # user
    user = os.environ.get('USER', os.environ.get('USERNAME'))
    logger.info(f'Current user: {user}')
    

