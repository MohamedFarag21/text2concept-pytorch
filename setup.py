from setuptools import setup, find_packages

exec(open('text2concept_pytorch/version.py').read())

setup(
    name                          = 'text2concept-pytorch',
    packages                      = find_packages(),
    version                       = __version__,
    license                       = 'MIT',
    description                   = 'Text2Concept - Concept Activation Vectors Directly from Text, in PyTorch',
    author                        = 'Mohamed Ibrahim',
    author_email                  = 'mibrahi2@uni-bonn.de',
    url                           = 'https://github.com/MohamedFarag21/text2concept-pytorch',
    long_description_content_type = 'text/markdown',
    keywords = [
        'artificial intelligence',
        'deep learning',
        'interpretability',
        'explainability',
        'concept activation vectors',
        'CLIP',
        'zero-shot classification',
    ],
    install_requires = [
        'accelerate',
        'einops>=0.7',
        'open-clip-torch>=2.20',
        'Pillow',
        'torch>=2.0',
        'torchvision',
        'tqdm',
    ],
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
