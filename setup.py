from setuptools import setup

setup(
    name='TensorflowObjectDetection',
    version='1.0',
    packages=[
    ],
    url='',
    license='beer',
    author='patrickryan',
    author_email='pat_ryan_99@yahoo.com',
    description='',
    install_requires = [
        'numpy==1.19.5',
        'pycocotools==2.0.1'
        'tensorflow==2.4.1',
        'opencv-python==4.5.1.48',
        'opencv-contrib-python',
        'pandas==1.2.1',
        'scikit-learn==0.24.1'
    ]
)

