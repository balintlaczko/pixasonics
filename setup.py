from setuptools import setup, find_packages

setup(
    name='pixasonics',
    version='0.1.3',
    author='Balint Laczko',
    author_email='balint.laczko@imv.uio.no',
    description='An Image Sonification Toolbox',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'pixasonics': [
            'figures/*', 'images/*', 'pixasonics_tutorial.ipynb'
        ]
    },
    entry_points={
        'console_scripts': [
            'pixasonics-notebook = pixasonics.launch:launch_notebook',
        ]
    },
    install_requires=[
        "numpy",
        "pillow",
        "signalflow",
        # "taichi",
        "ipython",
        "jupyter",
        "ipycanvas",
        "ipywidgets",

    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.10',
    ],
)