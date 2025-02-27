from setuptools import setup, find_packages

# Read the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='pixasonics',
    version='0.1.3',
    author='Balint Laczko',
    author_email='balint.laczko@imv.uio.no',
    description='An Image Sonification Toolbox',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/balintlaczko/pixasonics",
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
        "numba",
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