from setuptools import setup, find_packages

setup(
    name='RV_Autoaim_2026',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'gxipy',    
        'opencv-python',
        'flask',
        'numpy',
        'Pillow',
        'filterpy'
    ],
    python_requires='>=3.8',
    author='AMwhisper',
    description='Galaxy Camera + YOLOv8 + solvePnP + Web Stream',
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "autoaim_start=RV_Autoaim_2026.start:main",
        ]
    },
)
