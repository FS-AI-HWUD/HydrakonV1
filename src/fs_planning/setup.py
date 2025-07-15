from setuptools import find_packages, setup

package_name = 'fs_planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aditya S',
    maintainer_email='as2397@hw.ac.uk',
    description='Path planning package for Hydrakon',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nmea_bridge = fs_planning.nmea_bridge:main',
            'hydrakon_planning = fs_planning.hydrakon_planning:main',
        ],
    },
)