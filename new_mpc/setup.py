from setuptools import setup
import glob

package_name = 'mpc_controller'

map_files = glob.glob('map/*.csv')

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, f'{package_name}.modules'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/config.yaml']),
        ('share/' + package_name + '/map', map_files),
    ],
    install_requires=['setuptools', 'numpy', 'pyyaml', 'tf-transformations', 'casadi'],
    zip_safe=True,
    maintainer='Jun Lim',
    maintainer_email='gogownsud2@gmail.com',
    description='MPC-based local path planning and control for an RC car.',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kinematic_nonlinear = mpc_controller.kinematic_nonlinear:main',
        ],
    },
)


