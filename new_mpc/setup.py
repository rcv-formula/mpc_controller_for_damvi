from setuptools import setup
import glob

package_name = 'mpc_controller'

map_files = glob.glob('map/*.csv')

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/config.yaml']),
        ('share/' + package_name + '/map', map_files),
    ],
    install_requires=['setuptools', 'numpy', 'cvxpy', 'pyyaml', 'tf-transformations'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='MPC-based local path planning and control for an RC car.',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpc_node = mpc_controller.mpc_node:main',
            'new_mpc_node = mpc_controller.new_mpc_node:main',
            'compare = mpc_controller.compare:main',
        ],
    },
)


