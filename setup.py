from setuptools import setup

setup(
    name = "quad_simulator",
    version = "0.0.0",
    description = "Quadrotor Simulator",
    author = "hu kai chun",
    author_email = "hu.kaichun@gmail.com",
    license = "MIT",
    packages = ["quad.simulator", "quad.simulator.core", "quad.simulator.skeleton",
                "quad.utils"],
    install_requires=["numpy", "matplotlib"],
    zip_safe = False
)
