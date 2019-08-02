from setuptools import setup

setup(
    name = "quad",
    version = "0.0.1",
    description = "Quadrotor Simulator",
    author = "hu kai chun",
    author_email = "hu.kaichun@gmail.com",
    license = "",
    packages = ["quad",
                "quad.simulator", 
                "quad.simulator.core", 
                "quad.simulator.skeleton",
                "quad.utils",
                "quad.gym"],
    install_requires=["numpy", "matplotlib"],
    zip_safe = False
)
