from setuptools import setup

setup(
    name = "quad",
    version = "0.0.1",
    description = "Quadrotor Simulator with tensorflow 2.0",
    author = "hu kai chun",
    author_email = "hu.kaichun@gmail.com",
    license = "",
    packages = ["quad",
                "quad.simulator",
                "quad.simulator.skeleton"],
    install_requires=["numpy", "matplotlib"],
    zip_safe = False
)
