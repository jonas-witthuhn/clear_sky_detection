from setuptools import setup
setup(
    name="clear_sky_detection",
    version="0.1",
    description="Clear sky detection from broadband irradiance measurements.",
    url="https://github.com/jonas-witthuhn/clear_sky_detection-base",
    license="CC BY-NC",
    author="Jonas Witthuhn",
    author_email="witthuhn@tropos.de",
    packages=["clear_sky_detection"],
    package_dir={"":"src"},
    install_requires=["numpy",
                      "scipy",
                      ]
        )
