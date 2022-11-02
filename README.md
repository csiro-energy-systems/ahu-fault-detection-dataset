# ahu-fault-detection-dataset

Dataset containing fault-generation experiments on two of CSIRO Energy's Newcastle Office building Air Handling Units (AHU) system.
This is intended to provide a reference dataset for the development and comparison of new fault detection and diagnosis (FDD) algorithms for Heating Ventilation and Air Conditioning (HVAC) systems.

This dataset was generated by inserting faults into the Building Management System (BMS) of an operational 4-storey office building in Newcastle, Australia in April-November 2013.

Fault types generated include:

* Leaking supply-air duct
* Slipping supply-air fan belt
* Stuck/leaking hot water valve
* Stuck/leaking chilled water valve
* Stuck/leaking outside air damper
* Stuck/leaking return air damper
* Average zone air temperature sensor offset
* Single zone air temperature sensor failure

Faults were designed for comparing standardised rule-based Fault Detection and Diagnosis (FDD) rules to more advanced techniques.

These faults were generated by adding 'proxy' points into the BMS code which can override the control signals sent to actuators or received from sensors, while masking the real values from the BMS's
control logic. This diagram sums up the changes made to the BMS to simulate these faults:

![BMS Changes](docs/BMS Changes.png)

The actual fault type and magnitude was randomised by custom fault generation software. A summary of the faults generated is below.
![fault-summary](docs/fault-summary.png)

AHU 9 & 10 Schematic
![ahu9-schematic](docs/ahu9-schematic.png)

## Referencing

If you use this dataset in your research or publications, please cite the following paper:

`Guo, Ying; Wall, Josh; Li, Jiaming; West, Sam. Intelligent Model Based Fault Detection and Diagnosis for HVAC System Using Statistical Machine Learning Methods. In: ASHRAE 2013 Winter Conference; January 26 - 30, 2013; Dallas, USA. ASHRAE; 2013. 119-124. http://hdl.handle.net/102.100.100/98252?index=1`

See [here](http://hdl.handle.net/102.100.100/98252)
or [here](https://www.researchgate.net/publication/262640090_Intelligent_Model_Based_Fault_Detection_and_Diagnosis_for_HVAC_System_Using_Statistical_Machine_Learning_Methods) for the paper.

## License

See [LICENSE](LICENSE) for code license.
Data is licensed under the (Creative Commons Attribution-ShareAlike 4.0 International Public License)[http://creativecommons.org/licenses/by-sa/4.0/].

## Quick start:

If you just want to access the data, do this:

```shell
git clone <this-repo-url>
git lfs install
git lfs pull
```

After that, you should have the data in the `data/` directory.

If you also want to run the notebook examples, do this:

```shell
docker-compose up --build
# Then, open the notebook in your browser by clicking one of the `http://127.0.0.1:8888` urls printed.
```

If cloning this to set up the environment for development, run these commands:

```shell
git clone <this-repo.git>
cd  <this-repo-url>

# We recommend using `pyenv` to manage multiple python versions. If pyenv is installed, just run:
pyenv install 3.9.13 # or whatever >=3.9 version you'd prefer
pyenv local 3.9.13 

# ...OR if you don't use pyenv, run:
poetry env use c:\path\to\python3.9\python.exe # Windows  
poetry env use /path/to/python39 # Linux/Mac

# install libraries
poetry install

# install pre-commit hooks for code quality etc
pre-commit install

# run tests locally
pytest

```

Other useful commands include:

```shell

# generate html doc
scripts\gen-doc # Windows
scripts/gen-doc # Linux/Mac

# Add private pypi index for adding and publishing packages
poetry source add --secondary csiroenergy https://pkgs.dev.azure.com/csiro-energy/csiro-energy/_packaging/csiro-python-packages/pypi/simple/
poetry config repositories.csiroenergy https://pkgs.dev.azure.com/csiro-energy/csiro-energy/_packaging/csiro-python-packages/pypi/upload
poetry config http-basic.csiroenergy YOUR_IDENT YOUR_TOKEN # Generate a token at https://dev.azure.com/csiro-energy/_usersSettings/tokens 

# build a redistributable python package
poetry build

# publish a built wheel to the private csiroenergy pypi repo. Only works once for each unique project version (in pyproject.toml).
poetry publish -r csiroenergy

# build a docker image, and define two containers, one for tests, and one for launching your code
docker-compose build

# run tests in docker container
docker-compose run test
```



