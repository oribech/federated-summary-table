# Federated-summary-table

## File Descriptions

1. `testing_simulations.py`: This file contains a TestingSimulator class that handles the execution of simulations for
   testing under various configurations.

2. `config.py`: This file defines a Configurator class that holds configuration parameters for the simulations.

3. `estimation_simulations.py`: This file contains an EstimationSimulator class that handles the execution of
   simulations for estimation under various configurations.

4. `binning.py`: This file defines a Binning class that is responsible for creating bins from a given dataset using a
   privacy-preserving algorithm.

5. `utils.py`: This file contains a collection of utility functions used across the other scripts. These include
   functions for calculations related to the Yeo-Johnson transformation, the mixed normal and gamma quantiles, and other
   statistical and utility functions.

6. `federated_binning.py`: This file contains functionality for federated binning of data across multiple centers while
   preserving privacy.

## Usage

The primary classes to use are the TestingSimulator and EstimationSimulator classes, both of which run simulations under
various configurations. The configuration parameters can be adjusted in the Configurator class. The Binning class is
used to bin data in a privacy-preserving way, and the federated_binning file provides a way to bin data across multiple
centers.

## Dependencies

See requirements.txt for a list of dependencies.

## Installation

Clone this repository and import the necessary Python files to use the TestingSimulator, EstimationSimulator, and
Binning classes in your own projects.

## License

This project is licensed under the terms of the MIT license.

