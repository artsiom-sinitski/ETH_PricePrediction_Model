
# ETH_PricePrediction_Model

Predict Ethereum token price using time series forecasting LSTM Neural Network model

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Contributing](#contributing)
- [Team](#team)
- [FAQ](#faq)
- [Support](#support)
- [License](#license)

## Installation

#### Install Python 3.7+
#### Install the following Python packages:
- Keras (DL high level framework)
- Tensorflow (DL backend)
- pandas (data manipulation)
- scikit-learn (model crossvalidation)
- matplotlib (data visualization)

## Clone

- Clone this repo to your local machine using `https://github.com/artsiom-sinitski/ETH_PricePrediction_Model.git`

## Features

## Usage

- First, prepare the data for the model. In the command line type: 
```
$> python prepData.py
```
- Second, train the model by running from the command line:
```
$> python trainModel.py default 120 1
```

  'default' is the name of your model  
  '120' is the number of epochs for training  
  '1' is number of batches  

- Third, make the model predict by running from the command line:
```
$> python predictPrice.py default 120 1
```
## Team

| <a href="https://github.com/artsiom-sinitski" target="_blank">**Artsiom Sinitski**</a> |
| :---: |
| [![Artsiom Sinitski](https://github.com/artsiom-sinitski)](https://github.com/artsiom-sinitski)|
| <a href="https://github.com/artsiom-sinitski" target="_blank">`github.com/artsiom-sinitski`</a> |
- You can just grab their GitHub profile image URL
- You should probably resize their picture using `?s=200` at the end of the image URL.

## FAQ

## Support

Reach out to me at the following places:
- <a href="https://github.com/artsiom-sinitski" rel="noopener noreferrer" target="_blank">GitHub account</a>
- <a href="https://www.instagram.com/artsiom_sinitski/" rel="noopener noreferrer" target="_blank"> Instagram account</a>

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright ©2019 
