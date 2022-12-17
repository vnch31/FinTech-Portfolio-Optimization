
# FinTech Portfolio Optimization

It's a Group Project for course Financial Technology (EE5183 / 111-1) - National Taiwan University (NTU) applying Deep Learning for portfolio management. Its goal is to provide a tool to help investors applying their strategies to acquire assets weight allocation using various Deep Learning model approach and hyper-parameters.

Team-7 Members:
* 南維克 (VICTOR JUSTIN SERGE NANCHE)
* 沐西門 (SIMON YVES JEAN MOULIN)
* 葛 丁 (GLADHI GUARDDIN)
* ??? (ZOE EDEN)


#  Getting Started

Install in Linux/Unix
> git clone https://github.com/vnch31/FinTech-Portfolio-Optimization.git
> cd FinTech-Portfolio-Optimization/
> python3 -m venv env
> source env/bin/activate
> pip install -r requirements.txt

Running in CLI:
> python main.py

Available parameters:
```
# python main.py --help
usage: main.py [-h] [-c CONFIG] [-m MODELSCONFIG] [-n NAME] [-a AUTOTICKERS] [-t TICKERS [TICKERS ...]] [-s START] [-e END] 
               [-i INTERVAL] [-r RETRAIN] [--timestep TIMESTEP] [-d DEVICE]

Dataloader options, if no argument given config.json and models_config.json will be used

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Config File (default: config.json)
  -m MODELSCONFIG, --modelsconfig MODELSCONFIG
                        Models Config File (default: models_config.json)
  -n NAME, --name NAME  Name of the model
  -a AUTOTICKERS, --autotickers AUTOTICKERS
                        Using tickers from Sentiment Analysis (Max 6), required: --start & --end date
  -t TICKERS [TICKERS ...], --tickers TICKERS [TICKERS ...]
                        Tickers to use, required: start & end
  -s START, --start START
                        Start date : YYYY-MM-DD
  -e END, --end END     End date : YYYY-MM-DD
  -i INTERVAL, --interval INTERVAL
                        Fetch data by interval
  -r RETRAIN, --retrain RETRAIN
                        Retrain interval
  --timestep TIMESTEP   Timestep for the weight prediction
  -d DEVICE, --device DEVICE
                        Device to use

```
## 

- abc
- abc
- abc
- abc
- abc
- abc

