<h1 align="center">Reinforcement trading</h1>

<p align="center">
    <strong>Implement reinforcement learning algorithms such as DQN, PPO, and A2C in trading to optimize profits and reduce losses.</strong>
    <br />
    <br />
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#config">Config</a> • 
    <a href="#features">Features</a> •
    <a href="#contact">Contact</a> •
</p>

<hr />

<h2 id="installation">📁<ins>Installation</ins></h2>
<ul>
    <li><b>Step 1 : </b>Clone the repo.
        <pre><code>git clone https://github.com/golderalex6/Reinforcement-trading.git</code></pre>
    </li>
    <li><b>Step 2 : </b>Install dependencies.
        <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li><b>Step 3 : </b>Setup folders , .json files.
        <pre><code>python setup.py</code></pre>
    </li>
</ul>

<h2 id="usage">📈<ins>Usage</ins></h2>
<ul>
<li>
    <div><b>Gather data for training:</b></div>
<pre><code>from get_stock_data import StockData<br>
companies={
        'TSLA':'NASDAQ',
        'META':'NASDAQ',
        'AAPL':'NASDAQ',
    }
stock=StockData()
stock.get_data(companies)
</code></pre>
</li>
<li>
    <div><b>Train models (For example: with A2C,like PPO and DQN):</b></div>
<pre><code>from a2c import A2cTrading<br>
a2c = A2cTrading()
a2c.learn(total_timsteps = 40000)
a2c.load()
a2c.evaluate()
</code></pre>
</li>
</ul>

<h2 id="config">⚙️<ins>Config</ins></h2>
<ul>
<li>
    <div><b>Model hyperparameters : </b>All model hyperparameters are stored in the metadata folder, with each hyperparameter file named &ltmodel&gt_metadata.json .</div>
    <br>
<pre><code>{
    "layers": [100,50,20,10,5],
    "activation": "LeakyReLU",
    "optimizer": "Adam",
    "learning_rate": 0.0003,
    "policy": "MlpPolicy"
}
</code></pre>
</li>
<li>
    <div><b>Training parameters : </b>Training data parameters (e.g., symbol, start_date, end_date, etc.) are saved in parameters.json.</div>
<pre><code>{
    "symbol":"META",
    "start_date":"2018-01-01",
    "test_date":"2022-01-01",
    "end_date":"2024-01-01",
    "window_size":5
}
</code></pre>
</li>
<li>
    <div><b>Custom dataset : </b>For a custom dataset, ensure the dataframe includes at least the following columns: ['Datetime', 'Timestamp', 'Timeframe', 'Open', 'High', 'Low', 'Close', 'Volume']. Place your .csv file as follows:</div>
<pre><code>
|data
    |dataset_name
        |&ltdataset_name&gt_1d.csv
</code></pre>
Then change "symbol":"&ltdataset_name&gt" in parameters.json for model to read and train with it.
</li>
<li>
    <div><b>Pre-trained model : </b>For a pre-trained model, either place the .zip file in the models folder or provide its path to the load method.(Example with PPO)</div>
</li>
<pre><code>from ppo import PpoTrading</br>
ppo = PpoTrading()
ppo.load() #or ppo.load(path)
</code></pre>
<small><i>*Running `setup.py` initializes all configurations to default values. You can change the symbol for training or trading in `parameters.json`, but avoid modifying other values unless you fully understand them to prevent errors.</i></small>
</ul>


<h2 id="features">📜<ins>Features</ins></h2>
<ul>
    <li><b>Automated Stock Data Collection : </b>Fetch stock data programmatically for multiple companies using the StockData module.</li>
    <li><b>Reinforcement Learning Models: : </b>Implements advanced RL algorithms like DQN, PPO, and A2C.</li>
    <li><b>Custom Dataset Support : </b>Compatible with custom datasets containing essential trading columns.</li>
    <li><b>Configurable Hyperparameters : </b>Fully customizable model hyperparameters via metadata JSON files.</li>
</ul>

<h2 id="contact">☎️<ins>Contact</ins></h2>
<p>
    Golderalex6 <a href="mailto:golderalex6@gmail.com">golderalex6@gmail.com</a><br>
    Project Link: <a href="https://github.com/golderalex6/Reinforcement-trading">https://github.com/golderalex6/Reinforcement-trading</a>
</p>

<hr />

