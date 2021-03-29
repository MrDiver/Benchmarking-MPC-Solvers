# RoboticsIPBenchmark

## Installation and Usage

To install the MPCBenchmark package you need to type the following command in the root directory of this repository.

```shell
$ python -m pip install .
```

After a successfull install you will be able to run the basic test script to execute the Agents in an Environment.

```shell
$ python scripts/experiment_test.py
```

# Data of the experiments is publicly available

The data is stored on a MongoDB Database and can be accessed through the following details with compass or directly in python

```
URI: mongodb+srv://cluster0.5qs1s.mongodb.net/
Username: mpcbenchmark
Password: mpcbenchmark

Compass URI: mongodb+srv://mpcbenchmark:mpcbenchmark@cluster0.5qs1s.mongodb.net
```

i would advise using `mongodump` and `mongorestore` to make a local copy of the database to prevent it from being closed down due to high traffic.

```bash
$ mongodump -uri mongodb+srv://cluster0.5qs1s.mongodb.net/ --username dbUser --gzip --out /path/to/local/copy
$ mongorestore --host 127.0.0.1 --port 27017 --gzip /path/to/local/copy
```
