# The Inverse Problem for Cardiac Arrhythmias

This is the code repository to accompany the article:

***The Inverse Problem for Cardiac Arrhythmias***. *Thomas M. Bury, Khady Diagne, David Olshan, Leon Glass, Alvin Shrier, Bruce Lerman, Gil Bub.*

## Reproducible run on Code Ocean
The article is accompanied by a [code capsule](https://codeocean.com/capsule/1839795/tree/v2) on Code Ocean, which provides a software environment and compute resources to do a reproducible run of the results reported in the paper. This removes the need to install the software environment yourself to reproduce the results.

## Instructions to reproduce results on a local machine

- Clone the repository [~1min]
  ```
  git clone git@github.com:ThomasMBury/cardiac-inverse.git
  ```

- Navigate to the repository. Create a virtual environment [<1min]
  ```
  cd cardiac-inverse
  python -m venv venv
  source venv/bin/activate
  ```

- Install package dependencies [~2 min]
  ```
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
  
- Remove all files in `code/output` and `/results` directories to start with a clean slate [<1s]
  ```
  cd code
  ./clear.sh
  ```

- Run the code [~5 min]
  ```
  ./run.sh
  ```

- Results are saved in the ```/results``` directory.
