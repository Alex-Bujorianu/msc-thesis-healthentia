To be able to run the code, setup a virtual environment using pip and install all the required dependencies from requirements.txt:

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
Make sure to run the code with your virtualenv interpreter and not your system Python interpreter. 

This is necessary because the libraries used by this repository are known to introduce breaking changes and dependency conflicts. The version numbers have been frozen at the time of publication in order to ensure reproducibility.

Any questions should be forwarded to my a.bujorianu@student.utwente.nl email.