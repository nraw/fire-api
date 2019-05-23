Bringing fire to your model... right!

Call 

```make run```

for starting the API. You can then access the model either through API, CLI or directly from python.

API:

```http http://127.0.0.1:8000/predict p1=Petra p2=Gayane p3=Igor p4=Felipe```

CLI:

```python3 api.py predict Petra Gayane Igor Felipe```

Python:

```
from api import predict
predict('Petra', 'Gayane', 'Igor', 'Felipe')
```
