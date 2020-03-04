# Feed-Forward-Neural-Network-3-Layers

## This code is mathematically explained in my Medium article
https://medium.com/swlh/mathematics-behind-basic-feed-forward-neural-network-3-layers-python-from-scratch-df88085c8049

## Requirements
Only Numpy !

## RUN
```bash
git clone https://github.com/MarvinMartin24/
```
Go to your file directory, and run this command :
```bash
python3 main.py
```
## Usage
```python
fnn = FFNN(2, 4, 1)
fnn.fit(x_train, y_train)
print('Input',x_train)
print()
print('Predition', np.around(fnn.predict(x_train)))
print('Label', y_train)
```
## Implementation : XOR
* Input / Data:


| A | B | XOR |
| --- | --- |--- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

```python3
x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
y_train = np.array([[0], [1], [1], [0]])
```

