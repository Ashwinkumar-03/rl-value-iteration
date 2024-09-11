# VALUE ITERATION ALGORITHM

## AIM
Write the experiment AIM.

## PROBLEM STATEMENT
Explain the problem statement.

## POLICY ITERATION ALGORITHM
Include the steps involved in the value iteration algorithm

## VALUE ITERATION FUNCTION
### Name: Ashwin Kumar S
### Register Number: 212222240013
```
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward+gamma*V[next_state]*(not done))
        if np.max(np.abs(V-np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)
    pi= lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return V, pi

```


## OUTPUT:
### optimal policy
![image](https://github.com/user-attachments/assets/5259df0b-648d-499c-8626-39341213a121)

### optimal value function 
![image](https://github.com/user-attachments/assets/5f9e241a-0017-4bc8-91bb-fa27c40a4227)

### success rate for the optimal policy.
![image](https://github.com/user-attachments/assets/8eba4db7-7d58-4d03-bcd2-c483068002eb)

## RESULT:

Write your result here
