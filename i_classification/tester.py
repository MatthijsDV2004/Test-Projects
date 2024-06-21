import numpy as np

# Provided values
values = [
    [5.5965846e-03, 3.7116774e-03, 2.7332406e-03, 8.5355592e-04, 1.8838307e-02, 5.2351006e-03],
    [2.0730880e-03, 2.3089381e-02, 4.7285192e-07, 4.2008278e-08, 9.7223467e-01, 2.5929408e-03],
    [3.0675355e-01, 2.6285596e-02, 2.1885870e-02, 1.3057556e-03, 5.0067860e-01, 1.0265889e-01],
    # More rows...
]

# Find the maximum number of elements in a row
max_length = max(len(row) for row in values)

# Create a new matrix with zeros and fill in the provided values
matrix = np.zeros((len(values), max_length))
for i, row in enumerate(values):
    matrix[i, :len(row)] = row

print(matrix)
