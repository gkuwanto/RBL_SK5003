def read_csv(filename):
    with open(filename) as f:
        headers = f.readline().strip().split(',')
        data = { col: [] for col in headers }
        for line in f.readlines():
            values = (line.strip().split(','))
            for col, val in zip(headers, values):
                data[col].append(val)
        return data