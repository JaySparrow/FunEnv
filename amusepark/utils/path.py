import os

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(cur_path[:cur_path.find('/amusepark')], 'amusepark')
data_path = os.path.join(root_path, 'data')

if __name__ == '__main__':
    print(cur_path)
    print(root_path)
    print(data_path)