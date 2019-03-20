def insert_sort(arr):
    '''插入排序'''
    length = len(arr)
    for i in range(1,length):
        x = arr[i]
        for j in range(i,-1,-1):
            if x < arr[j-1]:
                arr[j] = arr[j-1]
            else:
                break
        arr[j] = x


arr = [8, 7, 6, 5, 4, 3, 2]
insert_sort(arr)
print(arr)