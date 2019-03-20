def merge(left, right):

    i, j = 0, 0
    result = []

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]

    return result


def merge_sort(lists):
    '''归并排序'''

    if len(lists) <= 1:
        return lists

    num = len(lists) // 2    #取整
    left = merge_sort(lists[:num])
    right = merge_sort(lists[num:])

    return merge(left, right)


arr = [8, 7, 6, 5, 4, 3, 2]
sorted_arr = merge_sort(arr)
print(sorted_arr)