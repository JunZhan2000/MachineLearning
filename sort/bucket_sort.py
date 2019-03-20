def bucket_sort(nums):
    '''桶排序'''

    max_num = max(nums)
    bucket = [0]*(max_num+1)
    for i in nums:
        bucket[i] += 1

    sort_nums = []
    for j in range(len(bucket)):
        if bucket[j] != 0:
            for y in range(bucket[j]):
                sort_nums.append(j)

    return sort_nums

arr = [8, 7, 6, 5, 4, 3, 2]
sorted_arr = bucket_sort(arr)
print(sorted_arr)