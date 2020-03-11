# define the subsets function to find all possible subsets and do no include null subset
def subsets(nums):
    results = []
    # return [] set if the collection is empty
    if not nums:
        return results

    # sort the array to avoid duplicate subsets
    nums.sort()
    length = len(nums)

    # SubsetsHelper function helps us to find subset recursively
    def SubsetsHelper(startIdx, length, subset):
        # check if the subset is not in the results and not a null subset
        if (subset not in results and subset):
            results.append(subset)
        # recursive call SubsetsHelper function
        for i in range(startIdx, length):
            # Increase the start index 1 every time
            SubsetsHelper(i + 1, length, subset + [nums[i]])

    # call the SubsetsHelper function
    SubsetsHelper(0, length, [])
    return results


collection = [1, 2, 2]
subsetsNoNull = subsets(collection)
print('Input:', collection)
print('Output:', subsetsNoNull)
