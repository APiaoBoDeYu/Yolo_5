
nums=[0,1,0,3,12]
i ,j ,numb =0 ,0 ,0
while j< len(nums) :
    if nums[j] != 0:
        nums[i] = nums[j]
        i += 1
        j += 1
    else:
        j += 1
        numb += 1
for k in range(0, numb):
    nums[i + k] = 0
