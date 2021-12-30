class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # m과 n에 입력 리스트의 길이 각각 저장
        m = len(nums1)
        n = len(nums2)
        
        # nums3 리스트에 입력 리스트 병합
        nums3 = nums1 + nums2
        # nums3 리스트 정렬
        nums3.sort()
        
        # nums3 리스트의 길이 - 개수
        mn = m + n
        
        if mn % 2 == 1 : # nums3 리스트의 개수가 홀수일때
            for i in range(mn // 2) : # 중간 값 제외 양 옆의 모든 값 제거
                nums3.pop(0)
                nums3.pop(-1)
            target_num = nums3[0] # target_num 에 중간값 저장
        elif mn % 2 == 0 : # nums3 리스트의 개수가 짝수일때
            tmp = (mn // 2) - 1 # 중간 값 후보 두개 제외 모든 값 제거
            for i in range(tmp) :
                nums3.pop(0)
                nums3.pop(-1)
            tmp_num = nums3[0] + nums3[1]
            target_num = tmp_num / 2 # target_num 에 중간 값 후보의 평균 저장
        
        return float(target_num) # target_num의 실수형 출력