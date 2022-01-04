class Solution:
    def longestPalindrome(self, s: str) -> str: # 가장 긴 대칭 문자열 구하기
        cnt = 1 # 카운트
        target_str = "" # 대칭 문자열
        target_str_list = [] # 대칭 문자열 리스트
        target = "" # 가장 긴 대칭 문자열
        rev_str_list1 = [] # 역순 문자열 리스트
        rev_str1 = "" # 역순 문자열
        rev_str_list2 = [] # 역순 문자열 리스트 2
        rev_str2 = "" # 역순 문자열 리스트 2
        for x in range(0, len(str(s)) - 1) : # 슬라이싱 x 는 0부터 문자열 길이 -1까지
            for y in range(x + 1, len(str(s)) + 1) : # 슬라이싱 y는 x + 1 부터 문자열 끝까지
                if len(s[x:y]) % 2 == 0  : # 비교 대상 문자열이 짝수일 때
                    for i in s[y : y * 2 - x ] : # 바로 붙어있는 문자열 역순
                        rev_str_list1.insert(0,i)
                    for j in rev_str_list1 :
                        rev_str1 += j
                    for a in s[y + 1 : y * 2 - x + 1] : # 한 문자 건너 있는 문자열 역순
                        rev_str_list2.insert(0,a)
                    for b in rev_str_list2 :
                        rev_str2 += b
                
                    if s[x:y] == rev_str1 :  # 바로 붙어있는 문자열이랑 대칭이면
                        target_str = s[x:y] + s[y : y * 2 - x ]
                        target_str_list.append(target_str) # 대칭 문자열 리스트에 등록
                    if s[x:y] == rev_str2 : # 떨어져 있는 문자열이랑 대칭이면
                        target_str = s[x:y*2 - x + 1]
                        target_str_list.append(target_str) # 대칭 문자열 리스트에 등록
                elif len(s[x:y]) % 2 == 1 : # 비교 대상 문자열이 홀수일 때
                    for i in s[y + 1 : y * 2 - x + 1] : # 한 문자 건너 있는 문자열 역순
                        rev_str_list1.insert(0,i)
                    for j in rev_str_list1 :
                        rev_str1 += j
                    for a in s[y : y * 2 - x] : # 바로 붙어 있는 문자열 역순
                        rev_str_list2.insert(0,a)
                    for b in rev_str_list2 :
                        rev_str2 += b
                    if s[x:y] == rev_str1 : # 한 문자 건너 있는 문자열이랑 대칭일 때
                        target_str = s[x:y*2 - x + 1]
                        target_str_list.append(target_str)
                    if s[x:y] == rev_str2 : # 바로 붙어 있는 문자열이랑 대칭일 때
                        target_str = s[x:y*2 - x]
                        target_str_list.append(target_str)
                rev_str_list1, rev_str_list2, rev_str1, rev_str2 = [], [], "", "" # 역순 문자열, 리스트 초기화
                cnt += 1 


        cnt = 1
        for i in target_str_list : # 대칭 문자열 리스트 길이 최댓값 찾기
            if cnt == 1 :
                first_num = len(i)
                cnt += 1
            else :
                next_num = len(i)
                if first_num >= next_num : 
                    cnt += 1
                elif next_num > first_num :
                    first_num = next_num
                    cnt += 1
     
        for i in target_str_list : # 최댓값의 문자 출력
            if len(i) == first_num :
                target = i

        if target == "" : # 입력 문자가 없거나 하나일 떄의 디폴트 값"
            target = s[0]


        return target # 타겟 반환

    

    


