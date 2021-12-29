class Solution(object):
    def lengthOfLongestSubstring(self, s):
        # 중복되지 않은 문자열 중 가장 긴 문자열 구하기
        """
        :type s: str
        :rtype: int
        """
        con_l = [] # 비교를 진행할 리스트
        target_l = [] # 정답 후보를 입력할 리스트
        cnt = 1 # 반복 횟수 카운트
        
        for i in str(s) :
            if cnt == 1 : # 첫 번째 반복일 경우 fisrst_let에 첫 번째 문자를 저장하고 비교 리스트에 문자 저장
                con_l.append(i)
                first_let = i
                cnt += 1
            else : # 두 번째 이상 반복에서 시행할 명령
                next_let = i # next_let에 값 저장
                if i not in con_l : # 비교 리스트에 값이 존재하지 않을 경우
                    con_l.append(i) # 비교 리스트에 값 추가
                    cnt += 1
                elif i in con_l : # 비교 리스트에 이미 값이 존재할 경우
                    if first_let == next_let : # 만약 동일한 글자가 두 개 붙어있을 경우
                        con_l = [] # 비교 리스트를 초기화
                        con_l.append(i) # 비교 리스트에 문자 추가
                        cnt += 1
                    else : # 동일한 글자가 서로 떨어져 있을 경우
                        ind_num = con_l.index(i) # 중복된 문자의 첫 번째 인덱스
                        for y in range(0, ind_num + 1) : # 중복된 문자가 나타난 곳까지 비교 리스트에서 삭제
                            con_l.pop(0)
                        con_l.append(i) # 비교 리스트에 값 추가
                        cnt += 1
                first_let = next_let # 비교 대상 값에 비교 했던 값 저장
            target_l.append(len(con_l)) # 정답 후보 리스트에 비교 리스트의 길이 저장
                
        cnt = 1
        first_num = 0 # 첫 번째 숫자를 0 으로 지정(빈 글자를 입력했을 경우의 디폴트 값)
        for j in target_l : 
            if cnt == 1 : # 첫 번째 반복일 경우
                first_num = j # first_num 에 값 저장
                cnt += 1
            next_num = j # 두 번째 반복 이상
            if first_num >= next_num : # first_num이 더 크거나 같을 경우 해당 반복 스킵
                cnt += 1
            elif next_num > first_num : # next_num이 더 클경우
                first_num = next_num # first_num과 next_num 스위칭
                cnt += 1

        
        return first_num # 정답 후보 리스트 중에서 가장 큰 값 ( 가장 긴 문자열의 길이 ) 반환
    

            

